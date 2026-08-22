import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import uuid

# ============ TOPOLOGY TYPES ============
class TopologyType(Enum):
    TREE = "tree"
    GRAPH = "graph"
    CYCLIC = "cyclic"
    DAG = "dag"
    STAR = "star"
    MESH = "mesh"
    HIERARCHICAL = "hierarchical"

# ============ BASE NODE ============
@dataclass
class TopologyNode:
    """Base node in any object topology"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Node"
    metadata: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[float, float] = (0, 0)  # Position in 2D space
    
    def __hash__(self):
        return hash(self.id)
    
    def __repr__(self):
        return f"{self.name}({self.id})"

# ============ RELATIONSHIP TYPES ============
class RelationshipType(Enum):
    OWNS = "owns"
    CONTAINS = "contains"
    REFERENCES = "references"
    DEPENDS_ON = "depends_on"
    INHERITS_FROM = "inherits_from"
    IMPLEMENTS = "implements"
    COMPOSED_OF = "composed_of"
    ASSOCIATED_WITH = "associated_with"
    ESCAPES_TO = "escapes_to"
    BOUNDARY_OF = "boundary_of"
    CONNECTS_TO = "connects_to"

@dataclass
class Relationship:
    source: TopologyNode
    target: TopologyNode
    rel_type: RelationshipType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============ TOPOLOGY MANAGER ============
class ObjectTopology:
    """Manages object relationships and traversals in various topologies."""
    
    def __init__(self, topology_type: TopologyType = TopologyType.GRAPH):
        self.topology_type = topology_type
        self.nodes: Dict[str, TopologyNode] = {}
        self.relationships: List[Relationship] = []
        self._adjacency: Dict[str, Set[str]] = {}
        self._reverse_adjacency: Dict[str, Set[str]] = {}
        self._sub_topologies: List[ObjectTopology] = []  # For hierarchical topology
    
    def add_node(self, node: TopologyNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists")
        self.nodes[node.id] = node
        self._adjacency[node.id] = set()
        self._reverse_adjacency[node.id] = set()
    
    def add_nodes(self, nodes: List[TopologyNode]) -> None:
        """Add multiple nodes at once"""
        for node in nodes:
            self.add_node(node)
    
    def add_relationship(self, source_id: str, target_id: str, 
                        rel_type: RelationshipType = RelationshipType.REFERENCES,
                        weight: float = 1.0) -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            # Check if nodes exist in sub-topologies
            if self._sub_topologies:
                for sub_top in self._sub_topologies:
                    if source_id in sub_top.nodes or target_id in sub_top.nodes:
                        # If either node is in a sub-topology, add it to main topology
                        if source_id in sub_top.nodes and source_id not in self.nodes:
                            self.add_node(sub_top.nodes[source_id])
                        if target_id in sub_top.nodes and target_id not in self.nodes:
                            self.add_node(sub_top.nodes[target_id])
                        break
                else:
                    raise ValueError(f"Both nodes must exist. source={source_id}, target={target_id}")
            else:
                raise ValueError(f"Both nodes must exist. source={source_id}, target={target_id}")
        
        rel = Relationship(self.nodes[source_id], self.nodes[target_id], rel_type, weight)
        self.relationships.append(rel)
        self._adjacency[source_id].add(target_id)
        self._reverse_adjacency[target_id].add(source_id)
    
    def add_sub_topology(self, sub_topology: 'ObjectTopology') -> None:
        """Add a sub-topology to this topology (for hierarchical structures)"""
        self._sub_topologies.append(sub_topology)
        # Add all nodes from sub-topology to this topology
        for node in sub_topology.nodes.values():
            if node.id not in self.nodes:
                self.add_node(node)
    
    def get_neighbors(self, node_id: str) -> List[TopologyNode]:
        """Get all neighbors of a node"""
        neighbors = set()
        neighbors.update(self._adjacency.get(node_id, set()))
        neighbors.update(self._reverse_adjacency.get(node_id, set()))
        return [self.nodes[nid] for nid in neighbors if nid in self.nodes]
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "relationship_count": len(self.relationships),
            "topology_type": self.topology_type.value,
            "sub_topologies": len(self._sub_topologies)
        }
    
    def clear(self):
        """Clear all nodes and relationships"""
        self.nodes.clear()
        self.relationships.clear()
        self._adjacency.clear()
        self._reverse_adjacency.clear()

# ============ MANDELBROT SET COMPUTATION ============

class MandelbrotTopology:
    """
    Creates a topology from Mandelbrot set computations.
    Points in the complex plane are nodes, connected based on their
    escape trajectories and relationships.
    """
    
    def __init__(self, width: int = 400, height: int = 400, 
                 x_range: Tuple[float, float] = (-2.5, 1.5),
                 y_range: Tuple[float, float] = (-1.5, 1.5),
                 max_iter: int = 100):
        self.width = width
        self.height = height
        self.x_range = x_range
        self.y_range = y_range
        self.max_iter = max_iter
        self.topology = ObjectTopology(TopologyType.GRAPH)
        self.escape_times = None
        self.mandelbrot_set = None
        
    def compute_mandelbrot(self) -> np.ndarray:
        """Compute the Mandelbrot set"""
        x = np.linspace(self.x_range[0], self.x_range[1], self.width)
        y = np.linspace(self.y_range[0], self.y_range[1], self.height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.zeros_like(C, dtype=complex)
        mandelbrot = np.zeros(C.shape, dtype=int)
        
        for i in range(self.max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            mandelbrot[mask] += 1
        
        # Points that never escape are in the Mandelbrot set
        self.mandelbrot_set = np.abs(Z) <= 2
        self.escape_times = mandelbrot
        return mandelbrot
    
    def create_topology_from_mandelbrot(self, sample_rate: int = 5,
                                       connect_boundary: bool = True,
                                       connect_escape_paths: bool = True) -> ObjectTopology:
        """
        Create a topology from the Mandelbrot set.
        
        Args:
            sample_rate: How many points to sample (1 = all points)
            connect_boundary: Connect points on the boundary of the Mandelbrot set
            connect_escape_paths: Connect points based on escape trajectories
        """
        if self.escape_times is None:
            self.compute_mandelbrot()
        
        # Clear previous topology
        self.topology.clear()
        
        x = np.linspace(self.x_range[0], self.x_range[1], self.width)
        y = np.linspace(self.y_range[0], self.y_range[1], self.height)
        
        # Sample points
        sampled_points = []
        for i in range(0, self.height, sample_rate):
            for j in range(0, self.width, sample_rate):
                if self.mandelbrot_set[i, j] or self.escape_times[i, j] > 10:
                    z = x[j] + 1j * y[i]
                    escape_time = self.escape_times[i, j]
                    is_in_set = self.mandelbrot_set[i, j]
                    
                    node = TopologyNode(
                        name=f"P({z.real:.3f},{z.imag:.3f})",
                        position=(z.real, z.imag),
                        metadata={
                            "escape_time": escape_time,
                            "in_mandelbrot": is_in_set,
                            "complex": z,
                            "grid_i": i,
                            "grid_j": j
                        }
                    )
                    self.topology.add_node(node)
                    sampled_points.append((i, j, node))
        
        # Connect nodes based on relationships
        if connect_boundary and sampled_points:
            self._connect_boundary_nodes(sampled_points)
        
        if connect_escape_paths and sampled_points:
            self._connect_escape_paths(sampled_points)
        
        # Connect nearby points with similar escape times
        if sampled_points:
            self._connect_similar_escape_times(sampled_points)
        
        return self.topology
    
    def _connect_boundary_nodes(self, points: List[Tuple[int, int, TopologyNode]]):
        """Connect nodes that are on the boundary of the Mandelbrot set"""
        boundary_nodes = []
        
        for i, j, node in points:
            # Check if this is a boundary point
            is_boundary = False
            if node.metadata["in_mandelbrot"]:
                # Check 4-neighborhood
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.height and 0 <= nj < self.width:
                        if not self.mandelbrot_set[ni, nj]:
                            is_boundary = True
                            break
            else:
                # Check if escape time is low (near boundary)
                if node.metadata["escape_time"] < 20:
                    is_boundary = True
            
            if is_boundary:
                boundary_nodes.append(node)
        
        # Connect boundary nodes in a cycle
        if len(boundary_nodes) > 1:
            for idx in range(len(boundary_nodes) - 1):
                self.topology.add_relationship(
                    boundary_nodes[idx].id,
                    boundary_nodes[idx + 1].id,
                    RelationshipType.BOUNDARY_OF,
                    weight=1.0
                )
            # Close the cycle
            self.topology.add_relationship(
                boundary_nodes[-1].id,
                boundary_nodes[0].id,
                RelationshipType.BOUNDARY_OF,
                weight=1.0
            )
    
    def _connect_escape_paths(self, points: List[Tuple[int, int, TopologyNode]]):
        """Connect nodes based on their escape paths"""
        # For each point, find the nearest point with higher escape time
        for i, j, node in points:
            if not node.metadata["in_mandelbrot"]:
                escape_time = node.metadata["escape_time"]
                # Find points with higher escape time (closer to boundary)
                for other_i, other_j, other_node in points:
                    if other_node.id != node.id:
                        other_time = other_node.metadata["escape_time"]
                        if other_time > escape_time and other_time < escape_time + 10:
                            # Check if they're close in space
                            dist = np.sqrt((i - other_i)**2 + (j - other_j)**2)
                            if dist < 5:  # Within 5 pixels
                                self.topology.add_relationship(
                                    node.id,
                                    other_node.id,
                                    RelationshipType.ESCAPES_TO,
                                    weight=1.0 / (dist + 1)
                                )
    
    def _connect_similar_escape_times(self, points: List[Tuple[int, int, TopologyNode]]):
        """Connect points with similar escape times that are close"""
        for idx1, (i1, j1, node1) in enumerate(points):
            for idx2 in range(idx1 + 1, min(idx1 + 20, len(points))):
                i2, j2, node2 = points[idx2]
                
                # Check if close in space
                dist = np.sqrt((i1 - i2)**2 + (j1 - j2)**2)
                if dist < 3:  # Very close
                    time1 = node1.metadata["escape_time"]
                    time2 = node2.metadata["escape_time"]
                    time_diff = abs(time1 - time2)
                    
                    if time_diff < 5:  # Similar escape times
                        in_set1 = node1.metadata["in_mandelbrot"]
                        in_set2 = node2.metadata["in_mandelbrot"]
                        
                        if in_set1 == in_set2:  # Both in or both out
                            self.topology.add_relationship(
                                node1.id,
                                node2.id,
                                RelationshipType.ASSOCIATED_WITH,
                                weight=1.0 / (dist + 1)
                            )

# ============ VISUALIZATION ============

def visualize_mandelbrot_topology(topology: ObjectTopology, 
                                  title: str = "Mandelbrot Set Topology",
                                  figsize: Tuple[int, int] = (14, 10)):
    """Visualize the Mandelbrot topology with relationships"""
    
    if len(topology.nodes) == 0:
        print("No nodes to visualize!")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Mandelbrot set background
    ax1.set_title(title)
    ax1.set_xlabel("Real")
    ax1.set_ylabel("Imaginary")
    ax1.set_aspect('equal')
    
    # Extract node positions and metadata
    positions = []
    in_set = []
    escape_times = []
    
    for node in topology.nodes.values():
        pos = node.position
        positions.append(pos)
        in_set.append(node.metadata.get("in_mandelbrot", False))
        escape_times.append(node.metadata.get("escape_time", 0))
    
    positions = np.array(positions)
    
    if len(positions) > 0:
        # Plot nodes
        colors = ['red' if in_set[i] else 'blue' for i in range(len(positions))]
        sizes = [20 + min(escape_times[i] / 10, 50) for i in range(len(positions))]
        
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], 
                            c=colors, s=sizes, alpha=0.6, edgecolors='none')
        
        # Plot relationships (limit to avoid cluttering)
        max_edges = 500
        edges_to_plot = topology.relationships[:max_edges]
        
        for rel in edges_to_plot:
            if rel.source.id in topology.nodes and rel.target.id in topology.nodes:
                source_pos = topology.nodes[rel.source.id].position
                target_pos = topology.nodes[rel.target.id].position
                
                # Determine color based on relationship type
                if rel.rel_type == RelationshipType.BOUNDARY_OF:
                    color = 'green'
                    alpha = 0.8
                    linewidth = 2
                elif rel.rel_type == RelationshipType.ESCAPES_TO:
                    color = 'orange'
                    alpha = 0.5
                    linewidth = 1
                else:
                    color = 'gray'
                    alpha = 0.3
                    linewidth = 0.5
                
                ax1.plot([source_pos[0], target_pos[0]], 
                        [source_pos[1], target_pos[1]], 
                        color=color, alpha=alpha, linewidth=linewidth)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='In Mandelbrot Set'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, label='Outside Set'),
            Line2D([0], [0], color='green', lw=2, label='Boundary Connection'),
            Line2D([0], [0], color='orange', lw=1, label='Escape Path'),
            Line2D([0], [0], color='gray', lw=0.5, alpha=0.3, label='Similarity')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Plot 2: Escape time heatmap
    ax2.set_title("Escape Times Heatmap")
    ax2.set_xlabel("Real")
    ax2.set_ylabel("Imaginary")
    ax2.set_aspect('equal')
    
    # Create a simple heatmap from the node data
    if len(positions) > 0:
        # Create grid for heatmap
        x_min, x_max = min(p[0] for p in positions), max(p[0] for p in positions)
        y_min, y_max = min(p[1] for p in positions), max(p[1] for p in positions)
        
        # Create grid
        grid_size = 100
        grid_x = np.linspace(x_min, x_max, grid_size)
        grid_y = np.linspace(y_min, y_max, grid_size)
        grid_z = np.zeros((grid_size, grid_size))
        grid_counts = np.zeros((grid_size, grid_size))
        
        # Fill grid with escape times
        for node in topology.nodes.values():
            pos = node.position
            escape_time = node.metadata.get("escape_time", 0)
            xi = int((pos[0] - x_min) / (x_max - x_min) * (grid_size - 1))
            yi = int((pos[1] - y_min) / (y_max - y_min) * (grid_size - 1))
            if 0 <= xi < grid_size and 0 <= yi < grid_size:
                grid_z[yi, xi] += escape_time
                grid_counts[yi, xi] += 1
        
        # Average escape times
        mask = grid_counts > 0
        grid_z[mask] = grid_z[mask] / grid_counts[mask]
        
        # Fill missing values with interpolation
        from scipy.ndimage import gaussian_filter
        grid_z = gaussian_filter(grid_z, sigma=1, mode='constant')
        
        im = ax2.imshow(grid_z.T, origin='lower', 
                       extent=[x_min, x_max, y_min, y_max],
                       cmap='hot', alpha=0.8)
        plt.colorbar(im, ax=ax2, label='Escape Time')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    stats = topology.get_statistics()
    print(f"\nTopology Statistics:")
    print(f"  Nodes: {stats['node_count']}")
    print(f"  Relationships: {stats['relationship_count']}")
    print(f"  Topology Type: {stats['topology_type']}")
    
    # Count relationship types
    rel_types = {}
    for rel in topology.relationships:
        rel_types[rel.rel_type.value] = rel_types.get(rel.rel_type.value, 0) + 1
    
    print("\nRelationship Types:")
    for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel_type}: {count}")

# ============ ADVANCED: MULTI-RESOLUTION MANDELBROT TOPOLOGY ============

class MultiResolutionMandelbrotTopology:
    """Creates a hierarchical topology of Mandelbrot set at multiple resolutions"""
    
    def __init__(self):
        self.hierarchy = ObjectTopology(TopologyType.HIERARCHICAL)
        self.resolutions = {}
        
    def build_hierarchy(self, resolutions: List[int] = [20, 40, 80]):
        """Build a hierarchical topology at multiple resolutions"""
        
        all_nodes = []
        
        for res in resolutions:
            print(f"Computing at resolution {res}x{res}...")
            mt = MandelbrotTopology(
                width=res, 
                height=res,
                x_range=(-2.5, 1.5),
                y_range=(-1.5, 1.5),
                max_iter=50
            )
            
            # Create topology at this resolution
            topology = mt.create_topology_from_mandelbrot(
                sample_rate=1,
                connect_boundary=True,
                connect_escape_paths=False
            )
            
            # Store the topology
            self.resolutions[res] = topology
            
            # Create a meta-node for this resolution
            meta_node = TopologyNode(
                name=f"Resolution_{res}",
                position=(0, 0),
                metadata={
                    "resolution": res, 
                    "nodes": len(topology.nodes),
                    "type": "meta"
                }
            )
            self.hierarchy.add_node(meta_node)
            all_nodes.append((res, meta_node))
            
            # Add all nodes from this resolution to the hierarchy
            for node in topology.nodes.values():
                self.hierarchy.add_node(node)
            
            # Connect meta-node to all nodes in this resolution
            for node in topology.nodes.values():
                self.hierarchy.add_relationship(
                    meta_node.id,
                    node.id,
                    RelationshipType.CONTAINS,
                    weight=1.0
                )
        
        # Connect nodes across resolutions
        self._connect_across_resolutions()
        
        return self.hierarchy
    
    def _connect_across_resolutions(self):
        """Connect nodes across different resolutions based on position"""
        if len(self.resolutions) < 2:
            return
        
        # Get all nodes from all resolutions
        all_nodes = {}
        for res, topology in self.resolutions.items():
            all_nodes[res] = list(topology.nodes.values())
        
        # Connect nodes from different resolutions that are close in space
        res_list = list(self.resolutions.keys())
        for i in range(len(res_list)):
            for j in range(i + 1, len(res_list)):
                res1, res2 = res_list[i], res_list[j]
                nodes1, nodes2 = all_nodes[res1], all_nodes[res2]
                
                # Sample connections to avoid O(n^2)
                sample_size = min(100, len(nodes1), len(nodes2))
                if len(nodes1) > sample_size:
                    import random
                    nodes1_sample = random.sample(nodes1, sample_size)
                else:
                    nodes1_sample = nodes1
                    
                if len(nodes2) > sample_size:
                    nodes2_sample = random.sample(nodes2, sample_size)
                else:
                    nodes2_sample = nodes2
                
                for node1 in nodes1_sample:
                    pos1 = node1.position
                    for node2 in nodes2_sample:
                        pos2 = node2.position
                        dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        
                        # Scale distance by resolution
                        scale = (res1 + res2) / 2
                        scaled_dist = dist * scale
                        
                        if scaled_dist < 0.5 and dist > 0:  # Close in space
                            self.hierarchy.add_relationship(
                                node1.id,
                                node2.id,
                                RelationshipType.CONNECTS_TO,
                                weight=1.0 / (dist + 0.1)
                            )

# ============ MAIN DEMO ============

def demo_mandelbrot_topology():
    print("=" * 70)
    print("MANDELBROT SET OBJECT TOPOLOGY")
    print("=" * 70)
    
    # Create the topology
    print("\nComputing Mandelbrot set and creating topology...")
    mt = MandelbrotTopology(
        width=200,
        height=200,
        x_range=(-2.5, 1.5),
        y_range=(-1.5, 1.5),
        max_iter=80
    )
    
    # Compute Mandelbrot set
    escape_times = mt.compute_mandelbrot()
    points_in_set = mt.mandelbrot_set.sum()
    print(f"Computed Mandelbrot set: {points_in_set} points in set")
    print(f"Total points: {mt.width * mt.height}")
    
    # Create topology with sampling
    topology = mt.create_topology_from_mandelbrot(
        sample_rate=3,
        connect_boundary=True,
        connect_escape_paths=True
    )
    
    # Visualize
    visualize_mandelbrot_topology(
        topology,
        title="Mandelbrot Set Topology",
        figsize=(14, 10)
    )
    
    # Build multi-resolution hierarchy
    print("\n" + "=" * 70)
    print("BUILDING MULTI-RESOLUTION HIERARCHY")
    print("=" * 70)
    
    try:
        mrmt = MultiResolutionMandelbrotTopology()
        hierarchy = mrmt.build_hierarchy(resolutions=[20, 40, 60])
        
        stats = hierarchy.get_statistics()
        print(f"\nHierarchy Statistics:")
        print(f"  Total nodes in hierarchy: {stats['node_count']}")
        print(f"  Total relationships: {stats['relationship_count']}")
        print(f"  Topology Type: {stats['topology_type']}")
        
        # Print resolution details
        for res, top in mrmt.resolutions.items():
            in_set = sum(1 for n in top.nodes.values() if n.metadata.get('in_mandelbrot', False))
            print(f"\n  Resolution {res}x{res}:")
            print(f"    Nodes: {len(top.nodes)}")
            print(f"    Relationships: {len(top.relationships)}")
            print(f"    Points in Mandelbrot set: {in_set}")
    except Exception as e:
        print(f"Error building hierarchy: {e}")

# ============ INTERACTIVE EXPLORATION ============

class InteractiveMandelbrotTopology:
    """Interactive exploration of Mandelbrot topology"""
    
    def __init__(self):
        self.mt = MandelbrotTopology(width=200, height=200, max_iter=60)
        self.topology = None
    
    def explore_region(self, x_center: float = -0.5, y_center: float = 0.0, 
                      zoom: float = 2.0, sample_rate: int = 3):
        """Explore a specific region of the Mandelbrot set"""
        
        # Update ranges
        half_width = zoom / 2
        half_height = zoom / 2
        
        self.mt.x_range = (x_center - half_width, x_center + half_width)
        self.mt.y_range = (y_center - half_height, y_center + half_height)
        
        # Compute and create topology
        self.mt.compute_mandelbrot()
        self.topology = self.mt.create_topology_from_mandelbrot(
            sample_rate=sample_rate,
            connect_boundary=True,
            connect_escape_paths=True
        )
        
        # Visualize
        visualize_mandelbrot_topology(
            self.topology,
            title=f"Mandelbrot Topology: Center=({x_center:.3f}, {y_center:.3f}), Zoom={zoom:.2f}",
            figsize=(12, 8)
        )
        
        return self.topology

# ============ RUN DEMO ============

if __name__ == "__main__":
    demo_mandelbrot_topology()
    
    # Example of interactive exploration
    print("\n" + "=" * 70)
    print("INTERACTIVE EXPLORATION EXAMPLE")
    print("=" * 70)
    
    try:
        # Explore the interesting region around the "seahorse valley"
        interactive = InteractiveMandelbrotTopology()
        topology = interactive.explore_region(
            x_center=-0.75, 
            y_center=0.1, 
            zoom=0.8,
            sample_rate=2
        )
        
        print(f"\nExploration complete!")
        print(f"Found {len(topology.nodes)} points and {len(topology.relationships)} relationships")
        
        # Count nodes in/out of Mandelbrot set
        in_set = sum(1 for n in topology.nodes.values() if n.metadata.get('in_mandelbrot', False))
        print(f"Nodes in Mandelbrot set: {in_set}")
        print(f"Nodes outside set: {len(topology.nodes) - in_set}")
        
    except Exception as e:
        print(f"Error in interactive exploration: {e}")