from typing import List, Dict, Any, Optional, Set
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

@dataclass
class Relationship:
    source: TopologyNode
    target: TopologyNode
    rel_type: RelationshipType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============ TOPOLOGY MANAGER ============
class ObjectTopology:
    """
    Manages object relationships and traversals in various topologies.
    Supports: Tree, Graph, DAG, Star, Mesh, Hierarchical, Cyclic
    """
    
    def __init__(self, topology_type: TopologyType = TopologyType.GRAPH):
        self.topology_type = topology_type
        self.nodes: Dict[str, TopologyNode] = {}
        self.relationships: List[Relationship] = []
        self._adjacency: Dict[str, Set[str]] = {}  # node_id -> set of connected node_ids
        self._reverse_adjacency: Dict[str, Set[str]] = {}
        self._validation_enabled: bool = True
        
    def add_node(self, node: TopologyNode) -> None:
        """Add a node to the topology"""
        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists")
        self.nodes[node.id] = node
        self._adjacency[node.id] = set()
        self._reverse_adjacency[node.id] = set()
    
    def add_relationship(self, source_id: str, target_id: str, 
                        rel_type: RelationshipType = RelationshipType.REFERENCES,
                        weight: float = 1.0, validate: bool = True) -> None:
        """Add a relationship between two nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both nodes must exist")
        
        source = self.nodes[source_id]
        target = self.nodes[target_id]
        
        # Check for cycles if topology doesn't allow them
        if self.topology_type in [TopologyType.TREE, TopologyType.DAG]:
            if self._would_create_cycle(source_id, target_id):
                raise ValueError(f"Relationship would create a cycle in {self.topology_type.value}")
        
        rel = Relationship(source, target, rel_type, weight)
        self.relationships.append(rel)
        self._adjacency[source_id].add(target_id)
        self._reverse_adjacency[target_id].add(source_id)
        
        # Validate if requested
        if validate:
            self._validate_topology()
    
    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """Check if adding edge source->target would create a cycle"""
        visited = set()
        stack = [target_id]
        while stack:
            node = stack.pop()
            if node == source_id:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._adjacency.get(node, set()))
        return False
    
    def _validate_topology(self) -> None:
        """Validate the current topology structure"""
        if not self.nodes or not self.relationships:
            return  # Empty topology or no relationships is valid
        
        if self.topology_type == TopologyType.TREE:
            # Tree: one root, no cycles, each node has at most one parent
            roots = [n for n in self.nodes if not self._reverse_adjacency.get(n, set())]
            
            # For a tree with relationships, we need exactly one root
            if len(roots) != 1:
                # Check if this is a work in progress
                # If not all nodes are connected yet, we might have multiple roots temporarily
                connected_nodes = set()
                for rel in self.relationships:
                    connected_nodes.add(rel.source.id)
                    connected_nodes.add(rel.target.id)
                
                # If all nodes are connected, then multiple roots is an error
                if len(connected_nodes) == len(self.nodes):
                    raise ValueError(f"Tree topology must have exactly one root (found {len(roots)})")
                # Otherwise, it's OK to have multiple roots during construction
            
            # Check each node has at most one parent
            for node_id in self.nodes:
                parent_count = len(self._reverse_adjacency.get(node_id, set()))
                if parent_count > 1:
                    raise ValueError(f"Tree node {node_id} has {parent_count} parents (max 1)")
            
            # Check for cycles
            if self.has_cycle():
                raise ValueError("Tree topology cannot have cycles")
                    
        elif self.topology_type == TopologyType.STAR:
            # Star: one center connected to all others, no other connections
            if len(self.nodes) > 1:
                # Find nodes with multiple outgoing connections
                centers = [n for n in self.nodes if len(self._adjacency.get(n, set())) >= 2]
                if len(centers) != 1:
                    # Check if all nodes are connected
                    connected_nodes = set()
                    for rel in self.relationships:
                        connected_nodes.add(rel.source.id)
                        connected_nodes.add(rel.target.id)
                    
                    if len(connected_nodes) == len(self.nodes):
                        raise ValueError("Star topology must have exactly one center")
                else:
                    # All non-center nodes must connect to center
                    center = centers[0]
                    for node_id in self.nodes:
                        if node_id != center:
                            if center not in self._adjacency.get(node_id, set()) and \
                               center not in self._reverse_adjacency.get(node_id, set()):
                                # Check if all nodes are connected
                                connected_nodes = set()
                                for rel in self.relationships:
                                    connected_nodes.add(rel.source.id)
                                    connected_nodes.add(rel.target.id)
                                
                                if len(connected_nodes) == len(self.nodes):
                                    raise ValueError(f"Node {node_id} must connect to center")
    
    def validate_complete(self) -> None:
        """Force complete validation of the topology"""
        self._validate_topology()
    
    def has_cycle(self) -> bool:
        """Detect if topology contains a cycle"""
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            for neighbor in self._adjacency.get(node_id, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        return False
    
    # ============ TRAVERSAL METHODS ============
    
    def bfs(self, start_id: str) -> List[TopologyNode]:
        """Breadth-First Search traversal"""
        if start_id not in self.nodes:
            raise ValueError(f"Node {start_id} not found")
        
        visited = set()
        queue = [start_id]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            result.append(self.nodes[node_id])
            queue.extend(self._adjacency.get(node_id, set()) - visited)
        
        return result
    
    def dfs(self, start_id: str, recursive: bool = True) -> List[TopologyNode]:
        """Depth-First Search traversal"""
        if start_id not in self.nodes:
            raise ValueError(f"Node {start_id} not found")
        
        if recursive:
            return self._dfs_recursive(start_id, set())
        else:
            return self._dfs_iterative(start_id)
    
    def _dfs_recursive(self, node_id: str, visited: Set[str]) -> List[TopologyNode]:
        visited.add(node_id)
        result = [self.nodes[node_id]]
        for neighbor in self._adjacency.get(node_id, set()):
            if neighbor not in visited:
                result.extend(self._dfs_recursive(neighbor, visited))
        return result
    
    def _dfs_iterative(self, start_id: str) -> List[TopologyNode]:
        visited = set()
        stack = [start_id]
        result = []
        
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            result.append(self.nodes[node_id])
            stack.extend(self._adjacency.get(node_id, set()) - visited)
        
        return result
    
    def topological_sort(self) -> List[TopologyNode]:
        """Topological sort (works for DAG)"""
        if self.has_cycle():
            raise ValueError("Cannot topological sort a graph with cycles")
        
        in_degree = {node_id: len(self._reverse_adjacency.get(node_id, set())) 
                    for node_id in self.nodes}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(self.nodes[node_id])
            for neighbor in self._adjacency.get(node_id, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.nodes):
            raise ValueError("Graph has a cycle (or invalid DAG)")
        return result
    
    def find_path(self, start_id: str, end_id: str) -> List[TopologyNode]:
        """Find a path between two nodes using BFS"""
        if start_id not in self.nodes or end_id not in self.nodes:
            raise ValueError("Both nodes must exist")
        
        visited = set()
        queue = [(start_id, [start_id])]
        
        while queue:
            node_id, path = queue.pop(0)
            if node_id == end_id:
                return [self.nodes[nid] for nid in path]
            
            if node_id in visited:
                continue
            visited.add(node_id)
            
            for neighbor in self._adjacency.get(node_id, set()):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def get_neighbors(self, node_id: str, direction: str = "out") -> List[TopologyNode]:
        """Get neighbors (incoming or outgoing)"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        if direction == "out":
            return [self.nodes[nid] for nid in self._adjacency.get(node_id, set())]
        elif direction == "in":
            return [self.nodes[nid] for nid in self._reverse_adjacency.get(node_id, set())]
        else:
            raise ValueError("Direction must be 'in' or 'out'")
    
    def get_all_paths(self, start_id: str, end_id: str, max_depth: int = 10) -> List[List[TopologyNode]]:
        """Find all paths between two nodes (limited by max_depth)"""
        if start_id not in self.nodes or end_id not in self.nodes:
            raise ValueError("Both nodes must exist")
        
        paths = []
        
        def dfs(current_id: str, target_id: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if current_id == target_id:
                paths.append(path.copy())
                return
            
            for neighbor in self._adjacency.get(current_id, set()):
                if neighbor not in path:  # Avoid cycles
                    dfs(neighbor, target_id, path + [neighbor], depth + 1)
        
        dfs(start_id, end_id, [start_id], 0)
        return [[self.nodes[nid] for nid in path] for path in paths]
    
    # ============ METRICS ============
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get topology statistics"""
        # Temporarily disable validation for statistics
        self._validation_enabled = False
        stats = {
            "node_count": len(self.nodes),
            "relationship_count": len(self.relationships),
            "topology_type": self.topology_type.value,
            "has_cycle": self.has_cycle(),
            "degrees": {
                nid: {
                    "in": len(self._reverse_adjacency.get(nid, set())),
                    "out": len(self._adjacency.get(nid, set()))
                }
                for nid in self.nodes
            },
            "is_connected": self._is_connected(),
            "diameter": self._calculate_diameter()
        }
        self._validation_enabled = True
        return stats
    
    def _is_connected(self) -> bool:
        """Check if topology is weakly connected"""
        if not self.nodes:
            return True
        
        start = next(iter(self.nodes))
        visited = set()
        stack = [start]
        
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            stack.extend(self._adjacency.get(node_id, set()) - visited)
            stack.extend(self._reverse_adjacency.get(node_id, set()) - visited)
        
        return len(visited) == len(self.nodes)
    
    def _calculate_diameter(self) -> int:
        """Calculate the diameter (longest shortest path)"""
        if not self._is_connected():
            return -1
        
        max_dist = 0
        for source in self.nodes:
            distances = {source: 0}
            queue = [source]
            while queue:
                node_id = queue.pop(0)
                for neighbor in self._adjacency.get(node_id, set()):
                    if neighbor not in distances:
                        distances[neighbor] = distances[node_id] + 1
                        queue.append(neighbor)
            max_dist = max(max_dist, max(distances.values()))
        
        return max_dist
    
    # ============ VISUALIZATION ============
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert topology to dictionary representation"""
        return {
            "type": self.topology_type.value,
            "nodes": {nid: {"name": node.name, "metadata": node.metadata} 
                     for nid, node in self.nodes.items()},
            "relationships": [
                {
                    "source": rel.source.id,
                    "target": rel.target.id,
                    "type": rel.rel_type.value,
                    "weight": rel.weight
                }
                for rel in self.relationships
            ]
        }
    
    def __repr__(self) -> str:
        return f"ObjectTopology(type={self.topology_type.value}, nodes={len(self.nodes)}, rels={len(self.relationships)})"

# ============ DEMO: BUILDING DIFFERENT TOPOLOGIES ============

def demo_topologies():
    print("=" * 60)
    print("OBJECT TOPOLOGY DEMONSTRATION")
    print("=" * 60)
    
    # 1. Tree Topology (Hierarchical)
    print("\n1. TREE TOPOLOGY (File System)")
    tree = ObjectTopology(TopologyType.TREE)
    root = TopologyNode(name="Root", metadata={"type": "folder"})
    docs = TopologyNode(name="Documents", metadata={"type": "folder"})
    pics = TopologyNode(name="Pictures", metadata={"type": "folder"})
    file1 = TopologyNode(name="resume.pdf", metadata={"type": "file"})
    file2 = TopologyNode(name="photo.jpg", metadata={"type": "file"})
    
    # Add all nodes first
    for node in [root, docs, pics, file1, file2]:
        tree.add_node(node)
    
    # Add relationships with validation disabled temporarily
    # This allows building the tree incrementally
    tree.add_relationship(root.id, docs.id, RelationshipType.CONTAINS, validate=False)
    tree.add_relationship(root.id, pics.id, RelationshipType.CONTAINS, validate=False)
    tree.add_relationship(docs.id, file1.id, RelationshipType.CONTAINS, validate=False)
    tree.add_relationship(pics.id, file2.id, RelationshipType.CONTAINS, validate=False)
    
    # Now validate the complete tree
    tree.validate_complete()
    
    print(f"Tree BFS from root: {[n.name for n in tree.bfs(root.id)]}")
    print(f"Tree DFS from root: {[n.name for n in tree.dfs(root.id)]}")
    
    # 2. Graph Topology (Social Network)
    print("\n2. GRAPH TOPOLOGY (Social Network)")
    graph = ObjectTopology(TopologyType.GRAPH)
    alice = TopologyNode(name="Alice", metadata={"age": 30})
    bob = TopologyNode(name="Bob", metadata={"age": 28})
    charlie = TopologyNode(name="Charlie", metadata={"age": 32})
    diana = TopologyNode(name="Diana", metadata={"age": 27})
    
    for node in [alice, bob, charlie, diana]:
        graph.add_node(node)
    
    graph.add_relationship(alice.id, bob.id, RelationshipType.ASSOCIATED_WITH, 0.8)
    graph.add_relationship(bob.id, charlie.id, RelationshipType.ASSOCIATED_WITH, 0.6)
    graph.add_relationship(charlie.id, diana.id, RelationshipType.ASSOCIATED_WITH, 0.9)
    graph.add_relationship(diana.id, alice.id, RelationshipType.ASSOCIATED_WITH, 0.7)
    graph.add_relationship(alice.id, charlie.id, RelationshipType.ASSOCIATED_WITH, 0.5)
    
    print(f"Graph has cycle: {graph.has_cycle()}")
    print(f"Path Alice->Charlie: {[n.name for n in graph.find_path(alice.id, charlie.id)]}")
    print(f"All paths Alice->Charlie: {[[n.name for n in p] for p in graph.get_all_paths(alice.id, charlie.id)]}")
    
    # 3. DAG (Dependency Graph)
    print("\n3. DAG TOPOLOGY (Dependencies)")
    dag = ObjectTopology(TopologyType.DAG)
    task_a = TopologyNode(name="Task A", metadata={"priority": 1})
    task_b = TopologyNode(name="Task B", metadata={"priority": 2})
    task_c = TopologyNode(name="Task C", metadata={"priority": 2})
    task_d = TopologyNode(name="Task D", metadata={"priority": 3})
    task_e = TopologyNode(name="Task E", metadata={"priority": 4})
    
    for node in [task_a, task_b, task_c, task_d, task_e]:
        dag.add_node(node)
    
    dag.add_relationship(task_a.id, task_b.id, RelationshipType.DEPENDS_ON)
    dag.add_relationship(task_a.id, task_c.id, RelationshipType.DEPENDS_ON)
    dag.add_relationship(task_b.id, task_d.id, RelationshipType.DEPENDS_ON)
    dag.add_relationship(task_c.id, task_d.id, RelationshipType.DEPENDS_ON)
    dag.add_relationship(task_d.id, task_e.id, RelationshipType.DEPENDS_ON)
    
    print(f"DAG topological sort: {[n.name for n in dag.topological_sort()]}")
    
    # 4. Star Topology (Hub-and-Spoke)
    print("\n4. STAR TOPOLOGY (Hub-and-Spoke)")
    star = ObjectTopology(TopologyType.STAR)
    hub = TopologyNode(name="Hub", metadata={"type": "central"})
    spoke1 = TopologyNode(name="Spoke1", metadata={"type": "peripheral"})
    spoke2 = TopologyNode(name="Spoke2", metadata={"type": "peripheral"})
    spoke3 = TopologyNode(name="Spoke3", metadata={"type": "peripheral"})
    
    for node in [hub, spoke1, spoke2, spoke3]:
        star.add_node(node)
    
    star.add_relationship(hub.id, spoke1.id, RelationshipType.REFERENCES)
    star.add_relationship(hub.id, spoke2.id, RelationshipType.REFERENCES)
    star.add_relationship(hub.id, spoke3.id, RelationshipType.REFERENCES)
    
    print(f"Star neighbors of Hub: {[n.name for n in star.get_neighbors(hub.id)]}")
    print(f"Star statistics: {star.get_statistics()}")
    
    # 5. Cyclic Topology
    print("\n5. CYCLIC TOPOLOGY (Circular Reference)")
    cyclic = ObjectTopology(TopologyType.CYCLIC)
    a = TopologyNode(name="A")
    b = TopologyNode(name="B")
    c = TopologyNode(name="C")
    
    for node in [a, b, c]:
        cyclic.add_node(node)
    
    cyclic.add_relationship(a.id, b.id, RelationshipType.REFERENCES)
    cyclic.add_relationship(b.id, c.id, RelationshipType.REFERENCES)
    # Cycle will be auto-created: c -> a
    
    print(f"Cyclic has cycle: {cyclic.has_cycle()}")
    print(f"Cyclic adjacency: {cyclic._adjacency}")
    
    # Print all statistics
    print("\n" + "=" * 60)
    print("ALL TOPOLOGY STATISTICS")
    for name, top in [("Tree", tree), ("Graph", graph), ("DAG", dag), 
                      ("Star", star), ("Cyclic", cyclic)]:
        stats = top.get_statistics()
        print(f"\n{name}:")
        print(f"  Nodes: {stats['node_count']}")
        print(f"  Relationships: {stats['relationship_count']}")
        print(f"  Has cycle: {stats['has_cycle']}")
        print(f"  Connected: {stats['is_connected']}")
        print(f"  Diameter: {stats['diameter']}")

# ============ ADVANCED: COMPOSITE TOPOLOGY EXAMPLE ============

class CompositeTopology(ObjectTopology):
    """Topology that can contain other topologies (nested)"""
    
    def __init__(self, name: str = "Composite"):
        super().__init__(TopologyType.HIERARCHICAL)
        self.name = name
        self.sub_topologies: List[ObjectTopology] = []
    
    def add_sub_topology(self, topology: ObjectTopology) -> None:
        """Add a sub-topology as a node"""
        self.sub_topologies.append(topology)
        # Create a meta-node to represent the topology
        meta = TopologyNode(
            name=f"Meta:{topology.topology_type.value}",
            metadata={"sub_nodes": len(topology.nodes)}
        )
        self.add_node(meta)
        
        # Connect meta-node to all nodes in sub-topology
        for node in topology.nodes.values():
            self.add_relationship(meta.id, node.id, RelationshipType.COMPOSED_OF)

if __name__ == "__main__":
    demo_topologies()