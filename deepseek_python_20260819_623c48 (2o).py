import networkx as nx
import numpy as np
import hashlib
from collections import defaultdict
import random

class InfiniteFastHash:
    """
    Infinite Fast Hashing using NetworkX graphs
    Implements graph-based hashing with infinite capacity
    """
    
    def __init__(self, graph=None, hash_dim=128):
        """
        Initialize the hasher with a graph
        
        Args:
            graph: NetworkX graph (if None, creates empty graph)
            hash_dim: Dimension of hash output
        """
        self.graph = graph if graph else nx.Graph()
        self.hash_dim = hash_dim
        self.node_hashes = {}
        self.hash_counter = 0
        
    def add_node(self, node_id, features=None):
        """Add node to graph and compute its hash"""
        if node_id not in self.graph:
            self.graph.add_node(node_id, features=features)
            
        # Compute initial hash using features or node ID
        if features is not None:
            hash_val = self._feature_hash(features)
        else:
            hash_val = self._id_hash(node_id)
            
        self.node_hashes[node_id] = hash_val
        return hash_val
    
    def add_edge(self, u, v, weight=1.0):
        """Add edge between nodes and update hashes"""
        if u not in self.graph:
            self.add_node(u)
        if v not in self.graph:
            self.add_node(v)
            
        self.graph.add_edge(u, v, weight=weight)
        
        # Update hashes of connected nodes
        self._update_neighbor_hashes(u)
        self._update_neighbor_hashes(v)
        
    def _feature_hash(self, features):
        """Hash node features using SHA-256"""
        if isinstance(features, (list, np.ndarray)):
            feature_str = str(features)
        else:
            feature_str = str(features)
        return hashlib.sha256(feature_str.encode()).hexdigest()
    
    def _id_hash(self, node_id):
        """Hash node ID"""
        return hashlib.sha256(str(node_id).encode()).hexdigest()
    
    def _update_neighbor_hashes(self, node_id):
        """Update hash based on neighbors"""
        neighbors = list(self.graph.neighbors(node_id))
        if not neighbors:
            return
            
        # Combine neighbor hashes
        neighbor_hashes = [self.node_hashes[n] for n in neighbors if n in self.node_hashes]
        if neighbor_hashes:
            combined = ''.join(sorted(neighbor_hashes))
            new_hash = hashlib.sha256(combined.encode()).hexdigest()
            self.node_hashes[node_id] = new_hash
            
    def get_hash(self, node_id):
        """Get current hash for a node"""
        return self.node_hashes.get(node_id, None)
    
    def batch_hash(self, node_ids):
        """Get hashes for multiple nodes"""
        return {nid: self.get_hash(nid) for nid in node_ids}
    
    def graph_hash(self):
        """Compute global graph hash"""
        # Sort node hashes for deterministic output
        sorted_hashes = sorted(self.node_hashes.values())
        combined = ''.join(sorted_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _random_walk(self, start_node, walk_length, seed=None):
        """
        Perform a random walk on the graph
        
        Args:
            start_node: Starting node
            walk_length: Number of steps to walk
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            
        walk = [start_node]
        current = start_node
        
        for _ in range(walk_length):
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
            current = random.choice(neighbors)
            walk.append(current)
            
        return walk
    
    def fast_hash(self, data, k=10, walk_length=5):
        """
        Fast approximate hashing using random walks
        
        Args:
            data: Input data to hash
            k: Number of random walks
            walk_length: Length of each random walk
        """
        if len(self.graph.nodes()) == 0:
            return hashlib.sha256(str(data).encode()).hexdigest()
            
        # Perform random walks to get context
        context = []
        nodes = list(self.graph.nodes())
        
        for i in range(k):
            start_node = random.choice(nodes)
            walk = self._random_walk(start_node, walk_length, seed=i if k < 100 else None)
            context.extend([str(n) for n in walk])
            
        combined = str(data) + ''.join(context)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def incremental_hash_update(self, node_id, new_features):
        """Update hash incrementally when node features change"""
        if node_id in self.graph:
            self.graph.nodes[node_id]['features'] = new_features
            self.node_hashes[node_id] = self._feature_hash(new_features)
            self._update_neighbor_hashes(node_id)
            
            # Update neighbors recursively
            for neighbor in self.graph.neighbors(node_id):
                self._update_neighbor_hashes(neighbor)
                
    def merge_graphs(self, other_graph):
        """Merge another graph into this one"""
        # Add all nodes
        for node in other_graph.nodes():
            features = other_graph.nodes[node].get('features', None)
            self.add_node(node, features)
            
        # Add all edges
        for u, v, data in other_graph.edges(data=True):
            weight = data.get('weight', 1.0)
            self.add_edge(u, v, weight)
            
        # Recompute all hashes
        for node in self.graph.nodes():
            self._update_neighbor_hashes(node)
            
    def similarity_hash(self, node1, node2):
        """Compute similarity hash between two nodes"""
        hash1 = int(self.node_hashes.get(node1, '0'), 16)
        hash2 = int(self.node_hashes.get(node2, '0'), 16)
        return hash1 ^ hash2  # XOR similarity
    
    def get_graph_statistics(self):
        """Get statistics about the graph"""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': 2 * self.graph.number_of_edges() / max(1, self.graph.number_of_nodes()),
            'components': nx.number_connected_components(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }


# Alternative: Use biased random walk (like Node2Vec)
class InfiniteFastHashWithBiasedWalk(InfiniteFastHash):
    """
    Extended version with biased random walks (like Node2Vec)
    """
    
    def __init__(self, graph=None, hash_dim=128, p=1.0, q=1.0):
        """
        Initialize with bias parameters
        
        Args:
            p: Return parameter (1.0 = neutral)
            q: In-out parameter (1.0 = neutral)
        """
        super().__init__(graph, hash_dim)
        self.p = p
        self.q = q
        
    def _biased_random_walk(self, start_node, walk_length):
        """
        Perform biased random walk (similar to Node2Vec)
        
        Args:
            start_node: Starting node
            walk_length: Number of steps
        """
        walk = [start_node]
        
        while len(walk) < walk_length:
            current = walk[-1]
            
            # Get neighbors
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
                
            if len(walk) == 1:
                # First step: random choice
                next_node = random.choice(neighbors)
            else:
                # Biased choice
                prev = walk[-2]
                weights = []
                
                for neighbor in neighbors:
                    if neighbor == prev:
                        # Return to previous node
                        weights.append(1.0 / self.p)
                    elif self.graph.has_edge(neighbor, prev):
                        # Distance 2 from previous node
                        weights.append(1.0)
                    else:
                        # Distance > 2 from previous node
                        weights.append(1.0 / self.q)
                
                # Normalize and choose
                weights = np.array(weights) / np.sum(weights)
                next_node = np.random.choice(neighbors, p=weights)
                
            walk.append(next_node)
            
        return walk
    
    def fast_hash(self, data, k=10, walk_length=5, biased=True):
        """
        Fast hashing with optional biased random walks
        """
        if len(self.graph.nodes()) == 0:
            return hashlib.sha256(str(data).encode()).hexdigest()
            
        context = []
        nodes = list(self.graph.nodes())
        
        for i in range(k):
            start_node = random.choice(nodes)
            if biased:
                walk = self._biased_random_walk(start_node, walk_length)
            else:
                walk = self._random_walk(start_node, walk_length)
            context.extend([str(n) for n in walk])
            
        combined = str(data) + ''.join(context)
        return hashlib.sha256(combined.encode()).hexdigest()


# Demo function
def demo_infinite_fast_hash():
    """Demonstrate infinite fast hashing functionality"""
    
    # Create hasher instance
    hasher = InfiniteFastHash(hash_dim=128)
    
    # Add nodes with features
    features1 = [0.1, 0.2, 0.3, 0.4]
    features2 = [0.5, 0.6, 0.7, 0.8]
    features3 = [0.9, 1.0, 1.1, 1.2]
    
    hasher.add_node('A', features1)
    hasher.add_node('B', features2)
    hasher.add_node('C', features3)
    hasher.add_node('D', [1.3, 1.4, 1.5, 1.6])
    
    # Add edges
    hasher.add_edge('A', 'B', weight=0.8)
    hasher.add_edge('B', 'C', weight=0.6)
    hasher.add_edge('C', 'D', weight=0.7)
    hasher.add_edge('A', 'D', weight=0.3)
    
    # Get node hashes
    print("Node Hashes:")
    for node in ['A', 'B', 'C', 'D']:
        hash_val = hasher.get_hash(node)
        print(f"  {node}: {hash_val[:16] if hash_val else 'None'}...")
    
    # Graph hash
    print(f"\nGraph Hash: {hasher.graph_hash()[:16]}...")
    
    # Fast hash for new data
    new_data = "test_data_123"
    fast_hash = hasher.fast_hash(new_data, k=5, walk_length=3)
    print(f"\nFast Hash for '{new_data}': {fast_hash[:16]}...")
    
    # Incremental update
    print("\nUpdating node A features...")
    hasher.incremental_hash_update('A', [2.0, 3.0, 4.0, 5.0])
    print(f"New hash for A: {hasher.get_hash('A')[:16]}...")
    
    # Batch hashing
    batch_results = hasher.batch_hash(['A', 'B', 'C', 'D'])
    print("\nBatch Hash Results:")
    for node, hash_val in batch_results.items():
        print(f"  {node}: {hash_val[:16] if hash_val else 'None'}...")
    
    # Graph statistics
    stats = hasher.get_graph_statistics()
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return hasher


# Performance test
def performance_test(n_nodes=1000, n_edges=5000):
    """Test performance with large graphs"""
    import time
    
    print(f"\n{'='*50}")
    print(f"Performance Test: {n_nodes} nodes, {n_edges} edges")
    print(f"{'='*50}")
    
    # Create random graph
    G = nx.gnm_random_graph(n_nodes, n_edges)
    hasher = InfiniteFastHash(G)
    
    # Add features to nodes
    start = time.time()
    for i in range(n_nodes):
        features = np.random.randn(10)
        hasher.add_node(i, features)
    print(f"✓ Added {n_nodes} nodes in {time.time() - start:.3f}s")
    
    # Compute hashes
    start = time.time()
    for i in range(n_nodes):
        hasher.get_hash(i)
    print(f"✓ Computed {n_nodes} hashes in {time.time() - start:.3f}s")
    
    # Fast hash with random walks
    start = time.time()
    for i in range(100):
        hasher.fast_hash(f"test_{i}", k=5, walk_length=3)
    print(f"✓ 100 fast hashes in {time.time() - start:.3f}s")
    
    # Test biased random walk version
    print("\nTesting biased random walk version...")
    biased_hasher = InfiniteFastHashWithBiasedWalk(G, p=0.5, q=2.0)
    for i in range(n_nodes):
        biased_hasher.add_node(i, np.random.randn(10))
    
    start = time.time()
    for i in range(50):
        biased_hasher.fast_hash(f"biased_{i}", k=3, walk_length=4, biased=True)
    print(f"✓ 50 biased fast hashes in {time.time() - start:.3f}s")
    
    return hasher


# Advanced usage: Graph similarity
def graph_similarity_demo():
    """Demonstrate graph similarity using hashing"""
    print("\n" + "="*50)
    print("Graph Similarity Demo")
    print("="*50)
    
    # Create two similar graphs
    G1 = nx.erdos_renyi_graph(50, 0.1, seed=42)
    G2 = nx.erdos_renyi_graph(50, 0.1, seed=42)  # Same seed = same graph
    
    hasher1 = InfiniteFastHash(G1)
    hasher2 = InfiniteFastHash(G2)
    
    # Add features
    for i in range(50):
        hasher1.add_node(i, np.random.randn(5))
        hasher2.add_node(i, np.random.randn(5))
    
    # Compare hashes
    hash1 = hasher1.graph_hash()
    hash2 = hasher2.graph_hash()
    
    print(f"Graph 1 hash: {hash1[:16]}...")
    print(f"Graph 2 hash: {hash2[:16]}...")
    print(f"Hashes equal: {hash1 == hash2}")
    
    # Create different graph
    G3 = nx.erdos_renyi_graph(50, 0.2, seed=123)
    hasher3 = InfiniteFastHash(G3)
    for i in range(50):
        hasher3.add_node(i, np.random.randn(5))
    
    hash3 = hasher3.graph_hash()
    print(f"Graph 3 hash: {hash3[:16]}...")
    print(f"Hash1 vs Hash3: {'Same' if hash1 == hash3 else 'Different'}")


if __name__ == "__main__":
    # Run demo
    hasher = demo_infinite_fast_hash()
    
    # Run performance test
    performance_test(500, 1500)
    
    # Run similarity demo
    graph_similarity_demo()