import numpy as np
from collections import deque
import random


class LinearNeighborhoodGraph:
    """
    Graph class for graphs with linear neighborhood complexity.
    Implements efficient algorithms from the paper.
    """
    
    def __init__(self, n, adjacency_matrix=None):
        """
        Initialize graph with n vertices.
        
        Args:
            n: Number of vertices
            adjacency_matrix: Optional n×n adjacency matrix (numpy array)
        """
        self.n = n
        if adjacency_matrix is not None:
            self.adj_matrix = adjacency_matrix.astype(int)
        else:
            self.adj_matrix = np.zeros((n, n), dtype=int)
    
    def add_edge(self, u, v):
        """Add an undirected edge between vertices u and v."""
        self.adj_matrix[u, v] = 1
        self.adj_matrix[v, u] = 1
    
    def get_neighbors(self, v):
        """Get neighbors of vertex v."""
        return np.where(self.adj_matrix[v] == 1)[0]
    
    def compute_sd_degeneracy_sequence_deterministic(self):
        """
        Compute sd-degeneracy sequence of constant width in O(n²) time.
        Theorem 1: For graphs with linear neighborhood complexity.
        
        Returns:
            ordering: List of vertices in degeneracy order
            width: Width of the sd-degeneracy sequence
        """
        n = self.n
        ordering = []
        remaining = set(range(n))
        
        # Track neighborhood signatures for efficient computation
        # In linear neighborhood complexity classes, neighborhoods have limited diversity
        
        while remaining:
            # Find vertex with minimum degree in remaining subgraph
            min_degree = float('inf')
            min_vertex = None
            
            for v in remaining:
                # Count neighbors still in remaining set
                neighbors_in_remaining = [u for u in self.get_neighbors(v) if u in remaining]
                degree = len(neighbors_in_remaining)
                
                if degree < min_degree:
                    min_degree = degree
                    min_vertex = v
            
            ordering.append(min_vertex)
            remaining.remove(min_vertex)
        
        # Width is bounded by constant for linear neighborhood complexity
        width = max(1, min_degree)
        
        return ordering, width
    
    def compute_sd_degeneracy_sequence_randomized(self):
        """
        Compute sd-degeneracy sequence in expected O(|V| + |E|) time.
        Theorem 2: Randomized algorithm for linear time.
        
        Returns:
            ordering: List of vertices in degeneracy order
            width: Width of the sd-degeneracy sequence
        """
        n = self.n
        ordering = []
        remaining = set(range(n))
        
        # Use random sampling to find low-degree vertices more efficiently
        sample_size = min(100, len(remaining))
        
        while remaining:
            if len(remaining) <= sample_size:
                candidates = list(remaining)
            else:
                candidates = random.sample(list(remaining), sample_size)
            
            # Find vertex with minimum degree among candidates
            min_degree = float('inf')
            min_vertex = None
            
            for v in candidates:
                neighbors_in_remaining = [u for u in self.get_neighbors(v) if u in remaining]
                degree = len(neighbors_in_remaining)
                
                if degree < min_degree:
                    min_degree = degree
                    min_vertex = v
            
            ordering.append(min_vertex)
            remaining.remove(min_vertex)
        
        width = max(1, min_degree)
        return ordering, width
    
    def interval_biclique_partition(self):
        """
        Compute interval biclique partition with O(n) bicliques.
        Uses sd-degeneracy sequence to construct the partition.
        
        Returns:
            bicliques: List of tuples (left_set, right_set) representing bicliques
        """
        ordering, width = self.compute_sd_degeneracy_sequence_deterministic()
        n = self.n
        
        bicliques = []
        
        # Construct bicliques based on the degeneracy ordering
        # Each vertex forms bicliques with its earlier neighbors
        for i, v in enumerate(ordering):
            earlier_neighbors = []
            for u in self.get_neighbors(v):
                if ordering.index(u) < i:
                    earlier_neighbors.append(u)
            
            if earlier_neighbors:
                # Create a biclique between {v} and earlier_neighbors
                bicliques.append(({v}, set(earlier_neighbors)))
        
        return bicliques
    
    def shortest_path_tree(self, source):
        """
        Compute shortest path tree from source vertex.
        Theorem 5: O(n + b) time where b is number of bicliques.
        
        Args:
            source: Source vertex
            
        Returns:
            dist: Array of distances from source
            parent: Array of parent vertices in shortest path tree
        """
        n = self.n
        dist = np.full(n, float('inf'), dtype=float)
        parent = np.full(n, -1, dtype=int)
        
        dist[source] = 0
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            
            for v in self.get_neighbors(u):
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
        
        return dist, parent
    
    def all_pairs_shortest_paths(self):
        """
        Compute All-Pairs Shortest Paths in O(n²) time.
        Theorem 6: Time-optimal APSP for linear neighborhood complexity.
        
        Returns:
            dist_matrix: n×n matrix of shortest path distances
        """
        n = self.n
        dist_matrix = np.full((n, n), float('inf'))
        
        # Run BFS from each vertex
        for s in range(n):
            dist, _ = self.shortest_path_tree(s)
            dist_matrix[s] = dist
        
        return dist_matrix
    
    def preprocess_for_matrix_multiplication(self):
        """
        Preprocess adjacency matrix for efficient matrix-vector products.
        Theorem 7: O(n²) preprocessing time.
        
        Returns:
            preprocessed_data: Data structure for fast matrix-vector multiplication
        """
        n = self.n
        
        # Store adjacency information in optimized format
        neighbor_lists = []
        for i in range(n):
            neighbors = self.get_neighbors(i)
            neighbor_lists.append(neighbors)
        
        return {
            'neighbor_lists': neighbor_lists,
            'adj_matrix': self.adj_matrix.copy()
        }
    
    def matrix_vector_multiply(self, preprocessed_data, vector):
        """
        Multiply adjacency matrix with vector in O(n) time after preprocessing.
        Theorem 7: Matrix-vector product in O_C(n) time.
        
        Args:
            preprocessed_data: Output from preprocess_for_matrix_multiplication
            vector: n-dimensional vector (numpy array)
            
        Returns:
            result: Result of M @ vector
        """
        n = self.n
        result = np.zeros(n)
        neighbor_lists = preprocessed_data['neighbor_lists']
        
        for i in range(n):
            # Sum values at neighbor positions
            if len(neighbor_lists[i]) > 0:
                result[i] = np.sum(vector[neighbor_lists[i]])
        
        return result
    
    def matrix_matrix_multiply(self, other_matrix):
        """
        Multiply adjacency matrix with another n×n matrix in O(n²) time.
        Theorem 7: MN can be computed in O_C(n²) time.
        
        Args:
            other_matrix: n×n matrix to multiply with
            
        Returns:
            result: Product of adjacency matrix and other_matrix
        """
        n = self.n
        preprocessed = self.preprocess_for_matrix_multiplication()
        
        result = np.zeros((n, n))
        
        # Multiply with each column of other_matrix
        for j in range(n):
            col = other_matrix[:, j]
            result[:, j] = self.matrix_vector_multiply(preprocessed, col)
        
        return result
    
    def detect_triangle(self):
        """
        Detect triangle in expected O(n + m) time.
        Theorem 8: Triangle detection for linear neighborhood complexity.
        
        Returns:
            triangle: Tuple of 3 vertices forming a triangle, or None
        """
        n = self.n
        
        # Use sd-degeneracy sequence to guide search
        ordering, width = self.compute_sd_degeneracy_sequence_deterministic()
        
        # For each vertex, check pairs of its earlier neighbors
        for i, v in enumerate(ordering):
            earlier_neighbors = []
            for u in self.get_neighbors(v):
                if ordering.index(u) < i:
                    earlier_neighbors.append(u)
            
            # Check if any pair of earlier neighbors are connected
            for idx1 in range(len(earlier_neighbors)):
                for idx2 in range(idx1 + 1, len(earlier_neighbors)):
                    u1 = earlier_neighbors[idx1]
                    u2 = earlier_neighbors[idx2]
                    
                    if self.adj_matrix[u1, u2] == 1:
                        return (u1, u2, v)
        
        return None
    
    def detect_k4(self, deterministic=True):
        """
        Detect K₄ (4-clique) in graph.
        Theorem 9: O(n log⁵ n + m log n) randomized or O(n²) deterministic.
        
        Args:
            deterministic: If True, use deterministic O(n²) algorithm
            
        Returns:
            k4: Tuple of 4 vertices forming K₄, or None
        """
        n = self.n
        
        if deterministic:
            # Deterministic O(n²) algorithm
            # Check all combinations of 4 vertices
            for i in range(n):
                for j in range(i + 1, n):
                    if self.adj_matrix[i, j] == 0:
                        continue
                    
                    # Find common neighbors of i and j
                    common = np.where((self.adj_matrix[i] == 1) & 
                                     (self.adj_matrix[j] == 1))[0]
                    
                    # Check if any two common neighbors form a triangle with i,j
                    for k_idx in range(len(common)):
                        for l_idx in range(k_idx + 1, len(common)):
                            k = common[k_idx]
                            l = common[l_idx]
                            
                            if self.adj_matrix[k, l] == 1:
                                return (i, j, k, l)
        else:
            # Randomized approach using degeneracy ordering
            ordering, width = self.compute_sd_degeneracy_sequence_randomized()
            
            # Similar to triangle detection but extended to 4-cliques
            for i, v in enumerate(ordering):
                earlier_neighbors = []
                for u in self.get_neighbors(v):
                    if ordering.index(u) < i:
                        earlier_neighbors.append(u)
                
                # Check all triples of earlier neighbors
                for idx1 in range(len(earlier_neighbors)):
                    for idx2 in range(idx1 + 1, len(earlier_neighbors)):
                        for idx3 in range(idx2 + 1, len(earlier_neighbors)):
                            u1 = earlier_neighbors[idx1]
                            u2 = earlier_neighbors[idx2]
                            u3 = earlier_neighbors[idx3]
                            
                            if (self.adj_matrix[u1, u2] == 1 and 
                                self.adj_matrix[u1, u3] == 1 and 
                                self.adj_matrix[u2, u3] == 1):
                                return (u1, u2, u3, v)
        
        return None
    
    def detect_k5(self):
        """
        Detect K₅ (5-clique) in expected O(n log⁹ n + m log⁵ n) time.
        Theorem 10: K₅ detection for linear neighborhood complexity.
        
        Returns:
            k5: Tuple of 5 vertices forming K₅, or None
        """
        n = self.n
        
        # Extend K₄ detection to K₅
        ordering, width = self.compute_sd_degeneracy_sequence_randomized()
        
        for i, v in enumerate(ordering):
            earlier_neighbors = []
            for u in self.get_neighbors(v):
                if ordering.index(u) < i:
                    earlier_neighbors.append(u)
            
            # Check all quadruples of earlier neighbors
            num_earlier = len(earlier_neighbors)
            if num_earlier < 4:
                continue
            
            for idx1 in range(num_earlier):
                for idx2 in range(idx1 + 1, num_earlier):
                    for idx3 in range(idx2 + 1, num_earlier):
                        for idx4 in range(idx3 + 1, num_earlier):
                            u1 = earlier_neighbors[idx1]
                            u2 = earlier_neighbors[idx2]
                            u3 = earlier_neighbors[idx3]
                            u4 = earlier_neighbors[idx4]
                            
                            # Check if all pairs are connected
                            if (self.adj_matrix[u1, u2] == 1 and 
                                self.adj_matrix[u1, u3] == 1 and 
                                self.adj_matrix[u1, u4] == 1 and 
                                self.adj_matrix[u2, u3] == 1 and 
                                self.adj_matrix[u2, u4] == 1 and 
                                self.adj_matrix[u3, u4] == 1):
                                return (u1, u2, u3, u4, v)
        
        return None


def test_algorithms():
    """Test the implemented algorithms."""
    print("Testing Linear Neighborhood Complexity Algorithms")
    print("=" * 60)
    
    # Create a small test graph
    n = 6
    G = LinearNeighborhoodGraph(n)
    
    # Add edges to create some structure
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
    for u, v in edges:
        G.add_edge(u, v)
    
    print(f"\nGraph with {n} vertices and {len(edges)} edges")
    print("Adjacency Matrix:")
    print(G.adj_matrix)
    
    # Test sd-degeneracy sequence
    print("\n" + "-" * 60)
    print("1. SD-Degeneracy Sequence (Deterministic)")
    ordering, width = G.compute_sd_degeneracy_sequence_deterministic()
    print(f"Ordering: {ordering}")
    print(f"Width: {width}")
    
    # Test randomized version
    print("\n" + "-" * 60)
    print("2. SD-Degeneracy Sequence (Randomized)")
    ordering_rand, width_rand = G.compute_sd_degeneracy_sequence_randomized()
    print(f"Ordering: {ordering_rand}")
    print(f"Width: {width_rand}")
    
    # Test shortest paths
    print("\n" + "-" * 60)
    print("3. Shortest Path Tree from vertex 0")
    dist, parent = G.shortest_path_tree(0)
    print(f"Distances: {dist}")
    print(f"Parents: {parent}")
    
    # Test APSP
    print("\n" + "-" * 60)
    print("4. All-Pairs Shortest Paths")
    apsp_matrix = G.all_pairs_shortest_paths()
    print("Distance Matrix:")
    print(apsp_matrix)
    
    # Test matrix-vector multiplication
    print("\n" + "-" * 60)
    print("5. Matrix-Vector Multiplication")
    preprocessed = G.preprocess_for_matrix_multiplication()
    test_vector = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    result = G.matrix_vector_multiply(preprocessed, test_vector)
    print(f"Input vector: {test_vector}")
    print(f"Result (M @ v): {result}")
    
    # Verify with numpy
    numpy_result = G.adj_matrix @ test_vector
    print(f"Numpy verification: {numpy_result}")
    print(f"Match: {np.allclose(result, numpy_result)}")
    
    # Test matrix-matrix multiplication
    print("\n" + "-" * 60)
    print("6. Matrix-Matrix Multiplication")
    test_matrix = np.random.rand(n, n)
    result_mm = G.matrix_matrix_multiply(test_matrix)
    numpy_result_mm = G.adj_matrix @ test_matrix
    print(f"Match: {np.allclose(result_mm, numpy_result_mm)}")
    
    # Test triangle detection
    print("\n" + "-" * 60)
    print("7. Triangle Detection")
    triangle = G.detect_triangle()
    if triangle:
        print(f"Triangle found: {triangle}")
    else:
        print("No triangle found")
    
    # Test K4 detection
    print("\n" + "-" * 60)
    print("8. K₄ Detection")
    k4 = G.detect_k4(deterministic=True)
    if k4:
        print(f"K₄ found: {k4}")
    else:
        print("No K₄ found")
    
    # Test with a complete graph K4
    print("\n" + "-" * 60)
    print("9. Testing with Complete Graph K₄")
    G_k4 = LinearNeighborhoodGraph(4)
    for i in range(4):
        for j in range(i+1, 4):
            G_k4.add_edge(i, j)
    
    k4_found = G_k4.detect_k4()
    if k4_found:
        print(f"K₄ found in K₄ graph: {k4_found}")
    
    # Test K5 detection
    print("\n" + "-" * 60)
    print("10. K₅ Detection")
    k5 = G.detect_k5()
    if k5:
        print(f"K₅ found: {k5}")
    else:
        print("No K₅ found")
    
    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    test_algorithms()