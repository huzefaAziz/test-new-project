import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Set, Any, Optional
from dataclasses import dataclass
import random

@dataclass
class HashClassZ:
    """
    Implementation of the hash class Z from the paper.
    h_i(x) = (f_i(x) + sum_{j=1..c} z_i[j, g_j(x)]) mod m
    """
    m: int  # range of hash functions
    ell: int  # range of g functions
    c: int  # number of g functions
    d: int  # number of hash functions
    universe_size: int  # size of U
    
    def __post_init__(self):
        # Initialize 2-wise independent hash functions f_i: U -> [m]
        # Using polynomial hash: f(x) = (a*x + b) mod p mod m
        # Using a large prime for the field
        self.p = self._find_prime(max(self.universe_size, self.m) * 2)
        
        # Generate f_i hash functions (d of them)
        self.f_hash_funcs = []
        for _ in range(self.d):
            a = random.randint(1, self.p - 1)
            b = random.randint(0, self.p - 1)
            self.f_hash_funcs.append((a, b))
        
        # Generate g_j hash functions (c of them)
        # Using 2-universal hash: g(x) = ((a*x + b) mod p) mod ell
        self.g_hash_funcs = []
        for _ in range(self.c):
            a = random.randint(1, self.p - 1)
            b = random.randint(0, self.p - 1)
            self.g_hash_funcs.append((a, b))
        
        # Initialize z tables: z[i][j][k] for i in 0..d-1, j in 0..c-1, k in 0..ell-1
        self.z_tables = []
        for _ in range(self.d):
            table = np.random.randint(0, self.m, size=(self.c, self.ell))
            self.z_tables.append(table)
    
    def _find_prime(self, n: int) -> int:
        """Find a prime number >= n."""
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True
        
        while not is_prime(n):
            n += 1
        return n
    
    def _f(self, i: int, x: int) -> int:
        """Apply f_i hash function to x."""
        a, b = self.f_hash_funcs[i]
        return ((a * x + b) % self.p) % self.m
    
    def _g(self, j: int, x: int) -> int:
        """Apply g_j hash function to x."""
        a, b = self.g_hash_funcs[j]
        return ((a * x + b) % self.p) % self.ell
    
    def hash(self, i: int, x: int) -> int:
        """Compute h_i(x)."""
        result = self._f(i, x)
        for j in range(self.c):
            result = (result + self.z_tables[i][j, self._g(j, x)]) % self.m
        return result
    
    def hash_all(self, x: int) -> List[int]:
        """Compute all h_i(x) for i=1..d."""
        return [self.hash(i, x) for i in range(self.d)]


def build_hash_graph(keys: List[int], hash_func: HashClassZ) -> nx.Graph:
    """
    Build the d-partite graph G(S, h_vec) from the paper.
    For d=2, this creates a bipartite graph.
    """
    n = len(keys)
    m = hash_func.m
    d = hash_func.d
    
    # Create graph with d copies of [m] vertices
    # Vertex IDs: (copy_index, vertex_index)
    G = nx.Graph()
    
    # Add vertices for each copy
    for copy in range(d):
        for v in range(m):
            G.add_node((copy, v))
    
    # Add edges for each key
    edges_added = set()
    for key in keys:
        hash_values = hash_func.hash_all(key)
        edge = tuple((copy, val) for copy, val in enumerate(hash_values))
        G.add_edge(edge[0], edge[1])
        edges_added.add(edge)
    
    # Remove isolated vertices
    G.remove_nodes_from(list(nx.isolates(G)))
    
    return G


def get_deficiency(hash_func: HashClassZ, T: Set[int]) -> int:
    """
    Calculate the deficiency d_T for a set T.
    d_T = |T| - max_j |g_j(T)|
    """
    max_image_size = 0
    for j in range(hash_func.c):
        image = {hash_func._g(j, x) for x in T}
        max_image_size = max(max_image_size, len(image))
    return len(T) - max_image_size


def is_good(hash_func: HashClassZ, T: Set[int]) -> bool:
    """Check if hash function is T-good (deficiency <= 1)."""
    return get_deficiency(hash_func, T) <= 1


def is_critical(hash_func: HashClassZ, T: Set[int]) -> bool:
    """Check if hash function is T-critical (deficiency == 1)."""
    return get_deficiency(hash_func, T) == 1


def is_bad(hash_func: HashClassZ, T: Set[int]) -> bool:
    """Check if hash function is T-bad (deficiency > 1)."""
    return get_deficiency(hash_func, T) > 1


def find_minimal_obstruction_graphs(G: nx.Graph) -> List[nx.Graph]:
    """
    Find all minimal obstruction graphs (MOG) in G.
    A MOG is a cycle with a chord or two cycles connected by a path.
    """
    mog_graphs = []
    
    # Find all simple cycles
    try:
        cycles = list(nx.simple_cycles(G))
    except:
        # For directed conversion if needed
        cycles = []
    
    # For each cycle, check for chords or connections to other cycles
    # This is a simplified implementation
    for cycle in cycles:
        cycle_nodes = set(cycle)
        
        # Check for chords (edges connecting non-adjacent nodes in the cycle)
        chord_found = False
        for i in range(len(cycle)):
            for j in range(i + 2, len(cycle)):
                if j == i + 1 or (i == 0 and j == len(cycle) - 1):
                    continue
                if G.has_edge(cycle[i], cycle[j]):
                    chord_found = True
                    break
            if chord_found:
                break
        
        if chord_found:
            # Create subgraph for this MOG
            subgraph = G.subgraph(cycle_nodes)
            mog_graphs.append(subgraph)
    
    return mog_graphs


def simulate_fully_random_hash(keys: List[int], m: int, d: int) -> nx.Graph:
    """
    Simulate d fully random hash functions for graph building.
    Used for comparison/analysis.
    """
    G = nx.Graph()
    
    # Add vertices
    for copy in range(d):
        for v in range(m):
            G.add_node((copy, v))
    
    # Add random edges
    for key in keys:
        hash_values = [random.randint(0, m-1) for _ in range(d)]
        edge = tuple((copy, val) for copy, val in enumerate(hash_values))
        G.add_edge(edge[0], edge[1])
    
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def compute_mog_expectation(n: int, m: int, epsilon: float) -> float:
    """
    Compute the expected number of MOG subgraphs for fully random hash functions.
    From Lemma 16 in the paper.
    """
    expected = 0
    # Sum over t = 3 to n
    for t in range(3, n + 1):
        term = (t ** 2) / ((1 + epsilon) ** t)
        expected += term
    return (2 / m) * expected


def check_cuckoo_hashes(keys: List[int], hash_func: HashClassZ) -> bool:
    """
    Check if hash functions are suitable for cuckoo hashing.
    Returns True if all components have at most one cycle.
    """
    G = build_hash_graph(keys, hash_func)
    
    # Check each connected component
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        # Count cycles: |E| - |V| + 1 for connected graph
        # Component has at most one cycle iff |E| <= |V|
        if subgraph.number_of_edges() > subgraph.number_of_nodes():
            return False
    
    return True


def analyze_cuckoo_hashing(n: int, m: int, ell: int, c: int, 
                          keys: List[int]) -> Dict[str, Any]:
    """
    Analyze cuckoo hashing with hash class Z.
    Implements the analysis from the paper (Theorem 14).
    """
    hash_func = HashClassZ(m=m, ell=ell, c=c, d=2, universe_size=len(keys))
    
    # Build graph
    G = build_hash_graph(keys, hash_func)
    
    # Find MOG subgraphs
    mog_graphs = find_minimal_obstruction_graphs(G)
    
    # Calculate expected MOG count for fully random case
    epsilon = m / n - 1
    expected_mog = compute_mog_expectation(n, m, max(epsilon, 0.01))
    
    # Check suitability
    is_suitable = check_cuckoo_hashes(keys, hash_func)
    
    # Check for bad events
    has_bad_event = False
    for mog in mog_graphs:
        if is_bad(hash_func, set(mog.nodes())):
            has_bad_event = True
            break
    
    return {
        'is_suitable': is_suitable,
        'num_mog': len(mog_graphs),
        'expected_mog': expected_mog,
        'has_bad_event': has_bad_event,
        'deficiency': get_deficiency(hash_func, set(range(n))),
        'graph': G
    }


def test_hash_class():
    """Test the hash class implementation with a simple example."""
    # Parameters
    n = 100  # number of keys
    m = 110  # table size (slightly larger than n)
    ell = 50  # range of g functions
    c = 3    # number of g functions
    
    # Generate keys
    keys = list(range(n))
    
    # Create hash function
    hash_func = HashClassZ(m=m, ell=ell, c=c, d=2, universe_size=n * 2)
    
    # Build graph
    G = build_hash_graph(keys, hash_func)
    
    print(f"Number of vertices: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Number of components: {nx.number_connected_components(G)}")
    
    # Check suitability for cuckoo hashing
    is_suitable = check_cuckoo_hashes(keys, hash_func)
    print(f"Suitable for cuckoo hashing: {is_suitable}")
    
    # Analyze
    result = analyze_cuckoo_hashing(n, m, ell, c, keys)
    print(f"Number of MOG subgraphs: {result['num_mog']}")
    print(f"Expected MOG (fully random): {result['expected_mog']:.4f}")
    print(f"Has bad event: {result['has_bad_event']}")
    
    return result


def test_deficiency_properties():
    """Test the deficiency properties from Lemma 5."""
    n = 20
    m = 30
    ell = 15
    c = 2
    
    hash_func = HashClassZ(m=m, ell=ell, c=c, d=2, universe_size=n)
    keys = list(range(n))
    
    # Test deficiency for different subsets
    subsets = [
        set(range(5)),
        set(range(10)),
        set(range(15)),
        set(range(20))
    ]
    
    for subset in subsets:
        T = set(keys[i] for i in subset)
        d_T = get_deficiency(hash_func, T)
        good = is_good(hash_func, T)
        bad = is_bad(hash_func, T)
        critical = is_critical(hash_func, T)
        
        print(f"|T|={len(T)}, d_T={d_T}, good={good}, bad={bad}, critical={critical}")
        
        # Bound from Lemma 5(b)
        prob_bound = (len(T) ** 2 / ell) ** c
        print(f"  Probability bound from Lemma 5(b): {prob_bound:.4f}")


if __name__ == "__main__":
    # Run tests
    print("=" * 50)
    print("Testing Hash Class Z Implementation")
    print("=" * 50)
    test_hash_class()
    
    print("\n" + "=" * 50)
    print("Testing Deficiency Properties")
    print("=" * 50)
    test_deficiency_properties()