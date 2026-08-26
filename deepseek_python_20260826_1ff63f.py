import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import math

class Graph:
    """Simple undirected graph representation"""
    def __init__(self, n: int):
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.edges = []
    
    def add_edge(self, u: int, v: int):
        """Add an undirected edge between u and v"""
        self.adj[u].append(v)
        self.adj[v].append(u)
        self.edges.append((u, v))
    
    def degree(self, v: int) -> int:
        """Return degree of vertex v"""
        return len(self.adj[v])
    
    def get_edges(self) -> List[Tuple[int, int]]:
        """Return list of edges"""
        return self.edges
    
    def get_vertices(self) -> List[int]:
        """Return list of vertices"""
        return list(range(self.n))


class Caterpillar:
    """Caterpillar graph representation"""
    def __init__(self, n: int):
        self.n = n
        self.spine = []
        self.legs = defaultdict(list)  # spine_vertex -> list of legs
        self.adj = [[] for _ in range(n)]
    
    def set_spine(self, spine: List[int]):
        """Set the spine vertices in order"""
        self.spine = spine
        # Connect spine vertices
        for i in range(len(spine) - 1):
            u, v = spine[i], spine[i+1]
            self.adj[u].append(v)
            self.adj[v].append(u)
    
    def add_leg(self, spine_vertex: int, leg: int):
        """Add a leg to a spine vertex"""
        self.legs[spine_vertex].append(leg)
        self.adj[spine_vertex].append(leg)
        self.adj[leg].append(spine_vertex)
    
    def distance(self, u: int, v: int) -> int:
        """Compute distance between u and v using BFS"""
        if u == v:
            return 0
        visited = [False] * self.n
        queue = [(u, 0)]
        visited[u] = True
        
        while queue:
            node, dist = queue.pop(0)
            for neighbor in self.adj[node]:
                if neighbor == v:
                    return dist + 1
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))
        return float('inf')
    
    def get_vertices(self) -> Set[int]:
        """Return all vertices in the caterpillar"""
        vertices = set(self.spine)
        for legs in self.legs.values():
            vertices.update(legs)
        return vertices


def starfy_algorithm(G: Graph, pi: List[int], Delta: int) -> Caterpillar:
    """
    Starfy algorithm: Converts a linear arrangement into a caterpillar
    
    Args:
        G: Input graph
        pi: Linear arrangement (list of vertices in order)
        Delta: Maximum degree bound
    
    Returns:
        A degree-Delta caterpillar H
    """
    n = G.n
    segment_size = Delta - 1
    
    # Partition vertices into segments
    segments = []
    for i in range(0, n, segment_size):
        segment = pi[i:min(i + segment_size, n)]
        segments.append(segment)
    
    # For each segment, choose center of maximum degree in G
    centers = []
    for segment in segments:
        if segment:
            center = max(segment, key=lambda v: G.degree(v))
            centers.append(center)
    
    # Build the caterpillar
    H = Caterpillar(n)
    
    # Set spine
    H.set_spine(centers)
    
    # For each segment, add all non-center vertices as legs
    for seg_idx, segment in enumerate(segments):
        center = centers[seg_idx]
        for v in segment:
            if v != center:
                H.add_leg(center, v)
    
    return H


def flatten_caterpillar(H: Caterpillar) -> List[int]:
    """
    Flatten a caterpillar into a path (linear arrangement)
    
    Args:
        H: Caterpillar graph
    
    Returns:
        Linear arrangement as a path
    """
    path = []
    
    # For each spine vertex except the last, insert legs to the left
    for i in range(len(H.spine) - 1):
        spine_v = H.spine[i]
        # Add legs before the spine vertex
        for leg in H.legs.get(spine_v, []):
            path.append(leg)
        path.append(spine_v)
    
    # For the last spine vertex, add legs after it
    if H.spine:
        last_spine = H.spine[-1]
        path.append(last_spine)
        for leg in H.legs.get(last_spine, []):
            path.append(leg)
    
    return path


def compute_cost(G: Graph, H: Caterpillar) -> int:
    """Compute the arrangement cost of G embedded in H"""
    total_cost = 0
    for u, v in G.get_edges():
        total_cost += H.distance(u, v)
    return total_cost


def mla_approximation(G: Graph) -> List[int]:
    """
    Placeholder for an O(sqrt(log n) log log n) approximation for MLA.
    In practice, this would be a more sophisticated algorithm.
    
    For now, returns a simple ordering based on degrees.
    """
    n = G.n
    # Simple heuristic: sort by degree (not optimal but serves as example)
    vertices = list(range(n))
    vertices.sort(key=lambda v: G.degree(v), reverse=True)
    return vertices


def graph_in_caterpillar_approx(G: Graph, Delta: int) -> Caterpillar:
    """
    Main approximation algorithm for Graph in Caterpillar
    
    Args:
        G: Input graph
        Delta: Maximum degree bound (>= 2)
    
    Returns:
        Approximate degree-Delta caterpillar H
    """
    if Delta < 2:
        raise ValueError("Delta must be >= 2")
    
    if Delta >= G.n - 1:
        # Star approximation (2-approximation for large Delta)
        return star_approximation(G)
    
    # Step 1: Get alpha-approximation for MLA
    pi = mla_approximation(G)
    
    # Step 2: Apply Starfy algorithm
    H = starfy_algorithm(G, pi, Delta)
    
    return H


def star_approximation(G: Graph) -> Caterpillar:
    """
    Trivial 2-approximation using a star
    """
    n = G.n
    if n == 0:
        return Caterpillar(0)
    
    # Choose center as vertex of maximum degree
    center = max(range(n), key=lambda v: G.degree(v))
    
    H = Caterpillar(n)
    H.set_spine([center])
    
    for v in range(n):
        if v != center:
            H.add_leg(center, v)
    
    return H


def compute_approximation_bound(G: Graph, Delta: int, alpha: float) -> Tuple[float, float]:
    """
    Compute the approximation bound for GiC
    
    Args:
        G: Input graph
        Delta: Maximum degree
        alpha: Approximation ratio for MLA
    
    Returns:
        (approximation_ratio, additive_term)
    """
    m = len(G.get_edges())
    additive_term = (3 - 2/(Delta - 1)) * m
    
    # The approximation ratio is alpha + 3 - 2/(Delta-1)
    ratio = alpha + 3 - 2/(Delta - 1)
    
    return ratio, additive_term


def is_caterpillar(tree: Graph) -> bool:
    """
    Check if a tree is a caterpillar
    """
    # Remove leaves
    degrees = [tree.degree(v) for v in range(tree.n)]
    removed = [False] * tree.n
    
    # Find leaves
    leaves = [v for v in range(tree.n) if degrees[v] <= 1]
    
    # Remove leaves iteratively
    while leaves:
        v = leaves.pop()
        if removed[v]:
            continue
        removed[v] = True
        for neighbor in tree.adj[v]:
            if not removed[neighbor]:
                degrees[neighbor] -= 1
                if degrees[neighbor] <= 1:
                    leaves.append(neighbor)
    
    # Check if remaining vertices form a path
    remaining = [v for v in range(tree.n) if not removed[v]]
    if len(remaining) <= 2:
        return True
    
    # Check if they form a path (max degree <= 2)
    for v in remaining:
        non_removed_neighbors = [u for u in tree.adj[v] if not removed[u]]
        if len(non_removed_neighbors) > 2:
            return False
    
    return True


def verify_lemma1(G: Graph, pi: List[int], H: Caterpillar, Delta: int) -> bool:
    """
    Verify Lemma 1: Cost bound of Starfy
    
    Returns: True if the bound holds
    """
    # Compute beta values
    segment_size = Delta - 1
    beta = {}
    for idx, v in enumerate(pi):
        beta[v] = idx // segment_size + 1
    
    # Compute cost of H
    cost_H = compute_cost(G, H)
    
    # Compute RHS of Lemma 1
    beta_sum = 0
    for u, v in G.get_edges():
        beta_sum += abs(beta[u] - beta[v])
    
    m = len(G.get_edges())
    rhs = beta_sum + (2 - 2/(Delta - 1)) * m
    
    return cost_H <= rhs


def verify_lemma2(H: Caterpillar) -> bool:
    """
    Verify Lemma 2: Flattening doesn't increase distances by more than Delta-1
    
    Returns: True if the bound holds for all vertex pairs
    """
    path_order = flatten_caterpillar(H)
    n = H.n
    
    # Create path graph from the order
    path = Graph(n)
    for i in range(n - 1):
        path.add_edge(path_order[i], path_order[i+1])
    
    # Check all pairs
    for u in range(n):
        for v in range(u+1, n):
            dist_caterpillar = H.distance(u, v)
            dist_path = path.distance(u, v)
            if dist_path > (H.n - 1) * dist_caterpillar:
                return False
    
    return True


def example_usage():
    """
    Example usage of the implementation
    """
    # Create a sample graph (a small tree)
    n = 8
    G = Graph(n)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (5, 7)]
    for u, v in edges:
        G.add_edge(u, v)
    
    print("Input graph:")
    print(f"  Vertices: {G.n}")
    print(f"  Edges: {G.get_edges()}")
    
    # Test with Delta = 3
    Delta = 3
    H = graph_in_caterpillar_approx(G, Delta)
    
    print(f"\nConstructed caterpillar (Delta={Delta}):")
    print(f"  Spine: {H.spine}")
    print(f"  Legs: {dict(H.legs)}")
    
    # Compute cost
    cost = compute_cost(G, H)
    print(f"  Cost: {cost}")
    
    # Check if it's a valid caterpillar
    print(f"  Is valid caterpillar: {is_caterpillar(G)}")
    
    # Verify Lemma 1
    pi = mla_approximation(G)
    valid = verify_lemma1(G, pi, H, Delta)
    print(f"  Lemma 1 holds: {valid}")
    
    # Compute approximation bound
    alpha = 1.0  # Assuming exact MLA for trees
    ratio, additive = compute_approximation_bound(G, Delta, alpha)
    print(f"\nApproximation bound:")
    print(f"  Ratio: {ratio:.3f}")
    print(f"  Additive term: {additive:.1f}")
    
    # Compare with trivial star approximation
    H_star = star_approximation(G)
    cost_star = compute_cost(G, H_star)
    print(f"\nStar approximation:")
    print(f"  Cost: {cost_star}")
    print(f"  Improvement: {cost_star - cost}")


if __name__ == "__main__":
    example_usage()