import numpy as np
import random

class LCE:
    """
    Longest Common Extension data structure using rolling hash.
    For deterministic LCE, replace with suffix array + RMQ.
    """
    def __init__(self, word):
        self.word = word
        self.n = len(word)
        self.base = 911382323
        self.mod = 10**9 + 7
        self.pow = [1] * (self.n + 1)
        self.hash = [0] * (self.n + 1)
        for i in range(self.n):
            self.pow[i+1] = (self.pow[i] * self.base) % self.mod
            self.hash[i+1] = (self.hash[i] * self.base + ord(word[i])) % self.mod

    def get_hash(self, l, r):
        """Hash of word[l:r]"""
        return (self.hash[r] - self.hash[l] * self.pow[r-l]) % self.mod

    def lce(self, i, j):
        """Length of longest common prefix of suffixes starting at i and j"""
        lo, hi = 0, self.n - max(i, j)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.get_hash(i, i+mid) == self.get_hash(j, j+mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

def build_adj_word(adj):
    """Convert adjacency matrix to binary word for LCE"""
    n = len(adj)
    word = ''.join('1' if adj[i][j] else '0' for i in range(n) for j in range(n))
    return word

def sd_check(lce, n, u, v, d):
    """
    Check if sd_G(u,v) <= d using LCE queries.
    Implements Theorem 16.
    """
    p = 1
    q = d
    while p <= n and q >= 0:
        k = lce.lce((u-1)*n + (p-1), (v-1)*n + (p-1))
        if p + k <= n:
            q -= 1
            p += k + 1
        else:
            break
    return q >= 0

def list_near_twins(adj, d):
    """
    List all pairs of vertices with symmetric difference <= d.
    Implements Corollary 17.
    """
    n = len(adj)
    word = build_adj_word(adj)
    lce = LCE(word)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if sd_check(lce, n, i+1, j+1, d):
                pairs.append((i, j))
    return pairs

def compute_sd_degeneracy(adj, d, n_alpha=0):
    """
    Compute sd-degeneracy sequence (simplified).
    Implements Theorem 18 for constant d.
    """
    n = len(adj)
    # Ensure n_alpha is integer for range
    max_diff = d * (n ** (n_alpha/10)) if n_alpha > 0 else d
    sequence = []
    remaining = list(range(n))
    while len(remaining) > 1:
        # Build subgraph adjacency (simplified)
        sub_adj = adj[np.ix_(remaining, remaining)]
        pairs = list_near_twins(sub_adj, int(max_diff))
        if not pairs:
            break
        # Greedy select disjoint pairs (simplified)
        selected = []
        used = set()
        for u, v in pairs:
            if u not in used and v not in used:
                selected.append((remaining[u], remaining[v]))
                used.add(u); used.add(v)
        # Remove first from each pair
        for u, v in selected:
            sequence.append((u, v))
            remaining.remove(u)
        # Recompute max_diff for current size (optional)
        max_diff = d * (len(remaining) ** (n_alpha/10)) if n_alpha > 0 else d
    return sequence

def interval_biclique_partition(adj):
    """
    Placeholder for interval biclique partition (Lemma 13).
    In practice, would use signed tree models.
    """
    n = len(adj)
    # Dummy: create one biclique per edge
    bicliques = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                bicliques.append(((i, i), (j, j)))
    return bicliques

def sssp_interval_biclique(n, bicliques, source):
    """
    SSSP using interval biclique partition (Theorem 5).
    Simplified version without the full data structure.
    """
    dist = [float('inf')] * n
    dist[source] = 0
    queue = [source]
    visited = [False] * n
    visited[source] = True
    
    while queue:
        u = queue.pop(0)
        # For each biclique containing u, discover all vertices on the other side
        for (a_l, a_r), (b_l, b_r) in bicliques:
            if a_l <= u <= a_r:
                for v in range(b_l, b_r+1):
                    if not visited[v]:
                        visited[v] = True
                        dist[v] = dist[u] + 1
                        queue.append(v)
            elif b_l <= u <= b_r:
                for v in range(a_l, a_r+1):
                    if not visited[v]:
                        visited[v] = True
                        dist[v] = dist[u] + 1
                        queue.append(v)
    return dist

def matrix_vector_multiply(adj, vector):
    """
    Multiply adjacency matrix by vector using biclique partition (Lemma 14).
    """
    n = len(adj)
    result = np.zeros(n)
    bicliques = interval_biclique_partition(adj)
    # For each biclique, add vector elements of one side to the other
    for (a_l, a_r), (b_l, b_r) in bicliques:
        sum_b = sum(vector[b_l:b_r+1])
        for i in range(a_l, a_r+1):
            result[i] += sum_b
        sum_a = sum(vector[a_l:a_r+1])
        for i in range(b_l, b_r+1):
            result[i] += sum_a
    return result

# Example usage
if __name__ == "__main__":
    # Create a simple graph (e.g., cycle of 5 vertices)
    n = 5
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        adj[i][(i+1)%n] = 1
        adj[(i+1)%n][i] = 1
    
    print("Adjacency matrix:")
    print(adj)
    
    # Compute near-twin pairs (d=2)
    d = 2
    pairs = list_near_twins(adj, d)
    print(f"\nPairs with symmetric difference <= {d}: {pairs}")
    
    # Compute sd-degeneracy sequence
    seq = compute_sd_degeneracy(adj, d)
    print(f"\nSD-degeneracy sequence: {seq}")
    
    # Simulate SSSP using biclique partition
    bicliques = interval_biclique_partition(adj)
    distances = sssp_interval_biclique(n, bicliques, 0)
    print(f"\nDistances from vertex 0: {distances}")
    
    # Matrix-vector multiplication
    vec = np.ones(n)
    result = matrix_vector_multiply(adj, vec)
    print(f"\nM * ones: {result}")