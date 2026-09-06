import itertools
import math
import random
from typing import List, Tuple, Dict, Set, Optional

# ------------------------------------------------------------
# 1. Basic definitions: grid, tokens, permutations
# ------------------------------------------------------------

class Token:
    """A token represents a move in a specific direction and layer (scale)."""
    def __init__(self, direction: int, layer: int):
        self.direction = direction  # coordinate index
        self.layer = layer          # scale: 4^layer

    @property
    def vector(self) -> Tuple[int, ...]:
        """The actual vector added to the spine."""
        vec = [0] * (self.direction + 1)  # direction is 0-based
        vec[self.direction] = 4 ** self.layer
        return tuple(vec)

    def __repr__(self):
        return f"T(d={self.direction}, l={self.layer})"

    def __eq__(self, other):
        return self.direction == other.direction and self.layer == other.layer

    def __hash__(self):
        return hash((self.direction, self.layer))


def generate_tokens(k: int, U: int) -> List[Token]:
    """Generate all tokens for dimension k and max layer U.
       Total tokens = k * (U+1)."""
    return [Token(d, r) for d in range(k) for r in range(U+1)]


def is_balanced_permutation(perm: List[Token]) -> bool:
    """Check if a permutation of tokens is balanced:
       for every i, the i-th prefix token has the same layer as the i-th suffix token (in reverse)."""
    m = len(perm) // 2
    for i in range(m):
        if perm[i].layer != perm[-1 - i].layer:
            return False
    return True


def all_balanced_permutations(tokens: List[Token]) -> List[List[Token]]:
    """Generate all balanced permutations (for small token sets only)."""
    # This is for demonstration; for large sets it's infeasible.
    # We'll generate all permutations and filter.
    if len(tokens) % 2 != 0:
        raise ValueError("Number of tokens must be even for balanced permutations.")
    all_perms = itertools.permutations(tokens)
    return [list(p) for p in all_perms if is_balanced_permutation(p)]


# ------------------------------------------------------------
# 2. Spine and projection
# ------------------------------------------------------------

def spine_from_permutation(perm: List[Token]) -> List[Tuple[int, ...]]:
    """Compute the spine vertices from a token permutation."""
    vertices = []
    pos = [0] * (max(t.direction for t in perm) + 1)  # dimension = max direction + 1
    vertices.append(tuple(pos))
    for t in perm:
        vec = t.vector
        for i, val in enumerate(vec):
            pos[i] += val
        vertices.append(tuple(pos))
    return vertices


def project_to_spine(point: Tuple[int, ...], spine: List[Tuple[int, ...]], 
                     W: int) -> int:
    """
    Project a point onto the spine.
    Returns the index i such that spine[i] is the projection.
    For lower half (norm <= W/2): take largest i with spine[i] <= point coordinatewise.
    For upper half: take smallest i with point <= spine[i].
    """
    norm = sum(point)
    k = len(point)
    B = sum(spine[-1]) // k  # maximum coordinate (since spine ends at B*1)
    W = k * B
    if norm <= W // 2:
        # lower half: greatest index with spine[i] <= point
        idx = 0
        for i, v in enumerate(spine):
            if all(v[j] <= point[j] for j in range(k)):
                idx = i
            else:
                break
        return idx
    else:
        # upper half: smallest index with point <= spine[i]
        idx = len(spine) - 1
        for i in range(len(spine)-1, -1, -1):
            if all(point[j] <= spine[i][j] for j in range(k)):
                idx = i
            else:
                break
        return idx


def move_toward_midpoint(i: int, m: int) -> int:
    """Move index one step toward the midpoint m."""
    if i < m:
        return i + 1
    elif i > m:
        return i - 1
    else:
        return m


def hard_function(perm: List[Token]) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """
    Construct the hard function f_pi on the entire grid [n]^k.
    Returns a mapping from each point to its image.
    Note: This is for small grids only; for large n the grid is huge.
    """
    spine = spine_from_permutation(perm)
    k = len(spine[0])
    B = max(spine[-1])  # since spine[-1] = B*1
    n = B + 1
    W = k * B
    m = len(spine) // 2   # midpoint index

    # Build the full grid
    grid = [tuple(coords) for coords in itertools.product(range(n), repeat=k)]
    
    f = {}
    for point in grid:
        idx = project_to_spine(point, spine, W)
        new_idx = move_toward_midpoint(idx, m)
        f[point] = spine[new_idx]
    return f


# ------------------------------------------------------------
# 3. Verification: monotonicity and fixed points
# ------------------------------------------------------------

def is_monotone(f: Dict[Tuple[int, ...], Tuple[int, ...]], n: int, k: int) -> bool:
    """Check if f is monotone w.r.t. coordinatewise order."""
    grid = [tuple(coords) for coords in itertools.product(range(n), repeat=k)]
    for a in grid:
        for b in grid:
            if all(a[i] <= b[i] for i in range(k)):
                if not all(f[a][i] <= f[b][i] for i in range(k)):
                    return False
    return True


def fixed_points(f: Dict[Tuple[int, ...], Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Return all fixed points of f."""
    return [p for p, q in f.items() if p == q]


# ------------------------------------------------------------
# 4. Example: small instance from the paper (n=22, k=2)
# ------------------------------------------------------------

def example_from_paper():
    # Parameters: k=2, U=2 (since 1+4+16=21, n=22)
    k = 2
    U = 2
    tokens = generate_tokens(k, U)
    # The specific permutation from Example 3.9:
    # [(1,2), (0,0), (1,1), (0,1), (1,0), (0,2)]
    # Token order: (direction, layer)
    perm = [
        Token(1, 2), Token(0, 0), Token(1, 1),
        Token(0, 1), Token(1, 0), Token(0, 2)
    ]
    # Check balanced
    print("Is balanced?", is_balanced_permutation(perm))
    # Build spine
    spine = spine_from_permutation(perm)
    print("Spine vertices:", spine)
    # Build hard function (grid size 22x22 = 484 points)
    f = hard_function(perm)
    # Check monotonicity (WARNING: this is O(n^(2k)) = O(22^4) ~ 234k checks, fine)
    n = 22
    print("Is monotone?", is_monotone(f, n, k))
    # Fixed points
    fps = fixed_points(f)
    print("Fixed points:", fps)
    # Expected unique fixed point at spine[m] where m = len(spine)//2 = 3
    m = len(spine) // 2
    print("Expected fixed point:", spine[m])
    assert fps == [spine[m]]
    print("Verification passed.")


# ------------------------------------------------------------
# 5. (Optional) Skeleton for tree-filtration adversary method
#    This part is not fully executed; it outlines the data structures.
# ------------------------------------------------------------

class FiltrationTree:
    """Simplified tree for the adversary method."""
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.root = None
        # We would build the tree of prefix-suffix pairs here.
        # For small sets, we can generate all balanced permutations and their prefixes.
        pass

    # The full adversary method would construct the matrix Gamma and compute the bound.
    # This is left as a placeholder because the full computation is heavy.


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    example_from_paper()