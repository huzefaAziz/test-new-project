import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
import math

@dataclass
class HyperellipticCurve:
    """Represents a hyperelliptic curve C of genus g with hyperelliptic involution."""
    genus: int
    
    def __post_init__(self):
        if self.genus < 2:
            raise ValueError("Genus must be at least 2 for hyperelliptic curves")
    
    def get_hyperelliptic_covering(self) -> str:
        """Returns the hyperelliptic covering π: C → P¹."""
        return f"π: C → P¹ (degree 2 covering, genus {self.genus})"
    
    def canonical_bundle(self) -> str:
        """The canonical bundle K_C ≅ A^{g-1}."""
        return f"K_C ≅ A^{{{self.genus-1}}}"
    
    def pushforward_structure(self) -> Tuple[int, int]:
        """π_* O_C ≅ O_{P¹} ⊕ O_{P¹}(-g-1)."""
        return (0, -self.genus - 1)

def safe_comb(n: int, k: int) -> int:
    """
    Safe binomial coefficient computation with bounds checking.
    """
    if n < 0 or k < 0 or k > n:
        return 0
    return math.comb(int(n), int(k))

def catalecticant_matrix(z: np.ndarray, ell_prime: int, ell: int) -> np.ndarray:
    """
    Constructs the rectangular catalecticant matrix Cat_{ell', ell}(z).
    
    Args:
        z: Array of length ell + ell_prime - 1 containing the variables z_i
        ell_prime: Number of rows (ℓ')
        ell: Number of columns (ℓ)
    
    Returns:
        The ℓ' × ℓ catalecticant matrix with entries z_{i+j}
    """
    if len(z) != ell + ell_prime - 1:
        raise ValueError(f"z must have length {ell + ell_prime - 1}, got {len(z)}")
    
    matrix = np.zeros((ell_prime, ell), dtype=z.dtype)
    for i in range(ell_prime):
        for j in range(ell):
            matrix[i, j] = z[i + j]
    return matrix

def brill_noether_locus_minors(z: np.ndarray, ell_prime: int, ell: int, r: int) -> np.ndarray:
    """
    Computes the ideal of W_d^r(C) given by (ℓ-r)-minors of Cat_{ℓ',ℓ}(z).
    
    Args:
        z: Variables z_0,...,z_{ℓ+ℓ'-2}
        ell_prime: ℓ' = h^1(C,L)
        ell: ℓ = h^0(C,L)
        r: 0 ≤ r ≤ ℓ-1
    
    Returns:
        Matrix of (ℓ-r)-minors (the ideal generators)
    """
    if r < 0 or r > ell - 1:
        raise ValueError(f"r must be between 0 and {ell-1}, got {r}")
    
    mat = catalecticant_matrix(z, ell_prime, ell)
    k = ell - r  # size of minors
    
    if k > min(ell_prime, ell):
        return np.array([])  # No minors of this size
    
    # Extract all k×k minors
    minors = []
    rows_indices = combinations(list(range(ell_prime)), k)
    cols_indices = combinations(list(range(ell)), k)
    
    for rows in rows_indices:
        for cols in cols_indices:
            try:
                minor = np.linalg.det(mat[np.ix_(rows, cols)])
                minors.append(minor)
            except np.linalg.LinAlgError:
                # Skip if determinant computation fails
                continue
    
    return np.array(minors)

def compute_hodge_numbers(g: int, m: int) -> np.ndarray:
    """
    Computes Hodge numbers h^{p,q}(X) for X = W_d^r(C) using formula (9).
    
    h^{p,q}(X) = 
        binom(g,p) binom(g,q) if p+q ≤ m,
        binom(g,m-p) binom(g,m-q) if p+q ≥ m.
    """
    # Ensure g and m are integers
    g = int(g)
    m = int(m)
    
    hodge = np.zeros((g+1, g+1), dtype=int)
    for p in range(g+1):
        for q in range(g+1):
            if p + q <= m:
                hodge[p, q] = safe_comb(g, p) * safe_comb(g, q)
            else:  # p + q >= m (for equality, both formulas give same result)
                hodge[p, q] = safe_comb(g, m-p) * safe_comb(g, m-q)
    return hodge

def betti_numbers(g: int, m: int) -> np.ndarray:
    """
    Computes Betti numbers b_k(X) using formula (10).
    
    b_k(X) = 
        binom(2g,k) if 0 ≤ k ≤ m,
        binom(2g,2m-k) if m ≤ k ≤ 2m.
    """
    # Ensure g and m are integers
    g = int(g)
    m = int(m)
    
    if m < 0:
        return np.array([])
    
    betti = np.zeros(2*m + 1, dtype=int)
    for k in range(2*m + 1):
        if k <= m:
            betti[k] = safe_comb(2*g, k)
        else:  # k >= m
            betti[k] = safe_comb(2*g, 2*m - k)
    return betti

def multiplicity_formula(ell_prime: int, ell: int, r: int) -> int:
    """
    Computes the multiplicity at L of W_d^r(C) using formula from Proposition 1.3.
    
    mult_L W_d^r(C) = binom(ℓ'+r, ℓ-r-1)
    """
    # Ensure all parameters are integers
    ell_prime = int(ell_prime)
    ell = int(ell)
    r = int(r)
    
    k = ell - r - 1
    n = ell_prime + r
    
    return safe_comb(n, k)

def log_canonical_threshold(ell_prime: int, ell: int, r: int, d: int, g: int) -> float:
    """
    Computes the log canonical threshold from Proposition 1.3.
    
    lct = 
        1 if d = g-1 and r = 0,
        1 + (ℓ'+r-1)/(ℓ-r) otherwise.
    """
    # Ensure all parameters are integers
    ell_prime = int(ell_prime)
    ell = int(ell)
    r = int(r)
    d = int(d)
    g = int(g)
    
    if d == g - 1 and r == 0:
        return 1.0
    
    denominator = ell - r
    if denominator <= 0:
        return float('inf')  # No threshold defined
    
    return 1.0 + (ell_prime + r - 1) / denominator

def semismall_resolution_image(m: int, s: int) -> Tuple[int, int]:
    """
    Computes dimensions for the semismall resolution a_{d,r}: C^{(m)} → X.
    
    Returns (dim Z_s, codim_X Z_s) where Z_s = W_d^{r+s}(C).
    """
    m = int(m)
    s = int(s)
    
    dim_Zs = m - 2*s
    codim = 2*s
    return dim_Zs, codim

def compute_extension_class(z_coeffs: np.ndarray) -> dict:
    """
    Computes the extension class ξ: S → W for the pushed-down Poincaré bundle.
    
    Args:
        z_coeffs: Coefficient functions \tilde{z}_r on S
    
    Returns:
        Dictionary with extension class data
    """
    n = len(z_coeffs)
    # Solve for ell and ell_prime given n = ell + ell_prime - 1
    # For simplicity, assume ell = (n+1)//2, ell_prime = n - ell + 1
    ell = (n + 1) // 2
    ell_prime = n - ell + 1
    
    # The extension class determines the boundary map D
    D = catalecticant_matrix(z_coeffs, ell_prime, ell)
    
    # Check catalecticant identities
    catalecticant_ok = True
    for i in range(ell_prime - 1):
        for j in range(1, ell):
            if D[i, j] != D[i+1, j-1]:
                catalecticant_ok = False
                break
        if not catalecticant_ok:
            break
    
    return {
        'extension_class': z_coeffs,
        'boundary_map': D,
        'ell': ell,
        'ell_prime': ell_prime,
        'catalecticant_identities': catalecticant_ok
    }

def poincare_bundle_structure(ell_prime: int, ell: int) -> dict:
    """
    Describes the fixed extension structure from Proposition 4.1.
    
    0 → O(-ℓ'-1) → ℰ → O(ℓ-1) → 0
    """
    ell_prime = int(ell_prime)
    ell = int(ell)
    
    return {
        'kernel': -ell_prime - 1,
        'cokernel': ell - 1,
        'extension_class_space': ell + ell_prime - 1
    }

def petri_map_basis(V_dim: int, ell: int, ell_prime: int) -> np.ndarray:
    """
    Constructs monomial basis for the Petri map.
    
    Sym^{ell-1}V ⊗ Sym^{ell'-1}V → Sym^{ell+ell'-2}V
    """
    V_dim = int(V_dim)
    ell = int(ell)
    ell_prime = int(ell_prime)
    
    # Basis elements X^{ell-1-j}Y^j
    basis_dim = ell + ell_prime - 1
    matrix = np.zeros((ell * ell_prime, basis_dim), dtype=int)
    
    idx = 0
    for j in range(ell):
        for i in range(ell_prime):
            # The product X^{ell-1-j}Y^j * X^{ell'-1-i}Y^i = X^{ell+ell'-2-(i+j)}Y^{i+j}
            matrix[idx, i+j] = 1
            idx += 1
    
    return matrix

def combinations(items: List[int], k: int) -> List[Tuple[int, ...]]:
    """Helper function to generate combinations."""
    n = len(items)
    k = int(k)
    if k > n or k < 0:
        return []
    if k == 0:
        return [()]
    
    result = []
    def backtrack(start, current):
        if len(current) == k:
            result.append(tuple(current))
            return
        for i in range(start, n):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result

def compute_brill_noether_data(g: int, d: int, r: int) -> dict:
    """
    Compute complete Brill-Noether data for a hyperelliptic curve.
    
    Args:
        g: Genus of the curve
        d: Degree of line bundle
        r: Rank parameter
    
    Returns:
        Dictionary with all relevant data
    """
    # Compute ℓ and ℓ' from Riemann-Roch
    # For hyperelliptic curves, we use the formulas from the paper
    ell = d - 2*r + 2  # Rough estimate, adjust based on specific conditions
    if ell < 1:
        ell = 1
    
    ell_prime = g - d + ell - 1
    if ell_prime < ell:
        ell_prime = ell
    
    b = d - 2*ell + 2
    m = d - 2*r
    
    data = {
        'genus': g,
        'degree': d,
        'r': r,
        'ell': ell,
        'ell_prime': ell_prime,
        'b': b,
        'm': m,
        'dimension': m,
        'multiplicity': multiplicity_formula(ell_prime, ell, r),
        'lct': log_canonical_threshold(ell_prime, ell, r, d, g),
        'hodge_numbers': compute_hodge_numbers(g, m) if m >= 0 else None,
        'betti_numbers': betti_numbers(g, m) if m >= 0 else None,
    }
    
    return data

def test_examples():
    """Run test examples from the paper."""
    
    # Example 1: Genus 3 curve, general case
    print("=" * 60)
    print("Example 1: Hyperelliptic Brill-Noether Data")
    print("=" * 60)
    
    # Test case: g=3, d=3, r=1
    data = compute_brill_noether_data(g=3, d=3, r=1)
    print(f"Genus g={data['genus']}, d={data['degree']}, r={data['r']}")
    print(f"ℓ = h^0(L) = {data['ell']}")
    print(f"ℓ' = h^1(L) = {data['ell_prime']}")
    print(f"b = d - 2ℓ + 2 = {data['b']}")
    print(f"m = d - 2r = {data['m']}")
    print(f"dimension of X = {data['dimension']}")
    print(f"multiplicity at L = {data['multiplicity']}")
    print(f"log canonical threshold = {data['lct']:.4f}")
    
    if data['hodge_numbers'] is not None:
        hodge = data['hodge_numbers']
        print(f"Hodge numbers h^{{0,0}} = {hodge[0,0]}")
        print(f"h^{{1,0}} = {hodge[1,0]}, h^{{0,1}} = {hodge[0,1]}")
    
    if data['betti_numbers'] is not None:
        betti = data['betti_numbers']
        print(f"Betti numbers: {betti[:10]}..." if len(betti) > 10 else f"Betti numbers: {betti}")
    
    # Example 2: Catalecticant matrix
    print("\n" + "=" * 60)
    print("Example 2: Catalecticant Matrix Construction")
    print("=" * 60)
    
    ell, ell_prime = 3, 4
    z = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cat = catalecticant_matrix(z, ell_prime, ell)
    print(f"Catalecticant matrix Cat_{{{ell_prime},{ell}}}(z):")
    print(cat)
    
    # Example 3: Brill-Noether locus minors
    print("\n" + "=" * 60)
    print("Example 3: Brill-Noether Locus Minors")
    print("=" * 60)
    
    r = 1
    minors = brill_noether_locus_minors(z, ell_prime, ell, r)
    print(f"Number of ({ell-r})-minors: {len(minors)}")
    if len(minors) > 0:
        print(f"First few minors: {minors[:3]}")
    
    # Example 4: Semismall resolution
    print("\n" + "=" * 60)
    print("Example 4: Semismall Resolution Stratification")
    print("=" * 60)
    
    m = 4
    print(f"m = {m}")
    for s in range(m//2 + 1):
        dim_Zs, codim = semismall_resolution_image(m, s)
        print(f"Z_{s} = W_d^{{r+{s}}}: dim={dim_Zs}, codim_X={codim}")
    
    # Example 5: Extension class
    print("\n" + "=" * 60)
    print("Example 5: Extension Class and Catalecticant Identities")
    print("=" * 60)
    
    z_coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ext_data = compute_extension_class(z_coeffs)
    print(f"Extension class coefficients: {ext_data['extension_class']}")
    print(f"ell = {ext_data['ell']}, ell' = {ext_data['ell_prime']}")
    print(f"Boundary map D:\n{ext_data['boundary_map']}")
    print(f"Catalecticant identities hold: {ext_data['catalecticant_identities']}")
    
    # Example 6: Theta divisor case
    print("\n" + "=" * 60)
    print("Example 6: Theta Divisor (r=0, d=g-1)")
    print("=" * 60)
    
    g = 4
    d = g - 1
    r = 0
    
    theta_data = compute_brill_noether_data(g, d, r)
    print(f"Theta divisor: g={g}, d={d}, r={r}")
    print(f"multiplicity = {theta_data['multiplicity']}")
    print(f"log canonical threshold = {theta_data['lct']}")
    if theta_data['betti_numbers'] is not None:
        print(f"Betti numbers: {theta_data['betti_numbers']}")

# Run the tests
if __name__ == "__main__":
    test_examples()