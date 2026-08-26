import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from abc import ABC, abstractmethod
import math

# ============================================================================
# Section 2: Preliminaries on Reductive Groups
# ============================================================================

@dataclass
class Root:
    """Represents a root in the root system."""
    index: int
    vector: np.ndarray
    is_positive: bool
    
class WeylGroup:
    """Represents the Weyl group W = N_G(T)/T."""
    
    def __init__(self, rank: int):
        self.rank = rank
        self.simple_reflections = self._generate_simple_reflections()
        
    def _generate_simple_reflections(self) -> List[np.ndarray]:
        """Generate simple reflection matrices for type A_n."""
        # For simplicity, we use type A_n (GL_{n+1})
        # Each simple reflection swaps adjacent basis vectors
        reflections = []
        for i in range(self.rank):
            s = np.eye(self.rank + 1)
            s[i, i] = 0
            s[i, i+1] = 1
            s[i+1, i] = 1
            s[i+1, i+1] = 0
            reflections.append(s)
        return reflections
    
    def word_to_element(self, word: List[int]) -> np.ndarray:
        """Convert a word in simple reflections to a Weyl group element."""
        result = np.eye(self.rank + 1)
        for i in word:
            result = self.simple_reflections[i] @ result
        return result
    
    def longest_element(self) -> np.ndarray:
        """Return the longest element w_0."""
        # For type A_n, this is the anti-diagonal matrix
        return np.fliplr(np.eye(self.rank + 1))

class Torus:
    """Represents a maximal torus T."""
    
    def __init__(self, rank: int, coordinates: np.ndarray = None):
        self.rank = rank
        self.coordinates = coordinates if coordinates is not None else np.ones(rank)
        
    def eval_char(self, char: np.ndarray) -> complex:
        """Evaluate a character on the torus."""
        # For GL_n, characters are (a_1, ..., a_n) -> product a_i^{char_i}
        return np.prod(self.coordinates ** char)
    
    def eval_cochar(self, cochar: np.ndarray) -> 'Torus':
        """Apply a cocharacter to get a torus element."""
        return Torus(self.rank, self.coordinates ** cochar)

class RootSystem:
    """Represents the root system of a reductive group."""
    
    def __init__(self, rank: int):
        self.rank = rank
        self._setup_type_a(rank)
        
    def _setup_type_a(self, rank: int):
        """Setup type A_n root system."""
        n = rank + 1
        # Simple roots: e_i - e_{i+1}
        self.simple_roots = []
        for i in range(rank):
            v = np.zeros(n)
            v[i] = 1
            v[i+1] = -1
            self.simple_roots.append(v)
        
        # All positive roots: e_i - e_j for i < j
        self.positive_roots = []
        for i in range(n):
            for j in range(i+1, n):
                v = np.zeros(n)
                v[i] = 1
                v[j] = -1
                self.positive_roots.append(v)
        
        # All roots
        self.all_roots = self.positive_roots + [-r for r in self.positive_roots]
        
        # Simple coroots (for type A, same as roots)
        self.simple_coroots = [r.copy() for r in self.simple_roots]
        
    def is_positive(self, root: np.ndarray) -> bool:
        """Check if a root is positive."""
        # For type A, positive roots have first nonzero entry 1
        for i, val in enumerate(root):
            if val > 0:
                return True
            if val < 0:
                return False
        return False
    
    def root_pairing(self, coroot: np.ndarray, weight: np.ndarray) -> int:
        """Pairing between coroots and weights."""
        return int(np.dot(coroot, weight))

class ReductiveGroup:
    """Represents a reductive group G."""
    
    def __init__(self, rank: int, is_simply_connected: bool = True):
        self.rank = rank
        self.root_system = RootSystem(rank)
        self.weyl_group = WeylGroup(rank)
        self.is_simply_connected = is_simply_connected
        
        # Setup parabolic subgroups
        self._setup_parabolics()
        
    def _setup_parabolics(self):
        """Setup standard parabolic subgroups."""
        self.parabolics = {}
        for i in range(self.rank):
            # Parabolic P_i = associated to simple root α_i
            # For type A, this corresponds to block upper triangular
            self.parabolics[i] = {
                'simple_root': self.root_system.simple_roots[i],
                'subset': [i],
                'dimension': self.rank + 1,
                'levi_factor': self._get_levi_factor(i)
            }
    
    def _get_levi_factor(self, i: int) -> Dict:
        """Get the Levi factor L of the parabolic P_i."""
        # For type A, L = GL_i × GL_{n-i}
        n = self.rank + 1
        return {
            'size1': i + 1,
            'size2': n - i - 1,
            'center_dimension': 2  # Z(L) has dimension 2 for type A
        }
    
    def is_minuscule(self, i: int) -> bool:
        """Check if the parabolic P_i is minuscule."""
        # For type A_n, all fundamental coweights are minuscule
        return True
    
    def fundamental_coweight(self, i: int) -> np.ndarray:
        """Return the fundamental coweight ω_i^∨."""
        n = self.rank + 1
        omega = np.zeros(n)
        # For type A_n, ω_i = e_1 + ... + e_i normalized
        # As cocharacter: t -> diag(t, ..., t, 1, ..., 1)
        for j in range(i + 1):
            omega[j] = 1
        return omega

# ============================================================================
# Section 3: Geometric Crystals
# ============================================================================

class GeometricCrystal:
    """Represents the geometric crystal X = U Z(L) w_P U ∩ B_-."""
    
    def __init__(self, G: ReductiveGroup, P_index: int):
        self.G = G
        self.P_index = P_index
        self.P = G.parabolics[P_index]
        self.L = self.P['levi_factor']
        
        # Get w_P = w_0^P w_0
        self.w_P = self._compute_w_P(P_index)
        
        # Setup maps
        self._setup_maps()
        
    def _compute_w_P(self, P_index: int) -> np.ndarray:
        """Compute w_P = w_0^P * w_0."""
        G = self.G
        n = G.rank + 1
        i = P_index  # Use the parameter passed in
        
        # For type A with P_i, w_0^P is the longest element of W_P
        # W_P is S_{i+1} × S_{n-i-1}
        # w_0^P reverses each block
        
        # Build permutation for w_0^P
        block1 = list(range(i + 1))
        block2 = list(range(i + 1, n))
        w0P_perm = block1[::-1] + block2[::-1]
        
        # w_0 reverses everything
        w0_perm = list(range(n))[::-1]
        
        # w_P = w_0^P * w_0
        wP_perm = [w0P_perm[w0_perm[j]] for j in range(n)]
        
        # Convert permutation to matrix
        wP_matrix = np.zeros((n, n))
        for j, dest in enumerate(wP_perm):
            wP_matrix[dest, j] = 1
            
        return wP_matrix
    
    def _setup_maps(self):
        """Setup the maps π: X → Z(L), γ: X → T, f: X → A^1."""
        self.pi = self._pi_map
        self.gamma = self._gamma_map
        self.f = self._decoration
    
    def _pi_map(self, x: np.ndarray) -> np.ndarray:
        """Highest weight map π: X → Z(L)."""
        # In coordinates: x = u_1 t w_P u_2, π(x) = t
        # For type A with P_i, Z(L) = GL_1 × GL_1
        # We extract the central part
        n = len(x)
        # For block diagonal, central means scalar on each block
        # Placeholder implementation
        return np.array([x[0], x[-1]])
    
    def _gamma_map(self, x: np.ndarray) -> np.ndarray:
        """Weight map γ: X → T."""
        # γ(x) = x mod U_- ∈ B_-/U_- = T
        # Placeholder: extract diagonal entries
        return np.diag(x)
    
    def _decoration(self, x: np.ndarray) -> float:
        """Decoration f: X → A^1, f(x) = φ(u_1) + φ(u_2)."""
        # φ is the sum of simple root group coordinates
        # Placeholder
        return 0.0
    
    def enumerate_points_over_finite_field(self, q: int) -> List[Dict]:
        """
        Enumerate all points of X over F_q.
        In practice, this would be a more sophisticated enumeration.
        """
        points = []
        # For GL_n, points in X(F_q) correspond to Bruhat decomposition
        # This is a placeholder - real implementation would be complex
        n = self.G.rank + 1
        i = self.P_index
        
        # Simple enumeration for demonstration
        for t1 in range(q):
            for t2 in range(q):
                for u1 in range(q):
                    for u2 in range(q):
                        # Construct x = u1 * t * w_P * u2
                        t = np.array([t1, t2])
                        x = np.eye(n)
                        # Placeholder: build actual matrix
                        points.append({
                            'x': x,
                            'pi': self._pi_map(x),
                            'gamma': self._gamma_map(x),
                            'f': self._decoration(x)
                        })
        return points

# ============================================================================
# Section 4: Generic Representations and Bessel Functions
# ============================================================================

class FiniteField:
    """Represents the finite field F_q."""
    
    def __init__(self, q: int):
        self.q = q
        self.p = self._get_prime(q)
        self.elements = list(range(q))
        
    def _get_prime(self, q: int) -> int:
        """Get the characteristic p of F_q."""
        # Find smallest prime divisor
        for p in range(2, int(math.sqrt(q)) + 1):
            if q % p == 0:
                return p
        return q  # q is prime
    
    def additive_characters(self) -> Dict[int, complex]:
        """Generate additive characters of F_q."""
        # Characters: x -> exp(2πi * Tr_{F_q/F_p}(x) / p)
        chars = {}
        for x in range(self.q):
            # Simplified: use exp(2πi * x / p) for prime field
            chars[x] = math.e ** (2j * math.pi * x / self.p)
        return chars
    
    def multiplicative_characters(self) -> List[Dict[int, complex]]:
        """Generate multiplicative characters of F_q^×."""
        chars = []
        # For each character of the cyclic group F_q^×
        q_minus_1 = self.q - 1
        for k in range(q_minus_1):
            char = {}
            for x in range(1, self.q):
                # χ_k(x) = exp(2πi * k * log_g(x) / (q-1))
                # where g is a primitive root
                char[x] = math.e ** (2j * math.pi * k * x / q_minus_1)
            chars.append(char)
        return chars

class Character:
    """Represents a multiplicative character χ: T(F) → C×."""
    
    def __init__(self, values: Dict[Tuple[int, ...], complex]):
        self.values = values
        
    def eval(self, t: Tuple[int, ...]) -> complex:
        """Evaluate the character at t ∈ T(F)."""
        return self.values.get(t, 0j)
    
    @classmethod
    def from_torus_characters(cls, char_list: List[Dict[int, complex]]):
        """Create a character on T from individual torus characters."""
        values = {}
        # This would build the tensor product of characters
        # Placeholder implementation
        return cls({(1, 1): 1.0 + 0.0j})

class GenericRepresentation:
    """Represents a generic principal series representation."""
    
    def __init__(self, G: ReductiveGroup, chi: Character, psi: np.ndarray):
        self.G = G
        self.chi = chi
        self.psi = psi
        self.field = FiniteField(2)  # Placeholder: actual q from context
        
        # Setup induced representation Ind_{B_-}^{G}(χ)
        self._setup_induced()
        
    def _setup_induced(self):
        """Setup the induced representation."""
        self.dimension = len(self.G.weyl_group.simple_reflections) + 1
        self.whittaker_vector = self._compute_whittaker_vector()
        
    def _compute_whittaker_vector(self) -> np.ndarray:
        """Compute the ψ-Whittaker vector."""
        # v(t u' u) = χ(t) ψ(φ(u))
        # For a finite group, this is a vector in the induced representation
        size = self.dimension
        v = np.zeros(size, dtype=complex)
        
        # Placeholder: actually compute based on group structure
        for i in range(size):
            v[i] = 1.0 + 0.0j
            
        return v
    
    def bessel_function(self, g: np.ndarray) -> complex:
        """Compute the Bessel function J_{τ,ψ}(g)."""
        # J(g) = ⟨τ(g)v, v⟩ / ⟨v, v⟩
        # Placeholder: actual implementation would use group action
        return 1.0 + 0.0j

def bessel_function_sum_formula(
    crystal: GeometricCrystal,
    chi: Character,
    psi: np.ndarray,
    z: np.ndarray,
    q: int = 2
) -> complex:
    """
    Compute Bessel function using the sum formula:
    J_{τ,ψ}(z w_P) = q^{-d} Σ_{x∈X, π(x)=z} χ(γ(x)) ψ^{-1}(f(x))
    """
    d = len(crystal.G.weyl_group.simple_reflections)  # dim G/P
    
    # Get all points in the crystal fiber above z
    points = crystal.enumerate_points_over_finite_field(q)
    
    # Filter points with π(x) = z
    filtered = [p for p in points if np.array_equal(p['pi'], z)]
    
    # Compute sum
    total = 0.0 + 0.0j
    for point in filtered:
        gamma_val = tuple(point['gamma'])
        f_val = point['f']
        chi_val = chi.eval(gamma_val)
        psi_inv = math.e ** (-2j * math.pi * f_val / q)
        total += chi_val * psi_inv
    
    return q**(-d) * total

# ============================================================================
# Section 5: Weighted Character Sheaves
# ============================================================================

class WeightedCharacterSheaf:
    """
    Represents the weighted character sheaf:
    WC_G^P(φ, χ) = Rπ!(γ^*K_χ ⊗ f^*AS_{ψ^{-1}})(d/2)[d]
    """
    
    def __init__(self, crystal: GeometricCrystal, chi: Character, psi: np.ndarray):
        self.crystal = crystal
        self.chi = chi
        self.psi = psi
        self.d = len(crystal.G.weyl_group.simple_reflections)
        
    def trace(self, z: np.ndarray, q: int = 2) -> complex:
        """
        Compute trace of Frobenius at z ∈ Z(L)(F).
        This equals (-1)^d q^{d/2} J_{τ,ψ}(z w_P).
        """
        # Compute using the sum formula
        j_val = bessel_function_sum_formula(self.crystal, self.chi, self.psi, z, q)
        return (-1)**self.d * (q**(self.d/2)) * j_val

# ============================================================================
# Section 6: Kloosterman Sheaves
# ============================================================================

class AffineGrassmannian:
    """Represents the affine Grassmannian Gr = G(F((t)))/G(F[[t]])."""
    
    def __init__(self, G: ReductiveGroup):
        self.G = G
        
    def orbit(self, mu: np.ndarray) -> np.ndarray:
        """Return the orbit Gr_μ."""
        # Placeholder: compute orbit through τ^{-μ}
        return np.array([mu])
    
    def ic_sheaf(self, mu: np.ndarray):
        """Return the intersection cohomology sheaf IC_μ."""
        dim = len(mu)
        # IC_μ = j_!* (Q_l(dim/2)[dim])
        # Placeholder
        return {'dim': dim, 'tate_twist': dim/2}

class KloostermanSheaf:
    """
    Represents the Kloosterman sheaf:
    Kl_{G^}^μ(φ, χ) = Rpr_{2,!}(...)
    """
    
    def __init__(self, G: ReductiveGroup, mu: np.ndarray, chi: Character):
        self.G = G
        self.mu = mu
        self.chi = chi
        
    def trace(self, a: float, q: int = 2) -> complex:
        """
        Compute trace of Frobenius at a ∈ F×.
        This is the left side of Theorem 1.1.
        """
        # For GL_n, this relates to classical Kloosterman sums
        n = len(self.mu)
        k = int(np.sum(self.mu))  # Number of ones in the coweight
        
        # Classical Kloosterman sum for GL_n
        # Kl_n(ψ, a) = Σ_{x_1...x_n = a} ψ(x_1 + ... + x_n)
        total = 0.0 + 0.0j
        
        # Enumerate all tuples with product a
        # For q prime, this is manageable
        if q <= 10:  # For demonstration
            for x1 in range(1, q):
                for x2 in range(1, q):
                    # For n=2, x1*x2 = a mod q
                    if (x1 * x2) % q == a % q:
                        psi_val = math.e ** (2j * math.pi * (x1 + x2) / q)
                        total += psi_val
        
        return total

# ============================================================================
# Section 7: Main Theorem
# ============================================================================

class MainTheorem:
    """
    Implements Theorem 1.1:
    Kl_{G^}^{ω_i^∨}(φ, χ^{-1}; α_i(z)) = 
    χ(z_0) (-1)^{dim G/P} q^{dim G/P/2} J_{τ,ψ}(z w_P)
    """
    
    def __init__(self, G: ReductiveGroup, i: int, chi: Character):
        self.G = G
        self.i = i
        self.chi = chi
        self.crystal = GeometricCrystal(G, i)
        self.kloosterman = None
        
        # Setup components
        self.d = len(G.weyl_group.simple_reflections)  # dim G/P
        self.omega_i = G.fundamental_coweight(i)
        
    def verify_for_z(self, z: np.ndarray, q: int, psi: np.ndarray) -> bool:
        """
        Verify the theorem for a given z ∈ Z(L)(F).
        Returns True if the equality holds.
        """
        # Compute Bessel function at z w_P
        j_val = bessel_function_sum_formula(
            self.crystal, self.chi, psi, z, q
        )
        
        # Compute Kloosterman sheaf trace
        kl_sheaf = KloostermanSheaf(self.G, self.omega_i, self.chi)
        kl_val = kl_sheaf.trace(1.0, q)  # α_i(z) = 1 for z in center
        
        # Right side of theorem
        chi_z0 = self.chi.eval(tuple(z[:2]))  # z_0 in Z(G)
        right_side = (chi_z0.conjugate() * 
                     (-1)**self.d * 
                     (q**(self.d/2)) * 
                     j_val)
        
        # Compare
        tolerance = 1e-10
        return abs(kl_val - right_side) < tolerance
    
    def compute_bessel_at_special(self, z: np.ndarray, q: int) -> complex:
        """
        Compute Bessel function at special elements using the theorem.
        """
        # From theorem:
        # J(z w_P) = (-1)^{-d} q^{-d/2} χ(z_0) Kl_{G^}^{ω_i^∨}(φ, χ^{-1}; α_i(z))
        kl_sheaf = KloostermanSheaf(self.G, self.omega_i, self.chi)
        kl_val = kl_sheaf.trace(1.0, q)
        chi_z0 = self.chi.eval(tuple(z[:2]))
        
        return ((-1)**(-self.d) * 
                q**(-self.d/2) * 
                chi_z0.conjugate() * 
                kl_val)

# ============================================================================
# Section 8-9: Examples
# ============================================================================

class ExampleGLn:
    """Example implementation for G = GL_n."""
    
    def __init__(self, n: int, q: int):
        self.n = n
        self.q = q
        self.G = ReductiveGroup(n - 1)  # rank = n - 1
        self.field = FiniteField(q)
        
    def anti_diag_element(self, c: float, k: int) -> np.ndarray:
        """
        Construct element anti-diag(c I_k, I_{n-k}).
        This is the special element for GL_n.
        """
        matrix = np.zeros((self.n, self.n))
        for i in range(k):
            matrix[i, self.n - 1 - i] = c
        for i in range(k, self.n):
            matrix[i, self.n - 1 - i] = 1
        return matrix
    
    def compute_bessel_gl_n(
        self,
        c: float,
        k: int,
        chi: Character,
        psi: np.ndarray
    ) -> complex:
        """
        Compute Bessel function for GL_n at anti-diag(c I_k, I_{n-k}).
        """
        G = self.G
        # Parabolic P_k is minuscule for GL_n
        i = k - 1  # index of simple root
        crystal = GeometricCrystal(G, i)
        
        # The special element in Z(L)
        z = np.array([c, 1.0])
        
        # Compute Bessel function
        return bessel_function_sum_formula(crystal, chi, psi, z, self.q)
    
    def compute_kloosterman_gl_n(
        self,
        a: float,
        k: int,
        chi: Character
    ) -> complex:
        """
        Compute Kloosterman sheaf trace for GL_n at a ∈ F×.
        """
        G = self.G
        omega = G.fundamental_coweight(k - 1)
        kl_sheaf = KloostermanSheaf(G, omega, chi)
        return kl_sheaf.trace(a, self.q)

def run_example_gl2():
    """Example for GL_2."""
    print("=" * 60)
    print("Example: GL_2 (rank 1)")
    print("=" * 60)
    
    # Setup
    q = 2
    n = 2
    G = ReductiveGroup(1)  # GL_2 has rank 1
    example = ExampleGLn(n, q)
    
    # Define a character
    chi_values = {
        (1, 1): 1.0 + 0.0j,
        (1, -1): -1.0 + 0.0j,
        (-1, 1): -1.0 + 0.0j,
        (-1, -1): 1.0 + 0.0j
    }
    chi = Character(chi_values)
    
    # Define additive character
    psi = np.array([1.0, -1.0])  # Placeholder
    
    # Compute Bessel function
    c = 1.0
    k = 1
    bessel_val = example.compute_bessel_gl_n(c, k, chi, psi)
    print(f"Bessel function at anti-diag({c}I_1, I_1): {bessel_val}")
    
    # Compute Kloosterman sheaf
    kl_val = example.compute_kloosterman_gl_n(c, k, chi)
    print(f"Kloosterman sheaf at {c}: {kl_val}")
    
    # Verify theorem
    theorem = MainTheorem(G, 0, chi)  # i = 0 for GL_2
    z = np.array([c, 1.0])
    verified = theorem.verify_for_z(z, q, psi)
    print(f"Theorem verified: {verified}")
    
    return bessel_val, kl_val

def run_example_gl3():
    """Example for GL_3."""
    print("\n" + "=" * 60)
    print("Example: GL_3 (rank 2)")
    print("=" * 60)
    
    # Setup
    q = 2
    n = 3
    G = ReductiveGroup(2)  # GL_3 has rank 2
    example = ExampleGLn(n, q)
    
    # Define a character (simplified)
    chi_values = {
        (1, 1, 1): 1.0 + 0.0j,
        (1, 1, -1): -1.0 + 0.0j
    }
    chi = Character(chi_values)
    psi = np.array([1.0, -1.0, 1.0])  # Placeholder
    
    # Minuscule parabolics for GL_3: P_1 and P_2
    for k in [1, 2]:
        print(f"\nParabolic P_{k}:")
        c = 1.0
        bessel_val = example.compute_bessel_gl_n(c, k, chi, psi)
        print(f"  Bessel at anti-diag({c}I_{k}, I_{3-k}): {bessel_val}")
        
        theorem = MainTheorem(G, k-1, chi)
        z = np.array([c, 1.0])
        verified = theorem.verify_for_z(z, q, psi)
        print(f"  Theorem verified: {verified}")

def run_example_type_a_quasisplit():
    """
    Example for quasi-split type A_n.
    For n odd, there is a rational minuscule cocharacter.
    """
    print("\n" + "=" * 60)
    print("Example: Quasi-split Type A_3 (SO_4)")
    print("=" * 60)
    
    # For SO_4, this is essentially the same as GL_4 with determinant 1
    # The minuscule cocharacters come from outer automorphisms
    n = 4
    q = 2
    G = ReductiveGroup(n - 1)
    
    # In quasi-split case, we use the non-split form
    # The rational minuscule cocharacter exists for n odd
    print(f"Quasi-split A_{n-1} has rational minuscule cocharacter.")
    print("This corresponds to the outer automorphism of the Dynkin diagram.")
    
    # For SO_4, the minuscule parabolics give exceptional isomorphisms
    print("For SO_4: P_1 corresponds to spinors, P_2 to dual spinors.")

def run_klosterman_sheaf_computation():
    """Compute explicit Kloosterman sheaf traces."""
    print("\n" + "=" * 60)
    print("Kloosterman Sheaf Computations")
    print("=" * 60)
    
    q = 3  # Small prime for explicit computation
    n = 2  # GL_2
    
    G = ReductiveGroup(n - 1)
    chi = Character({(1, 1): 1.0})
    
    print(f"\nComputing Kloosterman sums for GL_{n} over F_{q}:")
    
    for a in range(1, q):
        # Fundamental representation (ω_1)
        omega = G.fundamental_coweight(0)
        kl_sheaf = KloostermanSheaf(G, omega, chi)
        kl_val = kl_sheaf.trace(a, q)
        print(f"  Kl(ψ, {a}) = {kl_val}")
        
        # Also compute classical Kloosterman sum directly for verification
        total = 0.0 + 0.0j
        for x in range(1, q):
            for y in range(1, q):
                if (x * y) % q == a % q:
                    total += math.e ** (2j * math.pi * (x + y) / q)
        print(f"  Classical: Kl(ψ, {a}) = {total}")

if __name__ == "__main__":
    # Run examples
    run_example_gl2()
    run_example_gl3()
    run_example_type_a_quasisplit()
    run_klosterman_sheaf_computation()
    
    print("\n" + "=" * 60)
    print("Summary of Key Mathematical Structures:")
    print("=" * 60)
    print("""
    1. Root System: Type A_n with simple roots α_i = e_i - e_{i+1}
    
    2. Weyl Group: S_{n+1} acting by permutations
    
    3. Parabolic Subgroups: P_i with Levi factor GL_i × GL_{n-i}
    
    4. Minuscule Coweights: ω_i^∨ for all i in type A
    
    5. Geometric Crystal: X = U Z(L) w_P U ∩ B_- 
       with maps π: X → Z(L), γ: X → T, f: X → A^1
    
    6. Bessel Functions: J_{τ,ψ}(g) = ⟨τ(g)v, v⟩ / ⟨v, v⟩
       with v the ψ-Whittaker vector
    
    7. Kloosterman Sheaves: Kl_{G^}^{μ}(φ, χ) 
       from Heinloth-Ngô-Yun
    
    8. Main Theorem: 
       Kl_{G^}^{ω_i^∨}(φ, χ^{-1}; α_i(z)) = 
       χ(z_0) (-1)^{dim G/P} q^{dim G/P/2} J_{τ,ψ}(z w_P)
    
    9. GL_n Example:
       anti-diag(c I_k, I_{n-k}) ∈ Z(L)
    """)