import numpy as np
from itertools import product

class FiniteGroupSL2:
    """Finite group SL(2, q) with basic operations"""
    def __init__(self, q):
        """
        Initialize SL(2, q) over finite field F_q
        
        Parameters:
        -----------
        q : int
            Prime power (must be odd for simplicity)
        """
        self.q = q
        self.p = self._prime_factor(q)
        self.elements = self._generate_elements()
        self.size = len(self.elements)
        
    def _prime_factor(self, q):
        """Find prime factor of q"""
        for p in range(2, int(np.sqrt(q)) + 1):
            if q % p == 0:
                return p
        return q
    
    def _generate_elements(self):
        """Generate all elements of SL(2, q)"""
        elements = []
        for a in range(self.q):
            for b in range(self.q):
                for c in range(self.q):
                    # d determined by ad - bc = 1
                    for d in range(self.q):
                        if (a*d - b*c) % self.q == 1:
                            elements.append(np.array([[a, b], [c, d]]))
        return elements
    
    def multiply(self, g1, g2):
        """Multiply two matrices mod q"""
        return np.mod(np.dot(g1, g2), self.q)
    
    def inverse(self, g):
        """Inverse of matrix in SL(2, q)"""
        # For SL(2), inverse is [[d, -b], [-c, a]]
        a, b = g[0, 0], g[0, 1]
        c, d = g[1, 0], g[1, 1]
        return np.array([[d, -b], [-c, a]]) % self.q

class BesselFunction:
    """Bessel function for principal series of SL(2, q)"""
    def __init__(self, q, chi, psi):
        """
        Initialize Bessel function
        
        Parameters:
        -----------
        q : int
            Field size
        chi : callable
            Character of F_q^*
        psi : callable
            Additive character of F_q
        """
        self.q = q
        self.G = FiniteGroupSL2(q)
        self.chi = chi
        self.psi = psi
        self.dim = q - 1  # dimension of principal series
        
        # Define Borel subgroups
        self.T = self._torus_elements()
        self.U = self._unipotent_elements()
        self.B = self._borel_elements()
        
    def _torus_elements(self):
        """Diagonal torus elements"""
        return [np.array([[t, 0], [0, pow(t, -1, self.q)]]) 
                for t in range(1, self.q) if np.gcd(t, self.q) == 1]
    
    def _unipotent_elements(self):
        """Upper unipotent elements"""
        return [np.array([[1, x], [0, 1]]) for x in range(self.q)]
    
    def _borel_elements(self):
        """Borel subgroup elements (upper triangular)"""
        elements = []
        for t in range(1, self.q):
            if np.gcd(t, self.q) == 1:
                for x in range(self.q):
                    elements.append(np.array([[t, x], [0, pow(t, -1, self.q)]]))
        return elements
    
    def whittaker_vector(self, g):
        """Compute Whittaker vector at group element g"""
        # For principal series, Whittaker vector is supported on Borel
        # and given by character of torus times additive character
        if not self._is_in_borel(g):
            return 0
        
        # Extract diagonal and off-diagonal parts
        t = g[0, 0] % self.q
        x = g[0, 1] % self.q
        
        return self.chi(t) * self.psi(x)
    
    def _is_in_borel(self, g):
        """Check if element is in Borel subgroup"""
        return g[1, 0] % self.q == 0
    
    def compute_bessel_at(self, g):
        """
        Compute Bessel function at group element g
        
        J(g) = sum_{u in U} psi^{-1}(u) * v(ug) / ||v||^2
        """
        # Compute norm squared of Whittaker vector
        norm_sq = 0
        for u in self.U:
            val = self.whittaker_vector(u)
            norm_sq += val * np.conjugate(val)
        
        if norm_sq == 0:
            return 0
        
        # Compute inner product <T(g)v, v>
        inner_prod = 0
        for u in self.U:
            ug = self.G.multiply(u, g)
            v_ug = self.whittaker_vector(ug)
            v_u = self.whittaker_vector(u)
            inner_prod += v_ug * np.conjugate(v_u) * np.conjugate(self.psi(u[0, 1]))
        
        return inner_prod / norm_sq
    
    def compute_bessel_principal_series(self, z, w):
        """Compute Bessel function at zw for special elements"""
        # w is the Weyl group representative (for SL2, w = [[0,1],[-1,0]])
        w_matrix = np.array([[0, 1], [-1, 0]]) % self.q
        g = self.G.multiply(z, w_matrix)
        return self.compute_bessel_at(g)

class KloostermanSum:
    """Kloosterman sum for SL(2, q)"""
    def __init__(self, q, chi, psi):
        self.q = q
        self.chi = chi
        self.psi = psi
    
    def compute_kloosterman(self, a):
        """
        Compute Kloosterman sum Kl(a) = sum_{x in F_q^*} chi(x) psi(x + a/x)
        
        Parameters:
        -----------
        a : int
            Parameter in F_q^*
        
        Returns:
        --------
        complex
            Kloosterman sum value
        """
        if a % self.q == 0:
            return 0
        
        kl_sum = 0
        for x in range(1, self.q):
            if np.gcd(x, self.q) == 1:
                x_inv = pow(x, -1, self.q)
                kl_sum += self.chi(x) * self.psi((x + a * x_inv) % self.q)
        
        return kl_sum / np.sqrt(self.q)  # Normalization

def verify_theorem(q, a):
    """
    Verify Theorem 1.1 numerically for SL(2, q)
    
    The theorem states:
    Kl(chi^{-1}, a) = (-1)^d q^{d/2} J(z * w_P)
    
    where d = dim(G/P) = 1 for SL2
    
    Parameters:
    -----------
    q : int
        Field size
    a : int
        Parameter
    """
    # Define characters (for simplicity, use trivial characters)
    def chi_trivial(x):
        return 1.0 + 0.0j
    
    def psi_trivial(x):
        # A simple additive character for finite fields
        return np.exp(2j * np.pi * x / q)
    
    # Compute Bessel function
    bessel = BesselFunction(q, chi_trivial, psi_trivial)
    
    # For SL2, z is diagonal with z = [[a, 0], [0, a^{-1}]]
    a_inv = pow(a, -1, q)
    z = np.array([[a, 0], [0, a_inv]]) % q
    
    j_value = bessel.compute_bessel_principal_series(z, None)
    
    # Compute Kloosterman sum
    kl_sum = KloostermanSum(q, chi_trivial, psi_trivial)
    kl_value = kl_sum.compute_kloosterman(a)
    
    # Expected from theorem: Kl = (-1)^d q^{d/2} * J
    d = 1  # dimension of SL2/P
    expected = (-1)**d * np.sqrt(q) * j_value
    
    print(f"q = {q}, a = {a}")
    print(f"Bessel function J = {j_value}")
    print(f"Kloosterman sum Kl = {kl_value}")
    print(f"Expected Kl = {expected}")
    print(f"Difference = {abs(kl_value - expected)}")
    print("-" * 40)
    
    return kl_value, expected

# Example verification
if __name__ == "__main__":
    print("Verifying Theorem 1.1 for SL(2, q)")
    print("=" * 50)
    
    # Test for q = 3, 5, 7
    for q in [3, 5, 7]:
        # Choose a random a in F_q^*
        for a in range(1, q):
            if np.gcd(a, q) == 1:
                verify_theorem(q, a)