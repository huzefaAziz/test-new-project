import numpy as np
from numpy import array, zeros, ones, zeros_like, arange
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

class HyperellipticBrillNoetherCA:
    """
    Cellular Automata implementation of Hyperelliptic Brill-Noether loci.
    Simulates the evolution of catalecticant matrices and Brill-Noether loci
    as described in the paper arXiv:2608.21301.
    """
    
    def __init__(self, 
                 genus: int = 3,
                 dimension: int = 64,
                 time_steps: int = 100,
                 seed: Optional[int] = None):
        """
        Initialize the cellular automaton for Brill-Noether loci simulation.
        
        Args:
            genus: Genus g of the hyperelliptic curve (g ≥ 2)
            dimension: Size of the cellular automaton grid
            time_steps: Number of evolution steps
            seed: Random seed for reproducibility
        """
        self.g = genus
        self.dim = dimension
        self.T = time_steps
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize the cellular automaton state
        # Each cell represents a point in Pic^d(C)
        self.state = np.random.rand(dimension, dimension)
        
        # Brill-Noether loci parameters
        self.d = 2  # degree d (0 ≤ d ≤ g-1)
        self.r = 0  # rank parameter
        
        # Catalecticant matrix dimensions
        # From the paper: ℓ = h^0(C,L), ℓ' = h^1(C,L)
        self.ell = 2  # typical value for hyperelliptic curves
        self.ell_prime = self.g - self.d + self.ell - 1
        
        # Precompute catalecticant patterns
        self.catalecticant_matrix = None
        self.brill_noether_loci = []
        
        # Initialize grid of local models
        self.local_models = self._initialize_local_models()
        
        # Log canonical threshold (LCT) tracking
        self.lct_values = []
        
    def _initialize_local_models(self) -> np.ndarray:
        """
        Initialize local models for each cell based on the paper's Theorem 1.1.
        
        Returns:
            Local model grid representing (A^{ℓ+ℓ'-1} × A^b, 0)
        """
        # From Theorem 1.1: b = d - 2ℓ + 2
        b = self.d - 2*self.ell + 2
        
        # Local dimension: ℓ + ℓ' - 1 + b = g
        local_dim = self.ell + self.ell_prime - 1 + b
        
        # Initialize grid of local coordinates
        models = np.zeros((self.dim, self.dim, local_dim))
        
        # Fill with random but structured values
        for i in range(self.dim):
            for j in range(self.dim):
                # Apply Catalan identity constraints
                # z_{i+j} = z_{i+j} (catalecticant pattern)
                val = np.random.randn(local_dim)
                # Enforce catalecticant symmetries
                for k in range(local_dim - 1):
                    val[k+1] = val[k] * 0.9 + 0.1 * np.random.randn()
                models[i, j] = val
                
        return models
    
    def compute_catalecticant_matrix(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the rectangular catalecticant matrix as in equation (3).
        
        Args:
            z: Vector of coefficients [z_0, ..., z_{ℓ+ℓ'-2}]
            
        Returns:
            ℓ' × ℓ rectangular catalecticant matrix
        """
        L = self.ell
        Lp = self.ell_prime
        total = L + Lp - 1
        
        # Ensure z has correct length
        if len(z) < total:
            z = np.pad(z, (0, total - len(z)))
        elif len(z) > total:
            z = z[:total]
        
        # Build catalecticant matrix
        cat_matrix = np.zeros((Lp, L))
        for i in range(Lp):
            for j in range(L):
                cat_matrix[i, j] = z[i + j]
                
        return cat_matrix
    
    def compute_brill_noether_locus(self, 
                                   z: np.ndarray, 
                                   r: Optional[int] = None) -> np.ndarray:
        """
        Compute the Brill-Noether locus W_d^r(C) locally.
        
        Args:
            z: Local coordinates
            r: Rank parameter (if None, use self.r)
            
        Returns:
            Indicator matrix of the Brill-Noether locus
        """
        if r is None:
            r = self.r
        
        L = self.ell
        Lp = self.ell_prime
        
        # Compute catalecticant matrix
        cat_mat = self.compute_catalecticant_matrix(z)
        
        # Compute minors of size L - r
        min_rank = L - r
        
        # For simplicity, compute using determinant of submatrices
        locus_indicator = np.zeros((Lp - min_rank + 1, 
                                    L - min_rank + 1))
        
        for i in range(Lp - min_rank + 1):
            for j in range(L - min_rank + 1):
                submat = cat_mat[i:i+min_rank, j:j+min_rank]
                if submat.shape == (min_rank, min_rank):
                    locus_indicator[i, j] = abs(np.linalg.det(submat)) < 1e-6
                    
        return locus_indicator
    
    def petri_map(self, local_state: np.ndarray) -> np.ndarray:
        """
        Compute the Petri map as in Proposition 3.2.
        
        The Petri map: H^0(C,L) ⊗ H^0(C,K_C⊗L^{-1}) → H^0(C,K_C)
        """
        L = self.ell
        Lp = self.ell_prime
        
        # Simulate the Petri map using the catalecticant structure
        # From equation (23): Sym^{ℓ-1}V ⊗ Sym^{ℓ'-1}V → Sym^{ℓ+ℓ'-2}V
        
        # Sample binary forms
        v1 = np.random.randn(L)
        v2 = np.random.randn(Lp)
        
        # Multiplication in symmetric algebra
        result = np.zeros(L + Lp - 1)
        for i in range(L):
            for j in range(Lp):
                result[i + j] += v1[i] * v2[j]
                
        return result
    
    def update_rule(self, 
                   neighborhood: np.ndarray, 
                   center: np.ndarray) -> np.ndarray:
        """
        Cellular automaton update rule based on Brill-Noether dynamics.
        
        The update rule implements the local evolution of the catalecticant
        matrix using the extension class from Lemma 5.1.
        """
        # Compute extension class ξ: S → W
        # From Lemma 5.1: dim W = ℓ + ℓ' - 1
        W_dim = self.ell + self.ell_prime - 1
        
        # Compute local extension class
        xi = np.zeros(W_dim)
        for i in range(W_dim):
            # Use neighboring values to compute extension
            neighbor_avg = np.mean(neighborhood[:, i]) if neighborhood.shape[0] > 0 else 0
            xi[i] = center[i] + 0.1 * (neighbor_avg - center[i])
        
        # Enforce catalecticant identities (Proposition 5.2)
        # D_{i+1,j} = D_{i,j+1}
        new_state = center.copy()
        for i in range(self.ell_prime - 1):
            for j in range(self.ell - 1):
                idx = i + j
                if idx + 1 < len(new_state):
                    new_state[idx + 1] = new_state[idx] * 0.95 + 0.05 * xi[idx]
                    
        return new_state
    
    def step(self) -> None:
        """
        Perform one time step of the cellular automaton evolution.
        """
        new_state = np.zeros_like(self.state)
        local_models_new = np.zeros_like(self.local_models)
        
        # Get neighborhood indices (3x3 Moore neighborhood)
        for i in range(self.dim):
            for j in range(self.dim):
                # Extract neighborhood
                i_min, i_max = max(0, i-1), min(self.dim, i+2)
                j_min, j_max = max(0, j-1), min(self.dim, j+2)
                
                neighborhood_vals = self.local_models[i_min:i_max, j_min:j_max]
                center_vals = self.local_models[i, j]
                
                # Flatten for processing
                neigh_flat = neighborhood_vals.reshape(-1, neighborhood_vals.shape[-1])
                center_flat = center_vals
                
                # Apply update rule
                new_center = self.update_rule(neigh_flat, center_flat)
                local_models_new[i, j] = new_center
                
                # Update state value (use first component as indicator)
                new_state[i, j] = new_center[0]
                
        self.local_models = local_models_new
        self.state = new_state
        
        # Track Brill-Noether loci evolution
        if len(self.brill_noether_loci) < self.T:
            locus = self.compute_brill_noether_locus(
                self.local_models[self.dim//2, self.dim//2]
            )
            self.brill_noether_loci.append(locus)
            
            # Compute log canonical threshold (Proposition 1.3)
            lct = self.compute_log_canonical_threshold()
            self.lct_values.append(lct)
    
    def compute_log_canonical_threshold(self) -> float:
        """
        Compute the log canonical threshold as in Proposition 1.3.
        
        Returns:
            lct_L(Pic^d(C), W_d^r(C))
        """
        L = self.ell
        Lp = self.ell_prime
        r = self.r
        
        # From Proposition 1.3:
        # If d = g-1 and r = 0: lct = 1
        # Otherwise: lct = 1 + (ℓ' + r - 1) / (ℓ - r)
        
        if self.d == self.g - 1 and r == 0:
            return 1.0
        else:
            return 1.0 + (Lp + r - 1) / (L - r)
    
    def compute_betti_numbers(self) -> np.ndarray:
        """
        Compute Betti numbers of the Brill-Noether locus using equation (10).
        
        Returns:
            Array of Betti numbers b_k(X)
        """
        m = self.d - 2*self.r
        g = self.g
        
        # From equation (10):
        # b_k(X) = C(2g, k) for 0 ≤ k ≤ m
        # b_k(X) = C(2g, 2m-k) for m ≤ k ≤ 2m
        
        max_dim = 2*m + 1
        betti = np.zeros(max_dim, dtype=int)
        
        for k in range(max_dim):
            if k <= m:
                betti[k] = self._binomial(2*g, k)
            elif k <= 2*m:
                betti[k] = self._binomial(2*g, 2*m - k)
                
        return betti
    
    def _binomial(self, n: int, k: int) -> int:
        """Compute binomial coefficient with bounds checking."""
        if k < 0 or k > n:
            return 0
        from math import comb
        return comb(n, k)
    
    def visualize_locus(self, 
                       step: Optional[int] = None, 
                       save: bool = False) -> None:
        """
        Visualize the Brill-Noether locus and catalecticant structure.
        """
        if step is None:
            step = len(self.brill_noether_loci) - 1
            
        if step < 0 or step >= len(self.brill_noether_loci):
            print(f"No data for step {step}")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # 1. Cellular automaton state
        ax1 = axes[0, 0]
        im1 = ax1.imshow(self.state, cmap='viridis', aspect='auto')
        ax1.set_title(f'CA State (Step {step})')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Brill-Noether locus
        ax2 = axes[0, 1]
        locus = self.brill_noether_loci[step]
        im2 = ax2.imshow(locus, cmap='Reds', aspect='auto')
        ax2.set_title(f'Brill-Noether Locus W_{self.d}^{self.r}(C)')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Log canonical threshold evolution
        ax3 = axes[1, 0]
        ax3.plot(self.lct_values)
        ax3.set_title('Log Canonical Threshold Evolution')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('LCT')
        ax3.grid(True)
        
        # 4. Betti numbers
        ax4 = axes[1, 1]
        betti = self.compute_betti_numbers()
        k_vals = np.arange(len(betti))
        ax4.bar(k_vals, betti, alpha=0.7, color='blue')
        ax4.set_title(f'Betti Numbers (m={self.d-2*self.r})')
        ax4.set_xlabel('k')
        ax4.set_ylabel('b_k(X)')
        ax4.grid(True)
        
        plt.tight_layout()
        if save:
            plt.savefig('brill_noether_ca.png', dpi=150)
        plt.show()

class CatalecticantSecantVariety:
    """
    Implementation of secant varieties of rational normal curves
    as used in Theorem 1.4 and Proposition 1.3.
    """
    
    def __init__(self, degree: int, secant_order: int):
        """
        Initialize secant variety of rational normal curve.
        
        Args:
            degree: Degree of the rational normal curve
            secant_order: s-th secant variety
        """
        self.N = degree
        self.s = secant_order
        
    def compute_dimension(self) -> int:
        """
        Compute dimension of the s-secant variety.
        
        Returns:
            Dimension of Σ_s^{(N)}
        """
        # For rational normal curve of degree N
        # dim(Σ_s) = min(2s + 1, N)
        return min(2*self.s + 1, self.N)
    
    def is_rational_homology_manifold(self) -> bool:
        """
        Check if the secant variety is a rational homology manifold.
        Used in Theorem 1.4(i).
        """
        dim = self.compute_dimension()
        # From [8, Corollary L]: secant varieties of rational normal curves
        # are rational homology manifolds when they are proper
        return dim < self.N

def simulate_hyperelliptic_ca(
    genus: int = 4,
    dim: int = 64,
    time_steps: int = 50,
    seed: int = 42
) -> HyperellipticBrillNoetherCA:
    """
    Run a complete simulation of the hyperelliptic Brill-Noether CA.
    
    Args:
        genus: Genus of the hyperelliptic curve
        dim: Grid dimension
        time_steps: Number of simulation steps
        seed: Random seed
        
    Returns:
        Simulated CA instance
    """
    ca = HyperellipticBrillNoetherCA(
        genus=genus,
        dimension=dim,
        time_steps=time_steps,
        seed=seed
    )
    
    print(f"Initialized Hyperelliptic CA:")
    print(f"  Genus: {ca.g}")
    print(f"  Degree d: {ca.d}")
    print(f"  Rank r: {ca.r}")
    print(f"  ℓ = h^0(C,L): {ca.ell}")
    print(f"  ℓ' = h^1(C,L): {ca.ell_prime}")
    print(f"  Local dimension: {ca.local_models.shape[-1]}")
    print(f"  Grid size: {ca.dim}×{ca.dim}")
    
    # Evolution
    print(f"\nEvolving for {time_steps} time steps...")
    for t in range(time_steps):
        ca.step()
        if (t + 1) % 10 == 0:
            print(f"  Step {t+1}/{time_steps}")
    
    # Compute final Betti numbers
    betti = ca.compute_betti_numbers()
    print(f"\nBetti numbers b_k(X):")
    for k, bk in enumerate(betti):
        if bk > 0:
            print(f"  b_{k} = {bk}")
    
    # Visualize
    ca.visualize_locus()
    
    return ca

# Example usage
if __name__ == "__main__":
    # Run simulation
    ca = simulate_hyperelliptic_ca(
        genus=4,
        dim=64,
        time_steps=50,
        seed=42
    )
    
    # Additional analysis of secant varieties
    print("\n--- Secant Variety Analysis ---")
    for s in range(3):
        secant = CatalecticantSecantVariety(degree=6, secant_order=s)
        dim = secant.compute_dimension()
        is_rh = secant.is_rational_homology_manifold()
        print(f"  Σ_{s}^{(6)}: dimension={dim}, rational homology manifold={is_rh}")