import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import comb
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

class HyperellipticBrillNoether:
    """
    Implementation of Hyperelliptic Brill-Noether loci computations
    with visualization capabilities.
    """
    
    def __init__(self, g: int, d: int, r: int = 0):
        """
        Initialize the Brill-Noether locus for a hyperelliptic curve of genus g.
        
        Parameters:
        -----------
        g : int
            Genus of the hyperelliptic curve (g >= 2)
        d : int
            Degree of the line bundle (0 <= d <= g-1)
        r : int
            Rank parameter (0 <= r <= g-1)
        """
        if g < 2:
            raise ValueError("Genus must be at least 2")
        if not (0 <= d <= g-1):
            raise ValueError("d must be in [0, g-1]")
        if not (0 <= r <= g-1):
            raise ValueError("r must be in [0, g-1]")
            
        self.g = g
        self.d = d
        self.r = r
        
        # Compute parameters from equation (1) in the paper
        self.ell = self._compute_ell()
        self.ell_prime = self._compute_ell_prime()
        self.b = self._compute_b()
        
        # Dimension of the Brill-Noether locus
        self.m = d - 2*r
        self.c = g - self.m
        
        # Validate parameters
        if self.b < 0:
            raise ValueError(f"b = {self.b} must be non-negative")
        if self.ell_prime < self.ell:
            raise ValueError(f"ell' = {self.ell_prime} must be >= ell = {self.ell}")
            
        # Store computed properties
        self._betti_numbers = None
        self._hodge_numbers = None
        self._multiplicity = None
        self._lct = None
        
        # Store the required length for z
        self.z_length = self.ell + self.ell_prime - 1
        
    def _compute_ell(self) -> int:
        """Compute h^0(C, L) for a generic L in W_d^0(C)."""
        return self.d // 2 + 1
    
    def _compute_ell_prime(self) -> int:
        """Compute h^1(C, L) = g - d + ell - 1."""
        return self.g - self.d + self.ell - 1
    
    def _compute_b(self) -> int:
        """Compute b = d - 2*ell + 2."""
        return self.d - 2*self.ell + 2
    
    def catalecticant_matrix(self, z: np.ndarray) -> np.ndarray:
        """
        Construct the catalecticant matrix Cat_{ell', ell}(z) as in equation (3).
        """
        if len(z) != self.z_length:
            raise ValueError(f"z must have length {self.z_length} (ell + ell' - 1), got {len(z)}")
            
        matrix = np.zeros((self.ell_prime, self.ell))
        for i in range(self.ell_prime):
            for j in range(self.ell):
                matrix[i, j] = z[i + j]
        return matrix
    
    def catalecticant_minors(self, z: np.ndarray, s: Optional[int] = None) -> np.ndarray:
        """Compute the s x s minors of the catalecticant matrix."""
        if s is None:
            s = self.ell - self.r
            
        if s < 1 or s > min(self.ell_prime, self.ell):
            raise ValueError(f"s must be between 1 and {min(self.ell_prime, self.ell)}")
            
        matrix = self.catalecticant_matrix(z)
        
        minors = []
        rows = list(range(self.ell_prime))
        cols = list(range(self.ell))
        
        from itertools import combinations
        for row_subset in combinations(rows, s):
            for col_subset in combinations(cols, s):
                submatrix = matrix[np.ix_(row_subset, col_subset)]
                minors.append(np.linalg.det(submatrix))
                
        return np.array(minors)
    
    def get_random_z(self) -> np.ndarray:
        """Generate a random z vector of the correct length."""
        return np.random.randn(self.z_length)
    
    def get_example_z(self) -> np.ndarray:
        """Generate an example z vector with simple values."""
        return np.arange(1, self.z_length + 1, dtype=float)
    
    def multiplicity(self) -> int:
        """Compute the multiplicity of W_d^r(C) at L."""
        if self._multiplicity is None:
            try:
                self._multiplicity = int(comb(self.ell_prime + self.r, 
                                             self.ell - self.r - 1))
            except ValueError:
                self._multiplicity = 0
        return self._multiplicity
    
    def log_canonical_threshold(self) -> float:
        """Compute the log canonical threshold."""
        if self._lct is None:
            if self.d == self.g - 1 and self.r == 0:
                self._lct = 1.0
            else:
                self._lct = 1.0 + (self.ell_prime + self.r - 1) / (self.ell - self.r)
        return self._lct
    
    def betti_numbers(self) -> List[int]:
        """Compute the Betti numbers of X = W_d^r(C)."""
        if self._betti_numbers is None:
            m = self.m
            if m < 0:
                return []
            betti = []
            for k in range(2*m + 1):
                if k <= m:
                    betti.append(int(comb(2*self.g, k)))
                else:
                    betti.append(int(comb(2*self.g, 2*m - k)))
            self._betti_numbers = betti
        return self._betti_numbers
    
    def hodge_numbers(self) -> np.ndarray:
        """Compute the Hodge numbers h^{p,q}(X)."""
        if self._hodge_numbers is None:
            m = self.m
            if m < 0:
                return np.zeros((self.g + 1, self.g + 1))
            hodge = np.zeros((self.g + 1, self.g + 1))
            
            for p in range(self.g + 1):
                for q in range(self.g + 1):
                    if p + q <= m:
                        hodge[p, q] = comb(self.g, p) * comb(self.g, q)
                    elif p + q >= m:
                        hodge[p, q] = comb(self.g, m - p) * comb(self.g, m - q)
                        
            self._hodge_numbers = hodge
        return self._hodge_numbers

    # ============ VISUALIZATION METHODS ============
    
    def plot_catalecticant_matrix(self, z: Optional[np.ndarray] = None, figsize: Tuple[int, int] = (8, 6)):
        """
        Visualize the catalecticant matrix as a heatmap.
        """
        if z is None:
            z = self.get_example_z()
            
        matrix = self.catalecticant_matrix(z)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Heatmap
        im = ax1.imshow(matrix, cmap='viridis', aspect='auto')
        ax1.set_title(f'Cat$_{{{self.ell_prime},{self.ell}}}(z)$', fontsize=12)
        ax1.set_xlabel('j', fontsize=10)
        ax1.set_ylabel('i', fontsize=10)
        
        # Add text annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax1.text(j, i, f'{matrix[i, j]:.1f}',
                        ha='center', va='center', 
                        color='white' if np.abs(matrix[i, j]) > 0.5 else 'black',
                        fontsize=8)
        
        plt.colorbar(im, ax=ax1)
        
        # Show structure (Toeplitz pattern)
        ax2.axis('off')
        ax2.set_title('Catalecticant Structure', fontsize=12)
        
        # Draw the matrix structure with indices
        table_data = [[''] + [f'j={j}' for j in range(self.ell)]]
        for i in range(self.ell_prime):
            row = [f'i={i}']
            for j in range(self.ell):
                row.append(f'z_{i+j}')
            table_data.append(row)
        
        table = ax2.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_betti_numbers(self, figsize: Tuple[int, int] = (10, 6)):
        """Plot the Betti numbers of X = W_d^r(C)."""
        betti = self.betti_numbers()
        if not betti:
            print("No Betti numbers to plot (m < 0)")
            return None
            
        k_values = list(range(len(betti)))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        colors = ['#2E86AB' if k <= self.m else '#A23B72' for k in k_values]
        ax1.bar(k_values, betti, color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('k', fontsize=12)
        ax1.set_ylabel(f'b$_k$(X)', fontsize=12)
        ax1.set_title(f'Betti Numbers of $W_{{{self.d}}}^{{{self.r}}}(C)$, genus {self.g}', fontsize=12)
        ax1.axvline(x=self.m, color='red', linestyle='--', alpha=0.5, 
                   label=f'm = {self.m}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(betti):
            ax1.text(i, v + max(betti)*0.02, str(v), ha='center', fontsize=8)
        
        # Symmetry check
        symmetry = [betti[k] - betti[2*self.m - k] for k in range(self.m + 1)]
        ax2.plot(range(self.m + 1), symmetry, 'o-', color='#F18F01', linewidth=2, markersize=8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('k', fontsize=12)
        ax2.set_ylabel(f'b$_k$ - b$_{{{2*self.m}-k}}$', fontsize=12)
        ax2.set_title('Poincaré Duality Symmetry Check', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_hodge_diamond(self, figsize: Tuple[int, int] = (10, 8)):
        """Visualize the Hodge diamond of X."""
        hodge = self.hodge_numbers()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        # Create triangular layout for Hodge diamond
        max_show = min(self.g, self.m)
        if max_show < 0:
            print("No Hodge numbers to display (m < 0)")
            return None
        
        # Build the diamond structure
        diamond = []
        for p in range(max_show + 1):
            row = []
            for q in range(max_show + 1):
                if p + q <= max_show:
                    row.append(f'{int(hodge[p, q])}')
                else:
                    row.append('')
            diamond.append(row)
        
        # Create table with triangular format
        # Each row needs to have the same length
        max_cols = max_show + 1
        table_data = []
        for i, row in enumerate(diamond):
            # Pad the row to have max_cols entries
            padded_row = row + [''] * (max_cols - len(row))
            table_data.append(padded_row)
        
        # Add headers - make sure headers match the number of columns
        headers = [f'p={i}' for i in range(max_cols)]
        table_data_with_headers = [[''] + headers]  # First row: headers
        for i, row in enumerate(table_data):
            table_data_with_headers.append([f'q={i}'] + row)
        
        # Now create the table
        table = ax.table(cellText=table_data_with_headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Color cells based on values
        max_val = np.max(hodge)
        if max_val > 0:
            for i in range(1, len(table_data_with_headers)):
                for j in range(1, len(table_data_with_headers[i])):
                    cell = table[(i, j)]
                    cell_text = table_data_with_headers[i][j]
                    if cell_text and cell_text.isdigit():
                        val = int(cell_text)
                        if val > 0:
                            cell.set_facecolor(plt.cm.Blues(val / max_val))
        
        ax.set_title(f'Hodge Numbers of $W_{{{self.d}}}^{{{self.r}}}(C)$, genus {self.g}\n'
                    f'm = {self.m}', fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_multiplicity_vs_genus(self, max_g: Optional[int] = None, figsize: Tuple[int, int] = (12, 6)):
        """Plot how multiplicity varies with genus for fixed d and r."""
        if max_g is None:
            max_g = max(self.g + 3, 10)
            
        multiplicities = []
        genera = list(range(max(2, self.r + 1), max_g + 1))
        
        for g in genera:
            if g >= self.d + 1:
                try:
                    bn = HyperellipticBrillNoether(g, self.d, self.r)
                    multiplicities.append(bn.multiplicity())
                except:
                    multiplicities.append(0)
            else:
                multiplicities.append(0)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(genera, multiplicities, 'o-', color='#2E86AB', linewidth=2, markersize=8)
        ax.set_xlabel('Genus g', fontsize=12)
        ax.set_ylabel(f'multiplicity of $W_{{{self.d}}}^{{{self.r}}}(C)$', fontsize=12)
        ax.set_title(f'Multiplicity vs Genus for d={self.d}, r={self.r}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (g, m) in enumerate(zip(genera, multiplicities)):
            if m > 0:
                ax.annotate(str(m), (g, m), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_lct_heatmap(self, d_range: Optional[Tuple[int, int]] = None, 
                         r_range: Optional[Tuple[int, int]] = None,
                         figsize: Tuple[int, int] = (10, 8)):
        """Create a heatmap of log canonical thresholds for varying d and r."""
        if d_range is None:
            d_range = (0, self.g - 1)
        if r_range is None:
            r_range = (0, max(1, self.g // 2))
        
        d_vals = list(range(d_range[0], d_range[1] + 1))
        r_vals = list(range(r_range[0], r_range[1] + 1))
        
        lct_matrix = np.zeros((len(r_vals), len(d_vals)))
        valid_matrix = np.zeros_like(lct_matrix, dtype=bool)
        
        for i, r in enumerate(r_vals):
            for j, d in enumerate(d_vals):
                try:
                    if d >= 0 and d <= self.g - 1 and r <= self.g - 1:
                        bn = HyperellipticBrillNoether(self.g, d, r)
                        lct_matrix[i, j] = bn.log_canonical_threshold()
                        valid_matrix[i, j] = True
                    else:
                        lct_matrix[i, j] = np.nan
                except:
                    lct_matrix[i, j] = np.nan
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Mask invalid entries
        masked_lct = np.ma.masked_where(~valid_matrix, lct_matrix)
        
        im = ax.imshow(masked_lct, cmap='viridis', aspect='auto', origin='lower',
                      extent=[d_vals[0]-0.5, d_vals[-1]+0.5, r_vals[0]-0.5, r_vals[-1]+0.5])
        
        ax.set_xlabel('d (degree)', fontsize=12)
        ax.set_ylabel('r (rank)', fontsize=12)
        ax.set_title(f'Log Canonical Thresholds for Genus {self.g}', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log canonical threshold', fontsize=12)
        
        # Add text annotations
        for i, r in enumerate(r_vals):
            for j, d in enumerate(d_vals):
                if valid_matrix[i, j]:
                    val = lct_matrix[i, j]
                    ax.text(d, r, f'{val:.1f}', ha='center', va='center', 
                           color='white' if val < 1.5 else 'black', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_decomposition_diagram(self, figsize: Tuple[int, int] = (12, 6)):
        """Visualize the decomposition from Theorem 1.4(ii)."""
        m = self.m
        if m < 0:
            print("No decomposition to show (m < 0)")
            return None
            
        max_s = m // 2
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Decomposition diagram
        y_positions = range(max_s + 1)
        dimensions = [m - 2*s for s in range(max_s + 1)]
        
        bars = ax1.barh(y_positions, dimensions, color=plt.cm.viridis(np.linspace(0.3, 0.9, max_s + 1)))
        ax1.set_yticks(y_positions)
        ax1.set_yticklabels([f'Z_{s}' for s in range(max_s + 1)])
        ax1.set_xlabel('Dimension', fontsize=12)
        ax1.set_title('Decomposition into Supports', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add dimension labels
        for i, (bar, dim) in enumerate(zip(bars, dimensions)):
            ax1.text(dim + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'dim = {dim}', va='center', fontsize=10)
        
        # Right: Cohomological shifts
        shifts = [-s for s in range(max_s + 1)]
        colors = ['#2E86AB' if s % 2 == 0 else '#A23B72' for s in range(max_s + 1)]
        
        ax2.bar(range(max_s + 1), shifts, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('s', fontsize=12)
        ax2.set_ylabel('Hodge shift', fontsize=12)
        ax2.set_title('Hodge Shifts in Decomposition', fontsize=12)
        ax2.set_xticks(range(max_s + 1))
        ax2.set_xticklabels([f'Z_{s}' for s in range(max_s + 1)])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_parameters_space(self, figsize: Tuple[int, int] = (12, 8)):
        """Visualize the parameter space and validity regions."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create grid of valid (d, r) pairs
        d_vals = np.arange(0, self.g)
        r_vals = np.arange(0, self.g)
        
        # Validity matrix
        validity = np.zeros((len(r_vals), len(d_vals)))
        for i, r in enumerate(r_vals):
            for j, d in enumerate(d_vals):
                try:
                    bn = HyperellipticBrillNoether(self.g, d, r)
                    if bn.b >= 0 and bn.ell_prime >= bn.ell:
                        validity[i, j] = 1
                except:
                    validity[i, j] = 0
        
        im = ax.imshow(validity, cmap='RdYlGn', extent=[-0.5, len(d_vals)-0.5, 
                                                         -0.5, len(r_vals)-0.5],
                      origin='lower', alpha=0.7)
        
        # Mark current point
        ax.plot(self.d, self.r, 'ro', markersize=12, label='Current')
        
        ax.set_xlabel('d (degree)', fontsize=12)
        ax.set_ylabel('r (rank)', fontsize=12)
        ax.set_title(f'Valid (d, r) Pairs for Genus {self.g}', fontsize=14)
        
        # Add grid lines
        ax.set_xticks(range(len(d_vals)))
        ax.set_yticks(range(len(r_vals)))
        ax.set_xticklabels(d_vals)
        ax.set_yticklabels(r_vals)
        
        # Add some labels for interesting points
        ax.text(0, 0, '0', ha='center', va='center', fontsize=8, color='black')
        if self.g > 1:
            ax.text(self.g-1, 0, 'theta\ndivisor', ha='center', va='center', fontsize=8, color='black')
        
        plt.colorbar(im, ax=ax, label='Valid')
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_comprehensive(self, z: Optional[np.ndarray] = None, save: bool = False):
        """Generate a comprehensive visualization suite."""
        if z is None:
            z = self.get_example_z()
        
        print(f"\n{'='*60}")
        print(f"HYPER-ELLIPTIC BRILL-NOETHER LOCUS W_{self.d}^{self.r}(C)")
        print(f"Genus g = {self.g}")
        print(f"{'='*60}")
        print(f"Parameters:")
        print(f"  ℓ = {self.ell}, ℓ' = {self.ell_prime}, b = {self.b}")
        print(f"  m = {self.m}, c = {self.c}")
        print(f"  z length = {self.z_length}")
        print(f"  Multiplicity = {self.multiplicity()}")
        print(f"  Log canonical threshold = {self.log_canonical_threshold():.4f}")
        print(f"{'='*60}\n")
        
        # Generate plots
        self.plot_catalecticant_matrix(z)
        self.plot_betti_numbers()
        self.plot_hodge_diamond()
        self.plot_decomposition_diagram()
        self.plot_parameters_space()
        
        if self.g > 3:
            self.plot_multiplicity_vs_genus(max_g=self.g+3)
            self.plot_lct_heatmap()


# ============ ADDITIONAL VISUALIZATION FUNCTIONS ============

def plot_secant_variety(n: int, s: int, num_points: int = 200, figsize: Tuple[int, int] = (10, 8)):
    """Visualize the secant variety of a rational normal curve."""
    from sklearn.decomposition import PCA
    
    # Generate points on the secant variety
    t = np.random.randn(s, num_points)
    curve_points = np.zeros((s, num_points, n + 1))
    for i in range(s):
        for j in range(num_points):
            for k in range(n + 1):
                curve_points[i, j, k] = t[i, j] ** k
    
    # Take random convex combinations
    weights = np.random.dirichlet(np.ones(s), size=num_points)
    points = np.zeros((num_points, n + 1))
    for i in range(num_points):
        for j in range(s):
            points[i] += weights[i, j] * curve_points[j, i]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if n <= 3:
        # For n <= 3, we can visualize directly
        ax = axes[0]
        if n == 1:
            ax.scatter(points[:, 0], points[:, 1], alpha=0.5, s=10)
        elif n == 2:
            ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], 
                      cmap='viridis', alpha=0.5, s=10)
        elif n == 3:
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=points[:, 3], cmap='viridis', alpha=0.5, s=10)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        
        ax.set_title(f'{s}-Secant Variety of degree {n} rational normal curve')
    else:
        # For higher dimensions, use PCA
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(points)
        axes[0].scatter(points_2d[:, 0], points_2d[:, 1], alpha=0.5, s=10)
        axes[0].set_title(f'{s}-Secant Variety (PCA projection)')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
    
    # Right: Show the curve itself
    ax2 = axes[1]
    t_curve = np.linspace(-2, 2, 100)
    curve = np.zeros((len(t_curve), n + 1))
    for i, t_val in enumerate(t_curve):
        for k in range(n + 1):
            curve[i, k] = t_val ** k
    
    if n <= 3:
        if n == 1:
            ax2.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2)
        elif n == 2:
            ax2.scatter(curve[:, 0], curve[:, 1], c=t_curve, cmap='plasma', s=20)
        elif n == 3:
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'b-', linewidth=2)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('z')
    else:
        curve_2d = pca.transform(curve)
        ax2.plot(curve_2d[:, 0], curve_2d[:, 1], 'b-', linewidth=2)
        ax2.set_title('Rational Normal Curve (PCA projection)')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
    
    ax2.set_title(f'Degree {n} Rational Normal Curve')
    plt.tight_layout()
    plt.show()


def plot_brill_noether_phase_diagram(g: int, figsize: Tuple[int, int] = (12, 10)):
    """Create a phase diagram showing different regions of the (d, r) parameter space."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    d_vals = np.arange(0, g)
    r_vals = np.arange(0, g)
    
    # Matrix for each quantity
    dim_matrix = np.zeros((len(r_vals), len(d_vals)))
    mult_matrix = np.zeros_like(dim_matrix)
    lct_matrix = np.zeros_like(dim_matrix)
    valid_matrix = np.zeros_like(dim_matrix, dtype=bool)
    
    for i, r in enumerate(r_vals):
        for j, d in enumerate(d_vals):
            try:
                bn = HyperellipticBrillNoether(g, d, r)
                if bn.b >= 0 and bn.ell_prime >= bn.ell:
                    dim_matrix[i, j] = bn.m
                    mult_matrix[i, j] = bn.multiplicity()
                    lct_matrix[i, j] = bn.log_canonical_threshold()
                    valid_matrix[i, j] = True
                else:
                    dim_matrix[i, j] = np.nan
                    mult_matrix[i, j] = np.nan
                    lct_matrix[i, j] = np.nan
            except:
                dim_matrix[i, j] = np.nan
                mult_matrix[i, j] = np.nan
                lct_matrix[i, j] = np.nan
    
    # Plot dimension
    im1 = ax1.imshow(dim_matrix, cmap='coolwarm', origin='lower', 
                     extent=[-0.5, len(d_vals)-0.5, -0.5, len(r_vals)-0.5])
    ax1.set_title('Dimension m = d - 2r')
    ax1.set_xlabel('d')
    ax1.set_ylabel('r')
    plt.colorbar(im1, ax=ax1)
    
    # Plot multiplicity
    masked_mult = np.ma.masked_where(~valid_matrix, mult_matrix)
    im2 = ax2.imshow(masked_mult, cmap='viridis', origin='lower',
                     extent=[-0.5, len(d_vals)-0.5, -0.5, len(r_vals)-0.5])
    ax2.set_title('Multiplicity')
    ax2.set_xlabel('d')
    ax2.set_ylabel('r')
    plt.colorbar(im2, ax=ax2)
    
    # Plot log canonical threshold
    masked_lct = np.ma.masked_where(~valid_matrix, lct_matrix)
    im3 = ax3.imshow(masked_lct, cmap='plasma', origin='lower',
                     extent=[-0.5, len(d_vals)-0.5, -0.5, len(r_vals)-0.5])
    ax3.set_title('Log Canonical Threshold')
    ax3.set_xlabel('d')
    ax3.set_ylabel('r')
    plt.colorbar(im3, ax=ax3)
    
    # Plot validity
    im4 = ax4.imshow(valid_matrix, cmap='RdYlGn', origin='lower',
                     extent=[-0.5, len(d_vals)-0.5, -0.5, len(r_vals)-0.5])
    ax4.set_title('Valid Region')
    ax4.set_xlabel('d')
    ax4.set_ylabel('r')
    plt.colorbar(im4, ax=ax4)
    
    # Add grid lines
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(range(len(d_vals)))
        ax.set_yticks(range(len(r_vals)))
        ax.set_xticklabels(d_vals)
        ax.set_yticklabels(r_vals)
    
    fig.suptitle(f'Brill-Noether Parameter Space for Genus {g}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ============ EXAMPLE USAGE ============

def example_visualization():
    """Comprehensive example with visualizations."""
    
    print("="*60)
    print("HYPER-ELLIPTIC BRILL-NOETHER LOCI VISUALIZATION")
    print("="*60)
    
    # Example 1: Genus 3, degree 2, rank 0
    print("\nExample 1: g=3, d=2, r=0")
    print("-"*40)
    bn1 = HyperellipticBrillNoether(g=3, d=2, r=0)
    z1 = bn1.get_example_z()
    print(f"z vector (length {len(z1)}): {z1}")
    bn1.plot_comprehensive(z1)
    
    # Example 2: Genus 4, degree 2, rank 0
    print("\nExample 2: g=4, d=2, r=0")
    print("-"*40)
    bn2 = HyperellipticBrillNoether(g=4, d=2, r=0)
    z2 = bn2.get_random_z()
    print(f"z vector (length {len(z2)}): {z2[:5]}...")
    bn2.plot_comprehensive(z2)
    
    # Additional visualizations
    try:
        print("\nVisualizing Secant Varieties...")
        plot_secant_variety(n=3, s=2, num_points=100)
    except ImportError:
        print("sklearn not installed, skipping secant variety visualization")
    
    print("\nPhase Diagram for genus 5...")
    plot_brill_noether_phase_diagram(g=5)


# Quick test
if __name__ == "__main__":
    example_visualization()