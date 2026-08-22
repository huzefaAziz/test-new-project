import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import warnings
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

@dataclass
class LineBundle:
    """Represents a line bundle on a curve with degree and torsion information."""
    degree: int
    is_torsion: bool = False
    torsion_order: int = 1
    
    def __add__(self, other):
        if not isinstance(other, LineBundle):
            raise TypeError("Can only add LineBundle objects")
        return LineBundle(
            degree=self.degree + other.degree,
            is_torsion=self.is_torsion or other.is_torsion,
            torsion_order=max(self.torsion_order, other.torsion_order)
        )
    
    def __mul__(self, n):
        if not isinstance(n, int):
            raise TypeError("Can only multiply by integers")
        return LineBundle(
            degree=self.degree * n,
            is_torsion=self.is_torsion and n % 2 == 1,
            torsion_order=self.torsion_order if self.is_torsion and n % 2 == 1 else 1
        )
    
    def inverse(self):
        return LineBundle(-self.degree, self.is_torsion, self.torsion_order)

@dataclass
class EllipticSurfaceData:
    """Data structure for a marked elliptic surface with double fibers."""
    base_curve_genus: int
    num_double_fibers: int
    double_fiber_types: List[str]
    epsilon_1: LineBundle
    epsilon_2_neg: LineBundle
    det_epsilon_1: LineBundle
    branch_divisor: 'BranchDivisor'
    singularities: List[Dict]

@dataclass
class BranchDivisor:
    """Branch divisor for the birational double cover model."""
    b0: List[Dict]
    gamma_fibers: List[Dict]
    local_singularities: List[Tuple[int, int]]

class ComplexDynamics:
    """Complex dynamics for Mandelbrot and Julia sets."""
    
    @staticmethod
    def mandelbrot(c: np.ndarray, max_iter: int = 100, threshold: float = 2.0) -> np.ndarray:
        """Compute Mandelbrot set iteration counts."""
        z = np.zeros_like(c, dtype=np.complex128)
        iteration_counts = np.zeros(c.shape, dtype=np.int32)
        
        for i in range(max_iter):
            # z = z^2 + c
            mask = np.abs(z) < threshold
            z[mask] = z[mask]**2 + c[mask]
            iteration_counts[mask] = i
            
        return iteration_counts
    
    @staticmethod
    def julia(c: complex, z_grid: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Compute Julia set for parameter c."""
        z = z_grid.copy()
        iteration_counts = np.zeros(z_grid.shape, dtype=np.int32)
        threshold = 2.0
        
        for i in range(max_iter):
            mask = np.abs(z) < threshold
            z[mask] = z[mask]**2 + c
            iteration_counts[mask] = i
            
        return iteration_counts
    
    @staticmethod
    def buddhabrot(c: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Compute Buddhabrot (orbits of Mandelbrot set)."""
        # Initialize orbit accumulation
        orbit_density = np.zeros((1000, 1000))
        threshold = 2.0
        
        for point in c.flatten():
            z = 0j
            orbit = []
            
            for i in range(max_iter):
                if abs(z) > threshold:
                    break
                z = z**2 + point
                # Normalize orbit point to image coordinates
                x = int((z.real + 2) / 4 * 999)
                y = int((z.imag + 2) / 4 * 999)
                if 0 <= x < 1000 and 0 <= y < 1000:
                    orbit_density[y, x] += 1
                    
        return orbit_density

class MandelbrotEllipticSurface:
    """
    Combines Mandelbrot set dynamics with elliptic surface geometry.
    Maps the parameter space of the Mandelbrot set to the base curve C
    and fiber data of the elliptic surface.
    """
    
    def __init__(self, surface_data: EllipticSurfaceData, resolution: int = 500):
        self.data = surface_data
        self.resolution = resolution
        self._build_complex_structures()
        self._build_elliptic_fibration()
    
    def _build_complex_structures(self):
        """Build complex plane structures for parameter mapping."""
        # Parameter space for Mandelbrot set
        x = np.linspace(-2.5, 1.5, self.resolution)
        y = np.linspace(-1.5, 1.5, self.resolution)
        self.X, self.Y = np.meshgrid(x, y)
        self.C = self.X + 1j * self.Y
        
        # Compute Mandelbrot set
        self.mandelbrot_iterations = ComplexDynamics.mandelbrot(self.C)
        self.mandelbrot_normalized = self.mandelbrot_iterations / np.max(self.mandelbrot_iterations)
        
        # Compute interesting Julia parameters from Mandelbrot set
        self.julia_params = self._extract_julia_parameters()
    
    def _extract_julia_parameters(self) -> List[complex]:
        """Extract interesting Julia parameters from Mandelbrot set boundary."""
        params = []
        # Select points near the boundary of the Mandelbrot set
        boundary_points = [
            -0.5 + 0.0j,  # Central bulb
            0.285 + 0.01j,  # Seahorse valley
            -0.8 + 0.15j,  # Elephant valley
            -1.3 + 0.0j,  # Period 2 bulb
            0.35 + 0.65j,  # Period 3 bulb
            -0.75 + 0.1j,  # Douady rabbit
            0.35 - 0.35j,  # Dendrite
        ]
        
        # Add points near the boundary
        for base_point in boundary_points:
            # Slightly perturb to get interesting boundary behavior
            for perturb in [0.0, 0.01, -0.01, 0.01j, -0.01j]:
                params.append(base_point + perturb)
                
        return params
    
    def _build_elliptic_fibration(self):
        """Build elliptic fibration structure using Mandelbrot parameter map."""
        # Map Mandelbrot parameters to base curve points
        self.parameter_to_base = np.zeros_like(self.C, dtype=np.float64)
        
        # Use Mandelbrot iteration count to determine position on base curve
        for i in range(self.resolution):
            for j in range(self.resolution):
                # Map iteration count to base curve position
                iter_count = self.mandelbrot_iterations[i, j]
                if iter_count > 0:
                    # Normalize to [0, 1] for base curve
                    self.parameter_to_base[i, j] = np.tanh(iter_count / 50)
                else:
                    self.parameter_to_base[i, j] = 0
        
        # Build fiber structure using branch divisor data
        self._build_fiber_structure()
    
    def _build_fiber_structure(self):
        """Build fiber structure with double fibers from elliptic surface data."""
        # Define fiber types based on Mandelbrot regions
        self.fiber_types = np.zeros_like(self.C, dtype=np.int32)
        
        # Regions where double fibers occur
        num_double = self.data.num_double_fibers
        for k in range(num_double):
            # Double fibers correspond to Mandelbrot bulbs
            center = -0.5 + 0.0j + k * 0.3
            radius = 0.15 + k * 0.02
            mask = np.abs(self.C - center) < radius
            self.fiber_types[mask] = k + 1
            
            # Add branch divisor singularities at double fiber locations
            for i, singularities in enumerate(self.data.branch_divisor.local_singularities):
                if i == k:
                    n = singularities[1]
                    # J_{2,n} singularity influences local dynamics
                    self._apply_singularity_effects(k, n)
    
    def _apply_singularity_effects(self, fiber_idx: int, n: int):
        """Apply J_{2,n} singularity effects to local complex dynamics."""
        mask = self.fiber_types == (fiber_idx + 1)
        if np.any(mask):
            # Modify iteration counts near singularities
            local_iter = self.mandelbrot_iterations[mask]
            # J_{2,n} singularities create specific branching patterns
            local_iter = local_iter * (1 + 0.1 * n * np.sin(local_iter / 10))
            self.mandelbrot_iterations[mask] = local_iter
            
            # Update normalized values
            self.mandelbrot_normalized[mask] = local_iter / np.max(local_iter)
    
    def compute_3d_manifold(self) -> np.ndarray:
        """
        Compute 3D manifold by combining Mandelbrot set with elliptic surface structure.
        This creates a 3D visualization where:
        - X,Y: Complex plane coordinates
        - Z: Elliptic surface height (fiber structure)
        """
        # Base surface from Mandelbrot set
        z_surface = np.zeros_like(self.C, dtype=np.float64)
        
        # Build height function from Mandelbrot dynamics and elliptic structure
        for i in range(self.resolution):
            for j in range(self.resolution):
                c = self.C[i, j]
                
                # Base height from Mandelbrot iteration count
                height = self.mandelbrot_normalized[i, j]
                
                # Add elliptic fiber structure
                fiber_type = self.fiber_types[i, j]
                if fiber_type > 0:
                    # Double fiber adds height
                    height += 0.3 * (1 - np.cos(2 * np.pi * fiber_type / self.data.num_double_fibers))
                    
                    # J_{2,n} singularities create peaks
                    for k, (idx, n) in enumerate(self.data.branch_divisor.local_singularities):
                        if idx == fiber_type - 1:
                            height += 0.2 * n * np.exp(-((height - 0.5)**2) / 0.1)
                
                # Add Julia set boundary effects
                if np.abs(c) < 2:
                    # Check if near Mandelbrot boundary
                    boundary_mask = (self.mandelbrot_iterations[i, j] > 80) & (self.mandelbrot_iterations[i, j] < 95)
                    if boundary_mask:
                        height += 0.5 * np.sin(np.abs(c) * 10)
                
                z_surface[i, j] = height
        
        # Smooth the surface
        z_surface = gaussian_filter(z_surface, sigma=2)
        
        return z_surface
    
    def compute_3d_julia_fibration(self, julia_param: complex) -> np.ndarray:
        """
        Compute 3D Julia fibration: combination of Julia set and elliptic surface.
        """
        # Compute Julia set for parameter
        z_grid = self.C.copy()
        julia_iter = ComplexDynamics.julia(julia_param, z_grid)
        julia_normalized = julia_iter / np.max(julia_iter)
        
        # Build 3D surface with elliptic structure
        z_surface = np.zeros_like(self.C, dtype=np.float64)
        
        # Threshold and normalize
        julia_mask = julia_iter > 0
        z_surface[julia_mask] = julia_normalized[julia_mask]
        
        # Add elliptic fiber structure
        for i in range(self.resolution):
            for j in range(self.resolution):
                if julia_mask[i, j]:
                    # Map Julia dynamics to elliptic fiber
                    c = self.C[i, j]
                    
                    # Fiber height based on Julia dynamics
                    fiber_height = 0.5 * (1 - np.cos(2 * np.pi * np.abs(z_grid[i, j])))
                    
                    # Add double fiber effects
                    for k, (idx, n) in enumerate(self.data.branch_divisor.local_singularities):
                        if np.abs(self.C[i, j] - self._get_double_fiber_center(idx)) < 0.2:
                            fiber_height += 0.3 * n * np.exp(-((np.abs(z_grid[i, j]) - 0.5)**2) / 0.1)
                    
                    z_surface[i, j] += fiber_height * 0.5
        
        return z_surface
    
    def _get_double_fiber_center(self, idx: int) -> complex:
        """Get center of double fiber in complex plane."""
        return -0.5 + 0.0j + idx * 0.3
    
    def visualize_3d(self, mode: str = 'mandelbrot'):
        """
        Create 3D visualization of combined Mandelbrot and elliptic surface.
        
        Args:
            mode: 'mandelbrot' for Mandelbrot surface,
                  'julia' for Julia fibration
        """
        fig = plt.figure(figsize=(16, 12))
        
        if mode == 'mandelbrot':
            ax = fig.add_subplot(111, projection='3d')
            
            # Compute 3D manifold
            Z = self.compute_3d_manifold()
            
            # Create surface plot with color mapping from Mandelbrot iterations
            colors = plt.cm.hot(self.mandelbrot_normalized)
            
            # Plot surface
            surf = ax.plot_surface(self.X, self.Y, Z, 
                                 facecolors=colors,
                                 rstride=2, cstride=2,
                                 alpha=0.9, antialiased=True)
            
            ax.set_xlabel('Real(z)', fontsize=12)
            ax.set_ylabel('Imag(z)', fontsize=12)
            ax.set_zlabel('Elliptic Surface Height', fontsize=12)
            
            # Add double fiber markers
            for i in range(self.data.num_double_fibers):
                center = self._get_double_fiber_center(i)
                # Find closest point on surface
                idx = np.argmin(np.abs(self.C - center))
                x_idx = idx // self.resolution
                y_idx = idx % self.resolution
                if x_idx < self.resolution and y_idx < self.resolution:
                    ax.scatter(center.real, center.imag, Z[x_idx, y_idx],
                             color='red', s=100, marker='*', 
                             label=f'Double Fiber {i+1}')
            
            ax.set_title('Mandelbrot Set as Elliptic Surface with Double Fibers', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            
        elif mode == 'julia':
            # Create 3D Julia fibration with multiple parameters
            ax = fig.add_subplot(111, projection='3d')
            
            # Use first Julia parameter
            if self.julia_params:
                param = self.julia_params[0]
                Z = self.compute_3d_julia_fibration(param)
                
                # Plot surface
                colors = plt.cm.viridis(Z / np.max(Z) if np.max(Z) > 0 else Z)
                surf = ax.plot_surface(self.X, self.Y, Z,
                                     facecolors=colors,
                                     rstride=2, cstride=2,
                                     alpha=0.9, antialiased=True)
                
                ax.set_xlabel('Real(z)', fontsize=12)
                ax.set_ylabel('Imag(z)', fontsize=12)
                ax.set_zlabel('Julia Fibration Height', fontsize=12)
                ax.set_title(f'3D Julia Fibration for c = {param:.3f}', 
                           fontsize=14, fontweight='bold')
        
        # Adjust viewing angle
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        return fig
    
    def visualize_combined(self):
        """Create combined visualization with multiple views."""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Mandelbrot set with fiber overlay (2D)
        ax1 = fig.add_subplot(231)
        im1 = ax1.imshow(self.mandelbrot_iterations, cmap='hot', 
                        extent=[-2.5, 1.5, -1.5, 1.5])
        ax1.set_title('Mandelbrot Set with Double Fibers')
        ax1.set_xlabel('Real(z)')
        ax1.set_ylabel('Imag(z)')
        
        # Add double fiber markers on Mandelbrot set
        for i in range(self.data.num_double_fibers):
            center = self._get_double_fiber_center(i)
            idx = np.argmin(np.abs(self.C - center))
            x_idx = idx // self.resolution
            y_idx = idx % self.resolution
            if x_idx < self.resolution and y_idx < self.resolution:
                ax1.plot(y_idx, x_idx, 'r*', markersize=10, markeredgewidth=2)
        
        plt.colorbar(im1, ax=ax1, label='Iterations')
        
        # 2. 3D Mandelbrot elliptic surface
        ax2 = fig.add_subplot(232, projection='3d')
        Z_mandel = self.compute_3d_manifold()
        colors_mandel = plt.cm.hot(self.mandelbrot_normalized)
        surf2 = ax2.plot_surface(self.X, self.Y, Z_mandel,
                                facecolors=colors_mandel,
                                rstride=2, cstride=2,
                                alpha=0.8, antialiased=True)
        ax2.set_title('3D Mandelbrot Elliptic Surface')
        ax2.set_xlabel('Re')
        ax2.set_ylabel('Im')
        ax2.set_zlabel('Height')
        ax2.view_init(elev=25, azim=60)
        
        # 3. 3D Julia fibration
        ax3 = fig.add_subplot(233, projection='3d')
        if self.julia_params:
            param = self.julia_params[0]
            Z_julia = self.compute_3d_julia_fibration(param)
            colors_julia = plt.cm.viridis(Z_julia / np.max(Z_julia) if np.max(Z_julia) > 0 else Z_julia)
            surf3 = ax3.plot_surface(self.X, self.Y, Z_julia,
                                    facecolors=colors_julia,
                                    rstride=2, cstride=2,
                                    alpha=0.8, antialiased=True)
            ax3.set_title(f'3D Julia Fibration\nc = {param:.3f}')
            ax3.set_xlabel('Re')
            ax3.set_ylabel('Im')
            ax3.set_zlabel('Height')
            ax3.view_init(elev=25, azim=60)
        
        # 4. Cross-section of elliptic surface
        ax4 = fig.add_subplot(234)
        # Show cross-section at y=0
        y_zero_idx = self.resolution // 2
        x_vals = self.X[y_zero_idx, :]
        z_cross = Z_mandel[y_zero_idx, :]
        fiber_cross = self.fiber_types[y_zero_idx, :]
        
        ax4.plot(x_vals, z_cross, 'b-', linewidth=2, label='Surface cross-section')
        # Highlight double fibers
        fiber_mask = fiber_cross > 0
        if np.any(fiber_mask):
            ax4.scatter(x_vals[fiber_mask], z_cross[fiber_mask], 
                       color='red', s=50, marker='*', label='Double fibers')
        ax4.set_xlabel('Real(z)')
        ax4.set_ylabel('Surface Height')
        ax4.set_title('Cross-section at y=0')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. Branch divisor visualization
        ax5 = fig.add_subplot(235)
        # Compute branch divisor from elliptic surface data
        branch_points = []
        for i, (idx, n) in enumerate(self.data.branch_divisor.local_singularities):
            center = self._get_double_fiber_center(idx)
            # J_{2,n} singularity creates branch points
            for k in range(4):  # 4 branch points per fiber
                angle = 2 * np.pi * k / 4 + i * 0.2
                point = center + 0.1 * (np.cos(angle) + 1j * np.sin(angle))
                branch_points.append((point.real, point.imag, n))
        
        branch_points = np.array(branch_points)
        if len(branch_points) > 0:
            scatter = ax5.scatter(branch_points[:, 0], branch_points[:, 1],
                                c=branch_points[:, 2], cmap='plasma',
                                s=50, alpha=0.8)
            ax5.set_title('Branch Divisor B = B0 + ΣΓᵢ')
            ax5.set_xlabel('Real(z)')
            ax5.set_ylabel('Imag(z)')
            plt.colorbar(scatter, ax=ax5, label='J_{2,n} type')
        
        # 6. Fiber structure graph
        ax6 = fig.add_subplot(236)
        # Build fiber graph using NetworkX
        G = nx.Graph()
        
        # Add base vertices
        for i in range(max(5, self.data.base_curve_genus * 2 + 1)):
            G.add_node(f'B{i}', type='base')
        
        # Add fiber vertices
        for i, fiber_type in enumerate(self.data.double_fiber_types):
            G.add_node(f'F{i}', type='fiber', kodaira=fiber_type)
            # Connect to base
            for j in range(2):
                G.add_edge(f'F{i}', f'B{j}')
        
        # Draw graph
        pos = nx.spring_layout(G, k=0.5)
        node_colors = ['lightblue' if G.nodes[n]['type'] == 'base' else 'orange' 
                      for n in G.nodes()]
        nx.draw(G, pos, ax=ax6, node_color=node_colors, 
               with_labels=True, node_size=500, font_size=8)
        ax6.set_title('Fiber Structure of Elliptic Surface')
        
        plt.tight_layout()
        return fig

class EllipticSurfaceConstructor:
    """Constructs marked elliptic surfaces with double fibers."""
    
    @staticmethod
    def from_halphen_index_2() -> EllipticSurfaceData:
        """Construct a Halphen surface of index 2."""
        data = EllipticSurfaceData(
            base_curve_genus=0,
            num_double_fibers=9,
            double_fiber_types=['I0'] * 9,
            epsilon_1=LineBundle(degree=-1),
            epsilon_2_neg=LineBundle(degree=4),
            det_epsilon_1=LineBundle(degree=-1),
            branch_divisor=None,
            singularities=[{'n': 0, 'type': 'T_2,3,6'}] * 9
        )
        
        # Build branch divisor
        data.branch_divisor = BranchDivisor(
            b0=[{
                'fiber_index': i,
                'singularity': f'A_3',
                'intersection_multiplicity': 4
            } for i in range(9)],
            gamma_fibers=[{
                'index': i,
                'type': 'I0',
                'intersection_with_b0': 4
            } for i in range(9)],
            local_singularities=[(i, 0) for i in range(9)]
        )
        
        return data
    
    @staticmethod
    def from_enriques_surface(special: bool = False) -> EllipticSurfaceData:
        """Construct an Enriques surface."""
        if special:
            data = EllipticSurfaceData(
                base_curve_genus=1,
                num_double_fibers=1,
                double_fiber_types=['I0'],
                epsilon_1=LineBundle(degree=-1, is_torsion=True, torsion_order=2),
                epsilon_2_neg=LineBundle(degree=2),
                det_epsilon_1=LineBundle(degree=-1, is_torsion=True, torsion_order=2),
                branch_divisor=None,
                singularities=[{'n': 1, 'type': 'T_2,3,7'}]
            )
            data.branch_divisor = BranchDivisor(
                b0=[{'fiber_index': 0, 'singularity': 'A_4', 'intersection_multiplicity': 4}],
                gamma_fibers=[{'index': 0, 'type': 'I0', 'intersection_with_b0': 4}],
                local_singularities=[(0, 1)]
            )
        else:
            data = EllipticSurfaceData(
                base_curve_genus=1,
                num_double_fibers=2,
                double_fiber_types=['I0', 'I0'],
                epsilon_1=LineBundle(degree=-1, is_torsion=True, torsion_order=2),
                epsilon_2_neg=LineBundle(degree=2),
                det_epsilon_1=LineBundle(degree=-1, is_torsion=True, torsion_order=2),
                branch_divisor=None,
                singularities=[{'n': 0, 'type': 'T_2,3,6'}, {'n': 0, 'type': 'T_2,3,6'}]
            )
            data.branch_divisor = BranchDivisor(
                b0=[{'fiber_index': i, 'singularity': 'A_3', 'intersection_multiplicity': 4} for i in range(2)],
                gamma_fibers=[{'index': i, 'type': 'I0', 'intersection_with_b0': 4} for i in range(2)],
                local_singularities=[(i, 0) for i in range(2)]
            )
        
        return data

def demo_mandelbrot_elliptic():
    """Demonstration of combined Mandelbrot and elliptic surface visualization."""
    print("=" * 70)
    print("Combined Mandelbrot Set and Marked Elliptic Surfaces")
    print("Based on arXiv:2608.19970v1")
    print("=" * 70)
    
    # Create Halphen surface of index 2
    print("\n1. Creating Halphen surface of index 2...")
    halphen_data = EllipticSurfaceConstructor.from_halphen_index_2()
    
    # Create combined visualization
    print("2. Building Mandelbrot elliptic surface...")
    mandel_elliptic = MandelbrotEllipticSurface(halphen_data, resolution=300)
    
    # Generate visualizations
    print("3. Generating 3D visualizations...")
    
    # 3D Mandelbrot surface
    fig1 = mandel_elliptic.visualize_3d(mode='mandelbrot')
    plt.show()
    
    # 3D Julia fibration
    fig2 = mandel_elliptic.visualize_3d(mode='julia')
    plt.show()
    
    # Combined views
    fig3 = mandel_elliptic.visualize_combined()
    plt.show()
    
    # Also demonstrate Enriques surfaces
    print("\n4. Enriques surfaces...")
    
    # Special Enriques
    print("   Special Enriques surface (Section 8.B):")
    special_data = EllipticSurfaceConstructor.from_enriques_surface(special=True)
    special_mandel = MandelbrotEllipticSurface(special_data, resolution=200)
    fig4 = special_mandel.visualize_3d(mode='mandelbrot')
    plt.show()
    
    # Non-special Enriques
    print("   Non-special Enriques surface (Section 8.C):")
    nonspecial_data = EllipticSurfaceConstructor.from_enriques_surface(special=False)
    nonspecial_mandel = MandelbrotEllipticSurface(nonspecial_data, resolution=200)
    fig5 = nonspecial_mandel.visualize_3d(mode='mandelbrot')
    plt.show()
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("The 3D surfaces combine Mandelbrot dynamics with elliptic surface geometry.")
    print("Red stars mark double fibers with J_{2,n} singularities.")
    print("=" * 70)

# Additional: Interactive 3D viewer with parameter exploration

class InteractiveMandelbrotElliptic:
    """Interactive exploration of Mandelbrot elliptic surfaces."""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.current_view = 'mandelbrot'
        self.rotation_angle = 0
        
    def create_interactive_animation(self, surface_data: EllipticSurfaceData):
        """Create animated 3D visualization rotating through different views."""
        mandel_elliptic = MandelbrotEllipticSurface(surface_data, resolution=200)
        Z = mandel_elliptic.compute_3d_manifold()
        
        # Create figure with animation
        from matplotlib.animation import FuncAnimation
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initial plot
        colors = plt.cm.hot(mandel_elliptic.mandelbrot_normalized)
        surf = ax.plot_surface(mandel_elliptic.X, mandel_elliptic.Y, Z,
                              facecolors=colors,
                              rstride=2, cstride=2,
                              alpha=0.8, antialiased=True)
        
        ax.set_xlabel('Real(z)')
        ax.set_ylabel('Imag(z)')
        ax.set_zlabel('Height')
        ax.set_title('Mandelbrot Elliptic Surface - Rotating')
        
        def update(frame):
            ax.view_init(elev=30, azim=frame * 2)
            return surf,
        
        anim = FuncAnimation(fig, update, frames=180, interval=50, blit=False)
        plt.show()
        return anim

if __name__ == "__main__":
    demo_mandelbrot_elliptic()
    
    # Interactive animation (uncomment to run)
    # print("\nCreating interactive animation...")
    # halphen_data = EllipticSurfaceConstructor.from_halphen_index_2()
    # interactive = InteractiveMandelbrotElliptic()
    # interactive.create_interactive_animation(halphen_data)