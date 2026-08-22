import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import warnings

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
    double_fiber_types: List[str]  # Kodaira types of reduced fibers
    epsilon_1: LineBundle  # f_* L
    epsilon_2_neg: LineBundle  # anti-invariant part of f_* L^2
    det_epsilon_1: LineBundle
    branch_divisor: 'BranchDivisor'
    singularities: List[Dict]  # Descriptions of singularities over double fibers

@dataclass
class BranchDivisor:
    """Branch divisor for the birational double cover model."""
    b0: 'DivisorOnRuledSurface'
    gamma_fibers: List['DivisorOnRuledSurface']
    local_singularities: List[Tuple[int, int]]  # (n, m) for J_{n,m} singularities

@dataclass
class DivisorOnRuledSurface:
    """Divisor on a ruled surface P(E) over C."""
    base_component: LineBundle
    fiber_coefficients: Dict[int, int]  # coefficient at fiber over point i
    section_coefficients: Dict[int, int]  # coefficient at section components
    
    def intersection_with_fiber(self, fiber_index: int) -> int:
        """Compute intersection with a fiber."""
        return self.fiber_coefficients.get(fiber_index, 0)
    
    def intersection_with_section(self, section_index: int) -> int:
        """Compute intersection with a section."""
        return self.section_coefficients.get(section_index, 0)

class WeightedProjectiveBundle:
    """Implementation of weighted projective bundle P(w1,...,wn)(F1,...,Fn)."""
    
    def __init__(self, weights: List[int], base: 'Curve'):
        self.weights = weights
        self.base = base
        self.variables = []
        self._build_variables()
    
    def _build_variables(self):
        """Build weighted homogeneous variables."""
        for i, w in enumerate(self.weights):
            self.variables.append({
                'name': f'x_{i}',
                'weight': w,
                'degree': w
            })
    
    def coordinate_ring(self) -> Dict:
        """Return the weighted coordinate ring structure."""
        return {
            'variables': self.variables,
            'relations': [],
            'grading': self.weights
        }
    
    def create_projective_coordinates(self, base_coords: np.ndarray) -> np.ndarray:
        """Create weighted projective coordinates over base points."""
        # Each fiber is weighted projective space
        coords = []
        for point in base_coords:
            fiber_coords = np.random.randn(len(self.weights))
            # Normalize according to weights
            norm = np.sum(fiber_coords ** np.array(self.weights))
            if norm > 0:
                fiber_coords = fiber_coords / (norm ** (1/np.mean(self.weights)))
            coords.append(fiber_coords)
        return np.array(coords)

class BirationalDoubleCoverModel:
    """
    Implementation of the birational double cover model from Section 4.
    Represents X_bar -> W with branch divisor B = B0 + sum(Gamma_i).
    """
    
    def __init__(self, surface_data: EllipticSurfaceData):
        self.data = surface_data
        self._build_ruled_surface()
        self._build_branch_divisor()
        self._build_cover()
    
    def _build_ruled_surface(self):
        """Build the ruled surface W = P(Epsilon_1)."""
        # Use NetworkX to represent the fiber structure
        self.ruled_surface = nx.Graph()
        
        # Add base curve vertices
        for i in range(self.data.base_curve_genus * 2 + 1):
            self.ruled_surface.add_node(f'base_{i}', type='base')
        
        # Add fiber vertices
        for i in range(len(self.data.double_fiber_types)):
            fiber_type = self.data.double_fiber_types[i]
            self.ruled_surface.add_node(f'fiber_{i}', type='fiber', kodaira=fiber_type)
            
            # Connect fiber to base points
            for j in range(2):
                self.ruled_surface.add_edge(f'fiber_{i}', f'base_{j}', weight=1)
    
    def _build_branch_divisor(self):
        """Build the branch divisor B = B0 + sum(Gamma_i)."""
        self.branch_divisor = {
            'b0': self._build_b0(),
            'gamma_fibers': self._build_gamma_fibers(),
            'singularities': self._build_singularities()
        }
    
    def _build_b0(self) -> Dict:
        """Build the B0 component of the branch divisor."""
        # B0 has An+3 singularities at intersection with Gamma_i
        b0 = {
            'components': [],
            'intersections': []
        }
        
        for i, (fiber_type, singularity) in enumerate(zip(
            self.data.double_fiber_types,
            self.data.singularities
        )):
            n = singularity.get('n', 0)
            # An+3 singularity at intersection point
            b0['components'].append({
                'fiber_index': i,
                'singularity': f'A_{n+3}',
                'intersection_multiplicity': 4
            })
            b0['intersections'].append({
                'fiber': i,
                'multiplicity': 4,
                'local_type': f'J_{{2,{n}}}'
            })
        
        return b0
    
    def _build_gamma_fibers(self) -> List[Dict]:
        """Build the Gamma_i fiber components in the branch divisor."""
        gamma_fibers = []
        for i, fiber_type in enumerate(self.data.double_fiber_types):
            gamma_fibers.append({
                'index': i,
                'type': fiber_type,
                'intersection_with_b0': 4,
                'elliptic_singularity_type': self.data.singularities[i].get('type', 'T_2,3,6')
            })
        return gamma_fibers
    
    def _build_singularities(self) -> List[Dict]:
        """Build singularity data for the cover."""
        singularities = []
        for i, singularity in enumerate(self.data.singularities):
            n = singularity.get('n', 0)
            singularities.append({
                'index': i,
                'arnold_type': f'J_{{2,{n}}}',
                't_type': f'T_{{2,3,{6+n}}}' if n >= 0 else 'T_2,3,6',
                'resolution_curves': self._get_resolution_curves(n)
            })
        return singularities
    
    def _get_resolution_curves(self, n: int) -> List[Dict]:
        """Get resolution data for J_{2,n} singularity."""
        curves = []
        # Exceptional divisor E1
        curves.append({'index': 0, 'self_intersection': -1, 'type': 'exceptional'})
        # Exceptional divisor E2
        curves.append({'index': 1, 'self_intersection': -1, 'type': 'exceptional'})
        # Additional curves for An-1 resolution if n >= 2
        for i in range(max(0, n-1)):
            curves.append({
                'index': i+2,
                'self_intersection': -2 if i < n-2 else -3,
                'type': 'resolution'
            })
        return curves
    
    def _build_cover(self):
        """Build the double cover X_bar."""
        self.cover_data = {
            'branch_divisor': self.branch_divisor,
            'line_bundle': self._compute_line_bundle(),
            'singularities': self.data.singularities,
            'deck_involution': self._build_deck_involution()
        }
    
    def _compute_line_bundle(self) -> LineBundle:
        """Compute M = alpha^* N(2)."""
        # From Lemma 4.6
        n = self.data.epsilon_2_neg.inverse()
        # M = alpha^* N(2) where N = epsilon_2_neg^{-1}(sum c_i)
        return LineBundle(
            degree=-self.data.epsilon_2_neg.degree + len(self.data.double_fiber_types),
            is_torsion=self.data.epsilon_2_neg.is_torsion
        )
    
    def _build_deck_involution(self) -> Dict:
        """Build the deck involution on the cover."""
        return {
            'fixed_locus': self.branch_divisor,
            'action_on_coordinates': self._coordinate_action(),
            'fixed_points': self._compute_fixed_points()
        }
    
    def _coordinate_action(self) -> Dict:
        """Compute the action of the involution on coordinates."""
        # For a double cover, involution sends y -> -y
        return {
            'branched_coordinate': {'sign': -1},
            'unbranched_coordinates': {'sign': 1}
        }
    
    def _compute_fixed_points(self) -> List[Tuple[float, float]]:
        """Compute fixed points of the involution (branch points)."""
        fixed_points = []
        # Branch points over Gamma_i fibers
        for i in range(len(self.data.double_fiber_types)):
            # There are 4 branch points on each Gamma_i
            for j in range(4):
                fixed_points.append((float(i), float(j)))
        return fixed_points
    
    def visualize(self):
        """Visualize the birational double cover model."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Visualize ruled surface with branch divisor
        ax1 = axes[0]
        ax1.set_title('Ruled Surface W = P(E_1) with Branch Divisor')
        
        # Draw base curve
        base_points = np.linspace(0, 1, 20)
        ax1.plot(base_points, np.zeros_like(base_points), 'k-', linewidth=2, label='Base C')
        
        # Draw fibers
        for i in range(len(self.data.double_fiber_types)):
            x = (i + 0.5) / len(self.data.double_fiber_types)
            y = np.linspace(-0.5, 0.5, 20)
            ax1.plot([x]*len(y), y, 'b--', alpha=0.7, label=f'Γ_{i}' if i==0 else "")
        
        # Mark branch points
        for point in self._compute_fixed_points():
            x = point[0] / len(self.data.double_fiber_types)
            y = np.random.uniform(-0.4, 0.4)
            ax1.plot(x, y, 'ro', markersize=5)
        
        ax1.set_xlabel('Base Curve C')
        ax1.set_ylabel('Fiber')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Visualize cover X_bar
        ax2 = axes[1]
        ax2.set_title('Double Cover X_bar → W')
        
        # Draw branch divisor
        for i, gamma in enumerate(self.branch_divisor['gamma_fibers']):
            x = (i + 0.5) / len(self.data.double_fiber_types)
            y = np.linspace(-0.5, 0.5, 20)
            ax2.plot([x]*len(y), y, 'b--', alpha=0.5)
        
        # Draw B0
        x_b0 = np.linspace(0, 1, 20)
        y_b0 = 0.3 * np.sin(2 * np.pi * x_b0)
        ax2.plot(x_b0, y_b0, 'r-', linewidth=2, label='B0')
        
        # Mark singularities
        for i, singularity in enumerate(self.data.singularities):
            x = (i + 0.5) / len(self.data.double_fiber_types)
            ax2.plot(x, 0, 'ks', markersize=8, label=f'J_{{2,{singularity.get("n",0)}}}')
        
        ax2.set_xlabel('Base Curve C')
        ax2.set_ylabel('Fiber')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

class DeterminantalModel:
    """
    Implementation of the relative determinantal model from Sections 5-6.
    Embeds X in weighted projective bundle P(1,1,2,2,?).
    """
    
    def __init__(self, surface_data: EllipticSurfaceData):
        self.data = surface_data
        self._build_weighted_bundle()
        self._build_equations()
    
    def _build_weighted_bundle(self):
        """Build the weighted projective bundle for the determinantal model."""
        # From Theorem 6.3: embedding in P(1,1,2) bundle
        self.weights = [1, 1, 2]
        self.base = self._build_base_curve()
        self.bundle = WeightedProjectiveBundle(self.weights, self.base)
        
        # Add splitting condition for global equations
        self.splitting_condition = self._check_splitting()
    
    def _build_base_curve(self) -> 'Curve':
        """Build the base curve C."""
        return Curve(self.data.base_curve_genus)
    
    def _check_splitting(self) -> bool:
        """Check if the splitting condition for global equations holds."""
        # From Section 6: need split vector bundle condition
        # E_2 = f_* L^2 should split as direct sum of line bundles
        # This is true for P^1 base or with additional assumptions
        return self.data.base_curve_genus == 0
    
    def _build_equations(self):
        """Build the determinantal equations defining X."""
        # The relative section ring has generators of degrees 1 and 2
        # with relations encoded by 2x2 determinants
        
        # Get generators
        gen1 = self._get_degree_1_generators()
        gen2 = self._get_degree_2_generators()
        
        # Build determinantal relations
        self.equations = self._build_determinantal_relations(gen1, gen2)
        
        # Add double fiber conditions
        self.double_fiber_equations = self._build_double_fiber_conditions()
    
    def _get_degree_1_generators(self) -> List[Dict]:
        """Get degree 1 generators (from Epsilon_1)."""
        # These correspond to variables of weight 1
        generators = []
        for i in range(self.data.epsilon_1.degree + 1):
            generators.append({
                'name': f'x_{i}',
                'degree': 1,
                'weight': 1,
                'line_bundle': self.data.epsilon_1
            })
        return generators
    
    def _get_degree_2_generators(self) -> List[Dict]:
        """Get degree 2 generators (anti-invariant part)."""
        # These correspond to variables of weight 2
        generators = []
        # From Epsilon_2^- (anti-invariant sections)
        for i in range(self.data.epsilon_2_neg.degree + 1):
            generators.append({
                'name': f'y_{i}',
                'degree': 2,
                'weight': 2,
                'line_bundle': self.data.epsilon_2_neg
            })
        return generators
    
    def _build_determinantal_relations(self, gen1: List, gen2: List) -> List[Dict]:
        """Build relations using 2x2 determinants."""
        relations = []
        
        # Generic determinantal relations from the section ring
        # These are equations of the form: x_i x_j = something
        for i in range(len(gen1)):
            for j in range(i+1, len(gen1)):
                # 2x2 determinant relations
                relations.append({
                    'type': 'determinant',
                    'degree': 2,
                    'variables': [gen1[i]['name'], gen1[j]['name']],
                    'result': self._compute_determinant_result(i, j)
                })
        
        return relations
    
    def _compute_determinant_result(self, i: int, j: int) -> Dict:
        """Compute the result of a 2x2 determinant relation."""
        # In practice, this would use the actual equations from the paper
        # Here we return a symbolic representation
        return {
            'type': 'linear_combination',
            'terms': [
                {'coefficient': 1, 'variable': f'y_{i+j}'},
                {'coefficient': -1, 'variable': f'x_{i}*x_{j}'}
            ]
        }
    
    def _build_double_fiber_conditions(self) -> List[Dict]:
        """Build additional equations for double fibers."""
        conditions = []
        for i, fiber_type in enumerate(self.data.double_fiber_types):
            # Over each double fiber, the branch divisor has a J_{2,n} singularity
            # This imposes specific local equations
            n = self.data.singularities[i].get('n', 0)
            conditions.append({
                'fiber_index': i,
                'type': f'J_{{2,{n}}}',
                'local_equation': self._get_local_equation(n)
            })
        return conditions
    
    def _get_local_equation(self, n: int) -> str:
        """Get local equation for J_{2,n} singularity."""
        if n == 0:
            return "z^2 = x^3 + y^6"  # T_{2,3,6}
        elif n == 1:
            return "z^2 = x^3 + x*y^4 + y^7"  # T_{2,3,7}
        else:
            return f"z^2 = x^3 + y^6 * (1 + y^{n-1} + ...)"  # T_{2,3,6+n}
    
    def visualize(self):
        """Visualize the determinantal model."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Visualize weighted projective bundle
        ax1 = axes[0]
        ax1.set_title('Weighted Projective Bundle P(1,1,2)')
        
        # Show fiber structure
        weights = [1, 1, 2]
        for w in weights:
            x = np.linspace(-1, 1, 100)
            y = np.linspace(-1, 1, 100)
            X, Y = np.meshgrid(x, y)
            Z = (X**2 + Y**2) ** (1/w)
            ax1.contour(X, Y, Z, levels=5, alpha=0.5)
        
        ax1.set_xlabel('Coordinate 1')
        ax1.set_ylabel('Coordinate 2')
        ax1.grid(True, alpha=0.3)
        
        # Visualize equations and singularities
        ax2 = axes[1]
        ax2.set_title('Determinantal Equations')
        
        # Plot the determinantal equations as surfaces
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, y)
        
        # Plot some sample equations
        for i in range(min(3, len(self.equations))):
            Z = X**2 + Y**2 - 0.5  # Simplified determinant equation
            ax2.contour(X, Y, Z, levels=[0], alpha=0.5)
        
        # Mark singularities from double fibers
        for i, condition in enumerate(self.double_fiber_equations):
            x_pos = (i + 0.5) / len(self.double_fiber_equations)
            ax2.plot(x_pos, 0, 'ks', markersize=8, 
                    label=f'J_{{2,{condition.get("n",0)}}}')
        
        ax2.set_xlabel('Base Curve Parameter')
        ax2.set_ylabel('Fiber Coordinate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

class Curve:
    """Simple representation of a base curve C."""
    
    def __init__(self, genus: int):
        self.genus = genus
        self.points = self._generate_points()
    
    def _generate_points(self):
        """Generate points on the curve."""
        # For simplicity, use rational points
        return np.linspace(0, 1, max(20, 2*self.genus + 1))

class EllipticSurfaceConstructor:
    """
    Main constructor for marked elliptic surfaces using the models from the paper.
    """
    
    @staticmethod
    def from_halphen_index_2() -> EllipticSurfaceData:
        """
        Construct a Halphen surface of index 2.
        Example from Section 7.C.
        """
        # Halphen surface of index 2 has:
        # - Base curve P^1
        # - 9 double fibers (all of type I0)
        # - epsilon_1 = O(-1) on P^1
        # - epsilon_2_neg = O(4) on P^1
        
        data = EllipticSurfaceData(
            base_curve_genus=0,
            num_double_fibers=9,
            double_fiber_types=['I0'] * 9,
            epsilon_1=LineBundle(degree=-1),
            epsilon_2_neg=LineBundle(degree=4),
            det_epsilon_1=LineBundle(degree=-1),
            branch_divisor=None,  # Will be built later
            singularities=[{'n': 0, 'type': 'T_2,3,6'}] * 9
        )
        
        # Build branch divisor
        data.branch_divisor = EllipticSurfaceConstructor._build_branch_divisor(data)
        
        return data
    
    @staticmethod
    def from_enriques_surface(special: bool = False) -> EllipticSurfaceData:
        """
        Construct an Enriques surface.
        Example from Section 8.
        
        Args:
            special: If True, construct a special Enriques surface (Section 8.B),
                    otherwise a non-special Enriques surface (Section 8.C).
        """
        if special:
            # Special Enriques surface: Section 8.B
            # Has a double fiber with n=1 singularity
            data = EllipticSurfaceData(
                base_curve_genus=1,
                num_double_fibers=1,
                double_fiber_types=['I0'],
                epsilon_1=LineBundle(degree=-1, is_torsion=True, torsion_order=2),
                epsilon_2_neg=LineBundle(degree=2, is_torsion=False),
                det_epsilon_1=LineBundle(degree=-1, is_torsion=True, torsion_order=2),
                branch_divisor=None,
                singularities=[{'n': 1, 'type': 'T_2,3,7'}]
            )
        else:
            # Non-special Enriques surface: Section 8.C
            data = EllipticSurfaceData(
                base_curve_genus=1,
                num_double_fibers=2,
                double_fiber_types=['I0', 'I0'],
                epsilon_1=LineBundle(degree=-1, is_torsion=True, torsion_order=2),
                epsilon_2_neg=LineBundle(degree=2, is_torsion=False),
                det_epsilon_1=LineBundle(degree=-1, is_torsion=True, torsion_order=2),
                branch_divisor=None,
                singularities=[{'n': 0, 'type': 'T_2,3,6'}, {'n': 0, 'type': 'T_2,3,6'}]
            )
        
        data.branch_divisor = EllipticSurfaceConstructor._build_branch_divisor(data)
        return data
    
    @staticmethod
    def _build_branch_divisor(data: EllipticSurfaceData) -> BranchDivisor:
        """Build branch divisor for the given surface data."""
        b0_components = []
        gamma_fibers = []
        
        for i, singularity in enumerate(data.singularities):
            n = singularity.get('n', 0)
            b0_components.append({
                'fiber_index': i,
                'singularity': f'A_{n+3}',
                'intersection_multiplicity': 4
            })
            gamma_fibers.append({
                'index': i,
                'type': data.double_fiber_types[i],
                'intersection_with_b0': 4
            })
        
        return BranchDivisor(
            b0=b0_components,
            gamma_fibers=gamma_fibers,
            local_singularities=[(i, s.get('n', 0)) for i, s in enumerate(data.singularities)]
        )
    
    @staticmethod
    def construct_models(data: EllipticSurfaceData) -> Tuple[BirationalDoubleCoverModel, DeterminantalModel]:
        """Construct both birational models for the given surface data."""
        double_cover = BirationalDoubleCoverModel(data)
        determinantal = DeterminantalModel(data)
        return double_cover, determinantal

def example_halphen_surface():
    """Example: Construct Halphen surface of index 2."""
    print("Constructing Halphen surface of index 2...")
    data = EllipticSurfaceConstructor.from_halphen_index_2()
    
    # Build models
    double_cover, determinantal = EllipticSurfaceConstructor.construct_models(data)
    
    # Visualize
    fig1 = double_cover.visualize()
    fig2 = determinantal.visualize()
    
    plt.show()
    
    return data, double_cover, determinantal

def example_enriques_surfaces():
    """Example: Construct special and non-special Enriques surfaces."""
    print("Constructing Enriques surfaces...")
    
    # Special Enriques surface
    print("  Special Enriques surface (Section 8.B):")
    special_data = EllipticSurfaceConstructor.from_enriques_surface(special=True)
    special_cover, special_det = EllipticSurfaceConstructor.construct_models(special_data)
    
    # Non-special Enriques surface
    print("  Non-special Enriques surface (Section 8.C):")
    nonspecial_data = EllipticSurfaceConstructor.from_enriques_surface(special=False)
    nonspecial_cover, nonspecial_det = EllipticSurfaceConstructor.construct_models(nonspecial_data)
    
    # Visualize special
    fig1 = special_cover.visualize()
    fig2 = special_det.visualize()
    
    # Visualize non-special
    fig3 = nonspecial_cover.visualize()
    fig4 = nonspecial_det.visualize()
    
    plt.show()
    
    return (special_data, special_cover, special_det), (nonspecial_data, nonspecial_cover, nonspecial_det)

# Run examples
if __name__ == "__main__":
    print("=" * 60)
    print("Marked Elliptic Surfaces with Double Fibers")
    print("Based on arXiv:2608.19970v1")
    print("=" * 60)
    
    # Example 1: Halphen surface of index 2
    data1, cover1, det1 = example_halphen_surface()
    
    # Example 2: Enriques surfaces
    special, nonspecial = example_enriques_surfaces()
    
    print("\n" + "=" * 60)
    print("Construction complete!")
    print("=" * 60)
    print("\nKey invariants:")
    print(f"  Halphen index 2: 9 double fibers of type I0")
    print(f"  Special Enriques: 1 double fiber with J_2,1 singularity")
    print(f"  Non-special Enriques: 2 double fibers with J_2,0 singularities")
    
    # Print some details
    print("\nBranch divisor details for Halphen surface:")
    print(f"  B0 component with {len(data1.branch_divisor.b0)} singularities")
    print(f"  {len(data1.branch_divisor.gamma_fibers)} Gamma fibers")
    for i, (b0, gamma) in enumerate(zip(data1.branch_divisor.b0, data1.branch_divisor.gamma_fibers)):
        print(f"    Fiber {i}: {b0['singularity']} singularity, intersection multiplicity {b0['intersection_multiplicity']}")
    
    print("\nDeterminantal model details:")
    print(f"  Splitting condition satisfied: {det1.splitting_condition}")
    print(f"  Number of equations: {len(det1.equations)}")
    print(f"  Double fiber conditions: {len(det1.double_fiber_equations)}")