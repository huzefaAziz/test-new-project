import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from collections import Counter
import warnings
import matplotlib.gridspec as gridspec

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PXPModel:
    """
    Implementation of the PXP model for quantum many-body scars.
    """
    
    def __init__(self, N, periodic=True):
        self.N = N
        self.periodic = periodic
        self.hilbert_space = self._build_hilbert_space()
        self.hamiltonian = self._build_hamiltonian()
        self.state_to_idx = {s: i for i, s in enumerate(self.hilbert_space)}
        
    def _build_hilbert_space(self):
        states = []
        for i in range(2**self.N):
            bits = format(i, f'0{self.N}b')
            if '11' not in bits:
                if self.periodic and bits[0] == '1' and bits[-1] == '1':
                    continue
                states.append(bits)
        return states
    
    def _build_hamiltonian(self):
        n_states = len(self.hilbert_space)
        H = np.zeros((n_states, n_states), dtype=complex)
        state_to_idx = {s: i for i, s in enumerate(self.hilbert_space)}
        
        for i, state in enumerate(self.hilbert_space):
            for site in range(self.N):
                left = (site - 1) % self.N if self.periodic else site - 1
                right = (site + 1) % self.N if self.periodic else site + 1
                
                if self.periodic:
                    can_flip = (state[left] == '0' and state[right] == '0')
                else:
                    can_flip = (left < 0 or state[left] == '0') and (right >= self.N or state[right] == '0')
                
                if can_flip and state[site] == '0':
                    new_state = list(state)
                    new_state[site] = '1'
                    new_state = ''.join(new_state)
                    if new_state in state_to_idx:
                        j = state_to_idx[new_state]
                        H[i, j] = 1.0
                        H[j, i] = 1.0
        
        return H
    
    def get_initial_state(self, state_type):
        if state_type == 'Z2':
            bits = ''.join('1' if i % 2 == 0 else '0' for i in range(self.N))
        elif state_type == 'Z2p':
            bits = ''.join('1' if i % 2 == 1 else '0' for i in range(self.N))
        elif state_type == 'Z3':
            bits = ''.join('1' if i % 3 == 0 else '0' for i in range(self.N))
        elif state_type == '0':
            bits = '0' * self.N
        elif state_type == 'random':
            bits = self._generate_random_valid_state()
        else:
            raise ValueError(f"Unknown state type: {state_type}")
        
        if '11' in bits:
            bits = self._fix_state(bits)
        
        return bits
    
    def _generate_random_valid_state(self):
        bits = ['0'] * self.N
        indices = list(range(self.N))
        np.random.shuffle(indices)
        
        for idx in indices:
            left = (idx - 1) % self.N if self.periodic else idx - 1
            right = (idx + 1) % self.N if self.periodic else idx + 1
            
            can_flip = True
            if self.periodic:
                if left >= 0 and bits[left] == '1':
                    can_flip = False
                if right < self.N and bits[right] == '1':
                    can_flip = False
            else:
                if left >= 0 and bits[left] == '1':
                    can_flip = False
                if right < self.N and bits[right] == '1':
                    can_flip = False
            
            if can_flip and np.random.random() > 0.5:
                bits[idx] = '1'
        
        return ''.join(bits)
    
    def _fix_state(self, state):
        bits = list(state)
        for i in range(len(bits) - 1):
            if bits[i] == '1' and bits[i+1] == '1':
                bits[i] = '0'
        if self.periodic and bits[0] == '1' and bits[-1] == '1':
            bits[0] = '0'
        return ''.join(bits)
    
    def is_valid_state(self, state):
        if '11' in state:
            return False
        if self.periodic and state[0] == '1' and state[-1] == '1':
            return False
        return True
    
    def evolve_state(self, initial_state, t, method='expm'):
        if isinstance(t, (int, float)):
            t = np.array([t])
        
        if not self.is_valid_state(initial_state):
            raise ValueError(f"Invalid state: {initial_state}")
        
        if initial_state not in self.state_to_idx:
            raise ValueError(f"State '{initial_state}' not in Hilbert space")
        
        from scipy.linalg import expm
        idx = self.state_to_idx[initial_state]
        psi0 = np.zeros(len(self.hilbert_space), dtype=complex)
        psi0[idx] = 1.0
        
        states = []
        for time in t:
            U = expm(-1j * self.hamiltonian * time)
            psi = U @ psi0
            states.append(psi)
        
        return np.array(states)
    
    def measure_probabilities(self, psi):
        return np.abs(psi)**2
    
    def sample_bitstrings(self, psi, n_samples):
        probs = self.measure_probabilities(psi)
        indices = np.random.choice(len(self.hilbert_space), size=n_samples, p=probs)
        return [self.hilbert_space[i] for i in indices]

class MultidimensionalScaling:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.embedding = None
        
    def fit_transform(self, distance_matrix):
        n = len(distance_matrix)
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ distance_matrix**2 @ J
        
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        self.embedding = eigvecs[:, :self.n_components] * np.sqrt(np.maximum(eigvals[:self.n_components], 0))
        return self.embedding

class ProbabilisticEarthMoverDistance:
    def __init__(self, pxp_model):
        self.model = pxp_model
        self.hamming_distances = self._compute_hamming_distances()
        
    def _compute_hamming_distances(self):
        n_states = len(self.model.hilbert_space)
        distances = np.zeros((n_states, n_states))
        
        for i, state_i in enumerate(self.model.hilbert_space):
            for j, state_j in enumerate(self.model.hilbert_space):
                distances[i, j] = sum(a != b for a, b in zip(state_i, state_j))
        
        return distances
    
    def compute_distance(self, psi_A, psi_B):
        probs_A = np.abs(psi_A)**2
        probs_B = np.abs(psi_B)**2
        
        from scipy.optimize import linear_sum_assignment
        
        n = len(probs_A)
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cost[i, j] = self.hamming_distances[i, j]
        
        row_ind, col_ind = linear_sum_assignment(cost)
        distance = np.sum(cost[row_ind, col_ind] * 
                         np.minimum(probs_A[row_ind], probs_B[col_ind]))
        
        return distance

class IntrinsicDimensionEstimator:
    def __init__(self, t1=2, t2=3):
        self.t1 = t1
        self.t2 = t2
        
    def _volume(self, d, t):
        from scipy.special import comb
        
        if t == 0:
            return 1
        
        if t == 1:
            return 1 + 2*d
        elif t == 2:
            return 1 + 2*d + 2*d*(d-1)
        else:
            volume = 0
            for k in range(t + 1):
                if k <= d:
                    volume += comb(d, k) * comb(t, k)
            return volume
    
    def estimate_id(self, data, max_dim=10):
        if len(data) < 2:
            return 0
        
        n_points = len(data)
        
        distances = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                distances[i, j] = sum(a != b for a, b in zip(data[i], data[j]))
        
        def count_neighbors(point_idx, t):
            return np.sum(distances[point_idx] <= t) - 1
        
        avg_n = np.mean([count_neighbors(i, self.t1) for i in range(n_points)])
        avg_k = np.mean([count_neighbors(i, self.t2) for i in range(n_points)])
        
        if avg_k == 0:
            return 0
        
        def objective(d):
            ratio = self._volume(d, self.t1) / self._volume(d, self.t2)
            return abs(ratio - avg_n / avg_k)
        
        dims = np.arange(1, max_dim + 1)
        errors = [objective(d) for d in dims]
        
        return int(dims[np.argmin(errors)])

class ScarDetection:
    def __init__(self, N=12, n_timesteps=20, n_samples=100):
        self.model = PXPModel(N)
        self.n_timesteps = n_timesteps
        self.n_samples = n_samples
        
    def analyze_initial_state(self, initial_state, max_time=10.0):
        times = np.linspace(0, max_time, self.n_timesteps)
        
        try:
            states = self.model.evolve_state(initial_state, times, method='expm')
        except ValueError as e:
            return {
                'estimated_id': None,
                'embedding': None,
                'times': times,
                'states': None,
                'error': str(e)
            }
        
        samples = []
        for psi in states:
            bits = self.model.sample_bitstrings(psi, self.n_samples)
            samples.extend(bits)
        
        id_estimator = IntrinsicDimensionEstimator()
        estimated_id = id_estimator.estimate_id(samples)
        
        distance_matrix = self._compute_distance_matrix(states)
        mds = MultidimensionalScaling(n_components=2)
        embedding = mds.fit_transform(distance_matrix)
        
        return {
            'estimated_id': estimated_id,
            'embedding': embedding,
            'times': times,
            'states': states
        }
    
    def _compute_distance_matrix(self, states):
        pem = ProbabilisticEarthMoverDistance(self.model)
        n_states = len(states)
        distances = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            for j in range(i+1, n_states):
                dist = pem.compute_distance(states[i], states[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def compare_initial_states(self, initial_states, max_time=10.0):
        results = {}
        valid_results = []
        
        for state in initial_states:
            print(f"Analyzing state: {state}")
            results[state] = self.analyze_initial_state(state, max_time)
            if results[state]['estimated_id'] is not None:
                valid_results.append(results[state]['estimated_id'])
        
        if valid_results:
            mean_id = np.mean(valid_results)
            std_id = np.std(valid_results)
            threshold = mean_id - 1.5 * std_id
            
            for state in initial_states:
                if results[state]['estimated_id'] is not None:
                    results[state]['is_scar'] = results[state]['estimated_id'] < threshold
                else:
                    results[state]['is_scar'] = False
        
        return results

def plot_mds_embedding(results, title="MDS Embedding of PXP Dynamics"):
    """
    Plot MDS embeddings for different initial states.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Define colors for different states
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for idx, (state, result) in enumerate(results.items()):
        ax = axes[idx % 4]
        
        if result['embedding'] is not None and result['embedding'].shape[0] > 0:
            embedding = result['embedding']
            
            # Plot trajectory with color gradient based on time
            n_points = embedding.shape[0]
            for i in range(n_points - 1):
                ax.plot(embedding[i:i+2, 0], embedding[i:i+2, 1], 
                       color=colors[idx % len(colors)], 
                       alpha=0.5 + 0.5 * (i / n_points),
                       linewidth=1.5)
            
            # Mark starting point
            ax.scatter(embedding[0, 0], embedding[0, 1], 
                      color='green', s=100, marker='*', 
                      label='Start', zorder=5)
            
            # Mark ending point
            ax.scatter(embedding[-1, 0], embedding[-1, 1], 
                      color='red', s=80, marker='o', 
                      label='End', zorder=5)
            
            # Plot all points
            ax.scatter(embedding[:, 0], embedding[:, 1], 
                      c=np.linspace(0, 1, n_points), 
                      cmap='viridis', s=20, alpha=0.6, zorder=2)
        
        scar_label = " (SCAR)" if result.get('is_scar', False) else ""
        ax.set_title(f"State: {state}{scar_label}")
        ax.set_xlabel("MDS Component 1")
        ax.set_ylabel("MDS Component 2")
        ax.grid(True, alpha=0.3)
        
        # Add info box
        info_text = f"ID: {result['estimated_id']:.2f}" if result['estimated_id'] is not None else "Error"
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_intrinsic_dimension_comparison(results, title="Intrinsic Dimension Comparison"):
    """
    Plot boxplot comparing intrinsic dimensions of different initial states.
    """
    # Prepare data
    states = []
    id_values = []
    is_scar = []
    
    for state, result in results.items():
        if result['estimated_id'] is not None:
            states.append(state)
            id_values.append(result['estimated_id'])
            is_scar.append(result.get('is_scar', False))
    
    if len(id_values) == 0:
        print("No valid ID values to plot")
        return None
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot of ID values
    data_to_plot = [id_values]
    bp = ax1.boxplot(data_to_plot, patch_artist=True)
    
    # Color the box
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    
    # Add individual points
    for i, (state, val, scar) in enumerate(zip(states, id_values, is_scar)):
        color = 'red' if scar else 'blue'
        marker = '*' if scar else 'o'
        size = 100 if scar else 50
        ax1.scatter(np.random.normal(1, 0.04), val, 
                   color=color, marker=marker, s=size, alpha=0.7, zorder=3)
        
        # Add state label for scar states
        if scar:
            ax1.annotate(state, (1, val), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='red')
    
    # Add threshold line
    if id_values:
        mean_id = np.mean(id_values)
        std_id = np.std(id_values)
        threshold = mean_id - 1.5 * std_id
        ax1.axhline(y=threshold, color='red', linestyle='--', 
                   label=f'Scar Threshold: {threshold:.2f}')
        ax1.axhline(y=mean_id, color='black', linestyle=':', 
                   label=f'Mean ID: {mean_id:.2f}')
    
    ax1.set_xticklabels(['All States'])
    ax1.set_ylabel('Estimated Intrinsic Dimension')
    ax1.set_title('Intrinsic Dimension Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar plot showing ID for each state
    x_pos = np.arange(len(states))
    colors_bar = ['red' if s else 'blue' for s in is_scar]
    bars = ax2.bar(x_pos, id_values, color=colors_bar, alpha=0.7)
    
    # Add threshold line on bar plot
    if id_values:
        ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(states, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Estimated Intrinsic Dimension')
    ax2.set_title('ID by Initial State')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Scar State'),
                      Patch(facecolor='blue', alpha=0.7, label='Thermal State')]
    ax2.legend(handles=legend_elements)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_scar_detection_pipeline(results, N=8):
    """
    Create a comprehensive figure showing the scar detection pipeline.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Hilbert space graph (schematic)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.5, "PXP Hilbert Space\nConstrained Graph", 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    # Create a small schematic graph
    G = nx.Graph()
    nodes = ['000', '001', '010', '100', '101']
    edges = [('000', '001'), ('000', '010'), ('000', '100'), 
             ('001', '101'), ('010', '101'), ('100', '101')]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    pos = {'000': (0, 0), '001': (-1, 1), '010': (0, 1), 
           '100': (1, 1), '101': (0, 2)}
    nx.draw(G, pos, ax=ax1, node_color='lightblue', node_size=500, 
            with_labels=True, font_size=8, font_weight='bold')
    ax1.set_title("Hilbert Space Graph (Schematic)")
    
    # Plot 2: MDS embedding for a scar state
    ax2 = fig.add_subplot(gs[0, 1])
    scar_state = None
    for state, result in results.items():
        if result.get('is_scar', False) and result['embedding'] is not None:
            scar_state = state
            embedding = result['embedding']
            n_points = embedding.shape[0]
            
            # Plot trajectory
            for i in range(n_points - 1):
                ax2.plot(embedding[i:i+2, 0], embedding[i:i+2, 1], 
                        color='blue', alpha=0.3 + 0.7 * (i / n_points),
                        linewidth=1.5)
            ax2.scatter(embedding[:, 0], embedding[:, 1], 
                       c=np.linspace(0, 1, n_points), 
                       cmap='viridis', s=30, alpha=0.6)
            ax2.scatter(embedding[0, 0], embedding[0, 1], 
                       color='green', s=100, marker='*', label='Start')
            ax2.scatter(embedding[-1, 0], embedding[-1, 1], 
                       color='red', s=80, marker='o', label='End')
            break
    
    ax2.set_title(f"Scar State MDS ({scar_state})" if scar_state else "MDS Embedding")
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MDS embedding for a thermal state
    ax3 = fig.add_subplot(gs[0, 2])
    thermal_state = None
    for state, result in results.items():
        if not result.get('is_scar', True) and result['embedding'] is not None:
            thermal_state = state
            embedding = result['embedding']
            n_points = embedding.shape[0]
            
            # Plot trajectory
            for i in range(n_points - 1):
                ax3.plot(embedding[i:i+2, 0], embedding[i:i+2, 1], 
                        color='red', alpha=0.3 + 0.7 * (i / n_points),
                        linewidth=1.5)
            ax3.scatter(embedding[:, 0], embedding[:, 1], 
                       c=np.linspace(0, 1, n_points), 
                       cmap='viridis', s=30, alpha=0.6)
            ax3.scatter(embedding[0, 0], embedding[0, 1], 
                       color='green', s=100, marker='*', label='Start')
            ax3.scatter(embedding[-1, 0], embedding[-1, 1], 
                       color='red', s=80, marker='o', label='End')
            break
    
    ax3.set_title(f"Thermal State MDS ({thermal_state})" if thermal_state else "MDS Embedding")
    ax3.set_xlabel("Component 1")
    ax3.set_ylabel("Component 2")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Intrinsic dimension comparison
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Prepare data for ID comparison
    states = []
    id_values = []
    is_scar = []
    state_labels = []
    
    for state, result in results.items():
        if result['estimated_id'] is not None:
            # Shorten state labels for display
            if len(state) > 6:
                label = state[:3] + '...' + state[-3:]
            else:
                label = state
            state_labels.append(label)
            states.append(state)
            id_values.append(result['estimated_id'])
            is_scar.append(result.get('is_scar', False))
    
    if id_values:
        x_pos = np.arange(len(states))
        colors_bar = ['red' if s else 'blue' for s in is_scar]
        bars = ax4.bar(x_pos, id_values, color=colors_bar, alpha=0.7, edgecolor='black')
        
        # Add threshold line
        mean_id = np.mean(id_values)
        std_id = np.std(id_values)
        threshold = mean_id - 1.5 * std_id
        ax4.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Scar Threshold: {threshold:.2f}')
        ax4.axhline(y=mean_id, color='black', linestyle=':', linewidth=2,
                   label=f'Mean ID: {mean_id:.2f}')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, id_values)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(state_labels, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('Estimated Intrinsic Dimension', fontsize=11)
        ax4.set_title('Intrinsic Dimension Comparison: Scar vs Thermal States', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add legend for bar colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Scar State'),
                          Patch(facecolor='blue', alpha=0.7, label='Thermal State')]
        ax4.legend(handles=legend_elements, loc='upper right')
    
    # Plot 5: Quantum revival dynamics
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Plot revival dynamics for scar state
    for state, result in results.items():
        if result.get('is_scar', False) and result['states'] is not None:
            states = result['states']
            times = result['times']
            
            # Compute overlap with initial state
            overlaps = [np.abs(np.vdot(states[0], s))**2 for s in states]
            ax5.plot(times, overlaps, 'b-', linewidth=2, label=f'Scar ({state})', alpha=0.8)
    
    # Plot thermal decay
    for state, result in results.items():
        if not result.get('is_scar', True) and result['states'] is not None:
            states = result['states']
            times = result['times']
            
            # Compute overlap with initial state
            overlaps = [np.abs(np.vdot(states[0], s))**2 for s in states]
            ax5.plot(times, overlaps, 'r-', linewidth=2, label=f'Thermal ({state})', alpha=0.8)
    
    ax5.set_xlabel('Time (t)', fontsize=11)
    ax5.set_ylabel('|⟨ψ(0)|ψ(t)⟩|²', fontsize=11)
    ax5.set_title('Quantum Revival Dynamics', fontsize=12)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.1])
    
    plt.suptitle("Quantum Many-Body Scar Detection Pipeline", fontsize=16, fontweight='bold', y=0.98)
    return fig

def plot_robustness_analysis(results, n_samples_list=[50, 100, 200, 500]):
    """
    Plot robustness analysis showing ID vs number of measurements.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get scar and thermal states
    scar_states = []
    thermal_states = []
    for state, result in results.items():
        if result.get('is_scar', False):
            scar_states.append((state, result))
        elif result['estimated_id'] is not None:
            thermal_states.append((state, result))
    
    # Plot 1: ID vs number of samples
    ax = axes[0]
    for state, result in scar_states[:2]:  # Plot up to 2 scar states
        n_samples_list = [20, 50, 100, 200]
        ids = []
        for n_samples in n_samples_list:
            # Re-analyze with different number of samples
            detector = ScarDetection(N=8, n_timesteps=15, n_samples=n_samples)
            new_result = detector.analyze_initial_state(state, max_time=5.0)
            if new_result['estimated_id'] is not None:
                ids.append(new_result['estimated_id'])
            else:
                ids.append(np.nan)
        ax.plot(n_samples_list, ids, 'o-', linewidth=2, label=f'Scar: {state}')
    
    for state, result in thermal_states[:2]:  # Plot up to 2 thermal states
        n_samples_list = [20, 50, 100, 200]
        ids = []
        for n_samples in n_samples_list:
            detector = ScarDetection(N=8, n_timesteps=15, n_samples=n_samples)
            new_result = detector.analyze_initial_state(state, max_time=5.0)
            if new_result['estimated_id'] is not None:
                ids.append(new_result['estimated_id'])
            else:
                ids.append(np.nan)
        ax.plot(n_samples_list, ids, 'o--', linewidth=2, label=f'Thermal: {state}')
    
    ax.set_xlabel('Number of Samples per Time Point', fontsize=11)
    ax.set_ylabel('Estimated Intrinsic Dimension', fontsize=11)
    ax.set_title('ID vs Number of Samples', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 2: ID vs number of timesteps
    ax = axes[1]
    for state, result in scar_states[:2]:
        n_timesteps_list = [5, 10, 15, 20]
        ids = []
        for n_timesteps in n_timesteps_list:
            detector = ScarDetection(N=8, n_timesteps=n_timesteps, n_samples=100)
            new_result = detector.analyze_initial_state(state, max_time=5.0)
            if new_result['estimated_id'] is not None:
                ids.append(new_result['estimated_id'])
            else:
                ids.append(np.nan)
        ax.plot(n_timesteps_list, ids, 'o-', linewidth=2, label=f'Scar: {state}')
    
    for state, result in thermal_states[:2]:
        n_timesteps_list = [5, 10, 15, 20]
        ids = []
        for n_timesteps in n_timesteps_list:
            detector = ScarDetection(N=8, n_timesteps=n_timesteps, n_samples=100)
            new_result = detector.analyze_initial_state(state, max_time=5.0)
            if new_result['estimated_id'] is not None:
                ids.append(new_result['estimated_id'])
            else:
                ids.append(np.nan)
        ax.plot(n_timesteps_list, ids, 'o--', linewidth=2, label=f'Thermal: {state}')
    
    ax.set_xlabel('Number of Time Points', fontsize=11)
    ax.set_ylabel('Estimated Intrinsic Dimension', fontsize=11)
    ax.set_title('ID vs Number of Time Points', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Histogram of ID values
    ax = axes[2]
    scar_ids = [r['estimated_id'] for s, r in scar_states if r['estimated_id'] is not None]
    thermal_ids = [r['estimated_id'] for s, r in thermal_states if r['estimated_id'] is not None]
    
    if scar_ids and thermal_ids:
        ax.hist(scar_ids, bins=10, alpha=0.7, label='Scar States', color='red', density=True)
        ax.hist(thermal_ids, bins=10, alpha=0.7, label='Thermal States', color='blue', density=True)
        ax.axvline(np.mean(scar_ids), color='red', linestyle='--', label=f'Scar Mean: {np.mean(scar_ids):.2f}')
        ax.axvline(np.mean(thermal_ids), color='blue', linestyle='--', label=f'Thermal Mean: {np.mean(thermal_ids):.2f}')
    
    ax.set_xlabel('Estimated Intrinsic Dimension', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('ID Distribution: Scar vs Thermal', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Robustness Analysis: Scar Detection under Limited Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_comprehensive_report(N=8, max_time=5.0, n_timesteps=20, n_samples=100):
    """
    Create a comprehensive analysis with all plots.
    """
    print("=" * 60)
    print("QUANTUM MANY-BODY SCAR DETECTION USING INTRINSIC DIMENSION")
    print("=" * 60)
    print(f"System size: N={N} sites")
    print(f"Hilbert space dimension: {len(PXPModel(N).hilbert_space)}")
    print(f"Time points: {n_timesteps}")
    print(f"Samples per time point: {n_samples}")
    print("=" * 60)
    
    # Initialize detector
    detector = ScarDetection(N=N, n_timesteps=n_timesteps, n_samples=n_samples)
    
    # Define initial states
    initial_states = [
        detector.model.get_initial_state('Z2'),   # Scar state (Néel)
        detector.model.get_initial_state('Z2p'),  # Scar state
        detector.model.get_initial_state('Z3'),   # Possible scar
        detector.model.get_initial_state('0'),    # Thermal state
    ]
    
    # Add random states
    for _ in range(4):
        random_state = detector.model.get_initial_state('random')
        if random_state not in initial_states:
            initial_states.append(random_state)
    
    print(f"\nAnalyzing {len(initial_states)} initial states...")
    
    # Run analysis
    results = detector.compare_initial_states(initial_states, max_time=max_time)
    
    # Create plots
    print("\nGenerating plots...")
    
    # Plot 1: MDS embedding
    fig1 = plot_mds_embedding(results, "MDS Embedding of PXP Dynamics")
    fig1.savefig('mds_embedding.png', dpi=300, bbox_inches='tight')
    print("  - Saved: mds_embedding.png")
    
    # Plot 2: ID comparison
    fig2 = plot_intrinsic_dimension_comparison(results, "Intrinsic Dimension Comparison")
    if fig2:
        fig2.savefig('id_comparison.png', dpi=300, bbox_inches='tight')
        print("  - Saved: id_comparison.png")
    
    # Plot 3: Pipeline overview
    fig3 = plot_scar_detection_pipeline(results, N=N)
    fig3.savefig('scar_detection_pipeline.png', dpi=300, bbox_inches='tight')
    print("  - Saved: scar_detection_pipeline.png")
    
    # Plot 4: Robustness analysis (requires re-running with different parameters)
    # This is a simplified version
    fig4 = plot_robustness_analysis(results)
    fig4.savefig('robustness_analysis.png', dpi=300, bbox_inches='tight')
    print("  - Saved: robustness_analysis.png")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    scar_states = [s for s, r in results.items() if r.get('is_scar', False)]
    thermal_states = [s for s, r in results.items() if not r.get('is_scar', True) and r['estimated_id'] is not None]
    
    print(f"\nScar states detected: {len(scar_states)}")
    for state in scar_states:
        print(f"  - {state}: ID = {results[state]['estimated_id']:.2f}")
    
    print(f"\nThermal states: {len(thermal_states)}")
    for state in thermal_states[:5]:  # Show first 5
        print(f"  - {state}: ID = {results[state]['estimated_id']:.2f}")
    
    if len(thermal_states) > 5:
        print(f"  ... and {len(thermal_states) - 5} more")
    
    print("\nPlots saved successfully!")
    return results

# Run the analysis
if __name__ == "__main__":
    results = create_comprehensive_report(N=8, max_time=5.0, n_timesteps=20, n_samples=100)
    plt.show()