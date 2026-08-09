import numpy as np
from scipy.fft import fft, ifft
import warnings

class ResonantPhaseAlignment:
    """
    Resonant Phase Alignment (RPA) Optimizer
    
    Instead of gradients, RPA uses:
    1. Phase scanning: Probe each parameter with sinusoidal perturbations
    2. Resonance detection: Measure loss response at different frequencies
    3. Phase locking: Adjust parameters based on phase coherence patterns
    4. Resonance amplification: Boost parameters that show constructive interference
    """
    
    def __init__(self, params, lr=0.01, n_frequencies=7, damping=0.9):
        self.params = np.array(params, dtype=float)
        self.n_params = len(params)
        self.lr = lr
        self.n_frequencies = n_frequencies
        self.damping = damping
        
        # Memory of previous resonance patterns
        self.resonance_memory = np.zeros(self.n_params)
        self.phase_history = np.zeros((self.n_params, n_frequencies))
        self.amplitude_history = np.zeros(self.n_params)
        
        # Initialize frequency bank (logarithmic spacing)
        self.frequencies = np.logspace(-3, 1, n_frequencies)
        
        # Phase coherence tracking
        self.coherence_metric = 0.0
        self.iteration = 0
        
    def probe_loss(self, loss_function, perturbation_scale=0.01):
        """
        Probe the loss landscape using resonance scanning
        
        Returns resonance signature vector
        """
        resonance_signature = np.zeros((self.n_params, self.n_frequencies))
        base_loss = loss_function(self.params)
        
        for i in range(self.n_params):
            for j, freq in enumerate(self.frequencies):
                # Generate phase-shifted perturbation
                phase = 2 * np.pi * np.random.random()
                
                # Apply resonant perturbation
                delta = perturbation_scale * np.sin(phase + freq * self.iteration)
                params_perturbed = self.params.copy()
                params_perturbed[i] += delta
                
                # Measure response
                perturbed_loss = loss_function(params_perturbed)
                response = perturbed_loss - base_loss
                
                # Store phase and amplitude information
                resonance_signature[i, j] = response * np.sin(phase)
                self.phase_history[i, j] = phase
        
        return resonance_signature
    
    def compute_resonance_amplification(self, resonance_signature):
        """
        Calculate which parameters resonate constructively
        Uses Fourier analysis to find coherent patterns
        """
        amplification = np.zeros(self.n_params)
        
        for i in range(self.n_params):
            # Convert resonance pattern to frequency domain
            spectrum = np.abs(fft(resonance_signature[i, :]))
            
            # Find dominant frequency and phase
            dominant_idx = np.argmax(spectrum[1:]) + 1  # Skip DC component
            
            # Compute phase coherence across frequencies
            coherence = np.mean(np.exp(1j * self.phase_history[i, :]))
            coherence_magnitude = np.abs(coherence)
            
            # Resonance amplification combines amplitude and coherence
            amplification[i] = (spectrum[dominant_idx] * 
                              coherence_magnitude * 
                              np.sin(2 * np.pi * dominant_idx / self.n_frequencies))
        
        # Normalize by total resonance energy
        total_energy = np.sum(np.abs(amplification)) + 1e-10
        amplification = amplification / total_energy
        
        return amplification
    
    def step(self, loss_function):
        """
        Perform one RPA optimization step
        """
        self.iteration += 1
        
        # 1. Probe the landscape
        resonance_sig = self.probe_loss(loss_function)
        
        # 2. Compute resonance amplification pattern
        amplification = self.compute_resonance_amplification(resonance_sig)
        
        # 3. Update memory with damping
        self.resonance_memory = (self.damping * self.resonance_memory + 
                                (1 - self.damping) * amplification)
        
        # 4. Apply update (resonant phase alignment)
        # The update is proportional to resonance amplitude but rotates phase
        update = self.lr * self.resonance_memory * np.cos(2 * np.pi * self.iteration / 3)
        
        # 5. Add small stochastic resonance exploration
        exploration = 0.01 * np.random.randn(self.n_params) * np.exp(-self.iteration / 1000)
        
        # 6. Apply update
        self.params = self.params - update + exploration
        
        # 7. Compute coherence metric (for convergence monitoring)
        self.coherence_metric = np.mean(np.abs(self.resonance_memory))
        
        return self.params, self.coherence_metric

# Example usage with mathematical proof concept
def demonstrate_rpa():
    """
    Demonstrate RPA on a simple quadratic loss function
    """
    np.random.seed(42)
    
    # Define a loss function: quadratic bowl with some ripples
    def loss_function(params):
        # Standard quadratic
        quad = np.sum((params - np.array([1.0, 2.0, -1.0]))**2)
        # Add small resonant ripples to make gradient noisy
        ripple = 0.1 * np.sum(np.sin(3 * params))
        return quad + ripple
    
    # Initialize parameters
    initial_params = np.array([0.0, 0.0, 0.0])
    optimizer = ResonantPhaseAlignment(initial_params, lr=0.05, 
                                       n_frequencies=9, damping=0.85)
    
    print("RPA Optimization - No Gradients Used!")
    print("=" * 50)
    print(f"Initial loss: {loss_function(initial_params):.6f}")
    print(f"Initial params: {initial_params}")
    print()
    
    # Optimization loop
    losses = []
    params_history = []
    
    for iteration in range(50):
        params, coherence = optimizer.step(loss_function)
        current_loss = loss_function(params)
        losses.append(current_loss)
        params_history.append(params.copy())
        
        if iteration % 10 == 0:
            print(f"Iter {iteration:3d} | Loss: {current_loss:.6f} | "
                  f"Coherence: {coherence:.4f} | Params: {[f'{p:.3f}' for p in params]}")
    
    print()
    print("=" * 50)
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Final params: {params}")
    print(f"Loss reduction: {losses[0] - losses[-1]:.6f}")
    
    return losses, params_history

# Run demonstration
losses, history = demonstrate_rpa()