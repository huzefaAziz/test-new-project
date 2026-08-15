"""
Fluctuation-Dissipation Relations for Stochastic Gradient Descent
Implementation of the key concepts from the paper by Sho Yaida (Facebook AI Research)

This module implements:
1. FDR1: First Fluctuation-Dissipation Relation for checking stationarity
2. FDR2: Second Fluctuation-Dissipation Relation for landscape analysis
3. Adaptive learning rate scheduler based on FDR1
"""

import numpy as np
from typing import Callable, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from collections import deque
import warnings


@dataclass
class SGDState:
    """State container for SGD training with momentum."""
    theta: np.ndarray  # Model parameters
    v: np.ndarray      # Velocity/momentum term
    step: int          # Current step
    epoch: int         # Current epoch
    learning_rate: float
    momentum: float
    dampening: float
    
    def __post_init__(self):
        if self.v is None:
            self.v = np.zeros_like(self.theta)


class FDRMetrics:
    """
    Metrics for evaluating Fluctuation-Dissipation Relations during training.
    
    This class computes the observables needed for FDR1 and FDR2 on the fly.
    """
    
    def __init__(self, 
                 momentum: float = 0.0,
                 dampening: float = 0.0,
                 running_window: int = 100):
        """
        Args:
            momentum: Momentum coefficient μ (default: 0)
            dampening: Dampening coefficient ν (default: 0)
            running_window: Window size for running averages
        """
        self.momentum = momentum
        self.dampening = dampening
        self.running_window = running_window
        
        # Buffers for running averages
        self.O_L_buffer = deque(maxlen=running_window)
        self.O_R_buffer = deque(maxlen=running_window)
        self.O_FB_buffer = deque(maxlen=running_window)
        
        # Full averages
        self.O_L_avg = 0.0
        self.O_R_avg = 0.0
        self.O_FB_avg = 0.0
        
        self._count = 0
        self._half_count = 0
        self._in_half = True  # For half-running average
        
    def update(self, 
               theta: np.ndarray,
               grad_mini_batch: np.ndarray,
               grad_full_batch: Optional[np.ndarray] = None,
               v: Optional[np.ndarray] = None,
               eta: float = 0.0) -> Dict[str, float]:
        """
        Update metrics with current training step.
        
        Args:
            theta: Current model parameters
            grad_mini_batch: Mini-batch gradient ∇f^B(θ)
            grad_full_batch: Full-batch gradient ∇f(θ) (optional)
            v: Current velocity (if using momentum)
            eta: Learning rate
            
        Returns:
            Dictionary of current metric values
        """
        self._count += 1
        if self._count <= self.running_window:
            self._in_half = True
            self._half_count += 1
        
        # FDR1 observables
        O_L = np.dot(theta, grad_mini_batch)
        
        if v is not None:
            # With momentum
            if self.dampening < 1.0:
                factor = (1 + self.momentum) / (2 * (1 - self.dampening))
            else:
                factor = (1 + self.momentum) / 2
            O_R = factor * eta * np.dot(v, v)
        else:
            # Without momentum
            O_R = 0.5 * eta * np.dot(grad_mini_batch, grad_mini_batch)
        
        self.O_L_buffer.append(O_L)
        self.O_R_buffer.append(O_R)
        
        # Update running averages
        self.O_L_avg = np.mean(self.O_L_buffer)
        self.O_R_avg = np.mean(self.O_R_buffer)
        
        # FDR2 observable (requires full-batch gradient)
        O_FB = None
        if grad_full_batch is not None:
            if v is not None:
                O_FB = (1 - self.dampening) * np.dot(grad_full_batch, grad_full_batch) - \
                       self.momentum * np.dot(v, grad_full_batch)
            else:
                O_FB = np.dot(grad_full_batch, grad_full_batch)
            
            self.O_FB_buffer.append(O_FB)
            self.O_FB_avg = np.mean(self.O_FB_buffer)
        
        # Ratio for checking FDR1 saturation
        ratio = self.O_L_avg / self.O_R_avg if self.O_R_avg != 0 else float('inf')
        
        return {
            'O_L': O_L,
            'O_R': O_R,
            'O_L_avg': self.O_L_avg,
            'O_R_avg': self.O_R_avg,
            'O_L_avg_halved': self._get_half_average(self.O_L_buffer),
            'O_R_avg_halved': self._get_half_average(self.O_R_buffer),
            'FDR1_ratio': ratio,
            'FDR1_saturation': abs(ratio - 1) if np.isfinite(ratio) else float('inf'),
            'O_FB_avg': self.O_FB_avg if grad_full_batch is not None else None,
        }
    
    def _get_half_average(self, buffer: deque) -> float:
        """Compute half-running average (discard initial half)."""
        arr = np.array(buffer)
        if len(arr) == 0:
            return 0.0
        half_idx = len(arr) // 2
        if half_idx >= len(arr):
            return np.mean(arr)
        return np.mean(arr[half_idx:])
    
    def get_FDR1_ratio(self) -> float:
        """Get current FDR1 ratio ⟨O_L⟩ / ⟨O_R⟩."""
        if self.O_R_avg == 0:
            return float('inf')
        return self.O_L_avg / self.O_R_avg
    
    def is_equilibrated(self, threshold: float = 0.01) -> bool:
        """
        Check if system is equilibrated based on FDR1.
        
        Args:
            threshold: Relative tolerance for |⟨O_L⟩/⟨O_R⟩ - 1| < threshold
            
        Returns:
            True if equilibrated
        """
        if self.O_R_avg == 0:
            return False
        ratio = self.O_L_avg / self.O_R_avg
        return abs(ratio - 1) < threshold


class AdaptiveScheduler:
    """
    Adaptive learning rate scheduler based on FDR1.
    
    Implements the algorithm from Section 3.3 of the paper:
    1. Evaluate half-running averages of O_L and O_R at end of each epoch
    2. If |⟨O_L⟩/⟨O_R⟩ - 1| < X, decrease learning rate: η → (1-Y)η
    """
    
    def __init__(self,
                 initial_lr: float,
                 threshold: float = 0.01,
                 decrease_factor: float = 0.1,
                 min_lr: float = 1e-6,
                 momentum: float = 0.0,
                 dampening: float = 0.0):
        """
        Args:
            initial_lr: Initial learning rate
            threshold: X - threshold for FDR1 saturation (default: 0.01 = 1%)
            decrease_factor: Y - amount to decrease learning rate (default: 0.1 = 10%)
            min_lr: Minimum learning rate
            momentum: Momentum coefficient
            dampening: Dampening coefficient
        """
        self.lr = initial_lr
        self.threshold = threshold
        self.decrease_factor = decrease_factor
        self.min_lr = min_lr
        self.momentum = momentum
        self.dampening = dampening
        
        # State for tracking
        self.decrease_count = 0
        self.epoch_count = 0
        self.epoch_reset = True  # Reset averages at start of each epoch
        self.metrics_at_check = None
        
        # Running buffer for epoch
        self.epoch_O_L = []
        self.epoch_O_R = []
        
    def step_epoch(self, metrics: FDRMetrics) -> Tuple[float, bool]:
        """
        Process end of epoch metrics and update learning rate if needed.
        
        Args:
            metrics: FDRMetrics instance with current averages
            
        Returns:
            (new_learning_rate, was_decreased) tuple
        """
        self.epoch_count += 1
        
        # Get half-running averages
        O_L_halved = metrics._get_half_average(metrics.O_L_buffer)
        O_R_halved = metrics._get_half_average(metrics.O_R_buffer)
        
        if O_R_halved == 0:
            return self.lr, False
        
        ratio = O_L_halved / O_R_halved
        
        # Check if equilibration criterion is met
        if abs(ratio - 1) < self.threshold and self.lr > self.min_lr:
            # Decrease learning rate
            self.lr = max(self.lr * (1 - self.decrease_factor), self.min_lr)
            self.decrease_count += 1
            
            # Reset buffers (set t=0 for half-running averages)
            metrics.O_L_buffer.clear()
            metrics.O_R_buffer.clear()
            self.epoch_reset = True
            
            return self.lr, True
        
        self.epoch_reset = False
        return self.lr, False
    
    def get_schedule(self) -> Dict[str, Any]:
        """Get current schedule information."""
        return {
            'learning_rate': self.lr,
            'decrease_count': self.decrease_count,
            'epoch': self.epoch_count,
            'threshold': self.threshold,
            'decrease_factor': self.decrease_factor,
        }


class FDRTrainer:
    """
    Trainer that implements FDR-based training with adaptive scheduling.
    
    This is a wrapper that can be used with any model that provides
    gradients and loss values.
    """
    
    def __init__(self,
                 model: Any,
                 loss_fn: Callable,
                 grad_fn: Callable,
                 learning_rate: float = 0.1,
                 momentum: float = 0.0,
                 dampening: float = 0.0,
                 weight_decay: float = 0.0,
                 use_adaptive_schedule: bool = False,
                 fdr_threshold: float = 0.01,
                 fdr_decrease_factor: float = 0.1):
        """
        Args:
            model: Model with parameters
            loss_fn: Function to compute loss
            grad_fn: Function to compute gradients
            learning_rate: Initial learning rate
            momentum: Momentum coefficient
            dampening: Dampening coefficient
            weight_decay: L2 regularization coefficient
            use_adaptive_schedule: Whether to use adaptive scheduling
            fdr_threshold: Threshold for FDR1 saturation
            fdr_decrease_factor: Learning rate decrease factor
        """
        self.model = model
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.weight_decay = weight_decay
        
        # Get parameter dimension
        try:
            self.param_dim = sum(p.size for p in model.parameters())
        except AttributeError:
            # If model doesn't have parameters attribute, assume it's a numpy array
            self.param_dim = model.shape[0] if hasattr(model, 'shape') else 0
        
        # Initialize FDR metrics
        self.metrics = FDRMetrics(
            momentum=momentum,
            dampening=dampening,
            running_window=100
        )
        
        # Initialize scheduler
        self.scheduler = AdaptiveScheduler(
            initial_lr=learning_rate,
            threshold=fdr_threshold,
            decrease_factor=fdr_decrease_factor,
            momentum=momentum,
            dampening=dampening
        ) if use_adaptive_schedule else None
        
        self.momentum = momentum
        self.dampening = dampening
        self.velocity = None
        self.step_count = 0
        self.epoch_count = 0
        
        # History for analysis
        self.history = {
            'loss': [],
            'grad_norm': [],
            'learning_rate': [],
            'FDR1_ratio': [],
            'FDR1_saturation': [],
            'O_L_avg': [],
            'O_R_avg': [],
            'O_FB_avg': [],
        }
    
    def step(self, 
             theta: np.ndarray,
             grad_mini_batch: np.ndarray,
             grad_full_batch: Optional[np.ndarray] = None,
             is_epoch_end: bool = False) -> Dict[str, float]:
        """
        Perform one training step with FDR monitoring.
        
        Args:
            theta: Current parameters
            grad_mini_batch: Mini-batch gradient
            grad_full_batch: Full-batch gradient (optional, for FDR2)
            is_epoch_end: Whether this is the end of an epoch
            
        Returns:
            Dictionary of metrics
        """
        self.step_count += 1
        
        # Update parameters (with momentum if enabled)
        if self.momentum > 0:
            if self.velocity is None:
                self.velocity = np.zeros_like(theta)
            
            self.velocity = self.momentum * self.velocity - \
                           (1 - self.dampening) * grad_mini_batch
            theta_new = theta + self.scheduler.lr * self.velocity
            
            # Get current learning rate for metrics
            lr = self.scheduler.lr if self.scheduler else 0.1
            v = self.velocity
        else:
            theta_new = theta - self.scheduler.lr * grad_mini_batch
            lr = self.scheduler.lr if self.scheduler else 0.1
            v = None
        
        # Update FDR metrics
        metrics_dict = self.metrics.update(
            theta=theta,
            grad_mini_batch=grad_mini_batch,
            grad_full_batch=grad_full_batch,
            v=v,
            eta=lr
        )
        
        # Update history
        self.history['learning_rate'].append(lr)
        self.history['FDR1_ratio'].append(metrics_dict['FDR1_ratio'])
        self.history['FDR1_saturation'].append(metrics_dict['FDR1_saturation'])
        self.history['O_L_avg'].append(metrics_dict['O_L_avg'])
        self.history['O_R_avg'].append(metrics_dict['O_R_avg'])
        if metrics_dict['O_FB_avg'] is not None:
            self.history['O_FB_avg'].append(metrics_dict['O_FB_avg'])
        
        # Handle epoch end and adaptive scheduling
        if is_epoch_end and self.scheduler is not None:
            self.epoch_count += 1
            new_lr, decreased = self.scheduler.step_epoch(self.metrics)
            if decreased:
                # Reset velocity when learning rate changes
                self.velocity = None
        
        # Compute loss if needed
        try:
            loss = self.loss_fn(theta_new)
            self.history['loss'].append(loss)
        except:
            pass
        
        # Compute gradient norm
        self.history['grad_norm'].append(np.linalg.norm(grad_mini_batch))
        
        return {
            'theta_new': theta_new,
            'metrics': metrics_dict,
            'step': self.step_count,
            'epoch': self.epoch_count,
        }
    
    def get_FDR2_analysis(self, 
                          eta_values: List[float],
                          n_samples: int = 100) -> Dict[str, Any]:
        """
        Analyze loss landscape using FDR2.
        
        This computes G(η) = ⟨(∇f)²⟩ as a function of learning rate.
        
        Args:
            eta_values: List of learning rates to test
            n_samples: Number of samples per learning rate
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'eta': eta_values,
            'G_eta': [],  # ⟨(∇f)²⟩
            'slope': [],  # Slope for small eta
            'anharmonicity': [],  # Deviation from linearity
        }
        
        # This would require running the model at each learning rate
        # and computing the stationary averages
        # Implementation depends on the specific model
        
        return results
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot training results (requires matplotlib).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed, cannot plot")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        # Loss
        axes[0].plot(self.history['loss'])
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Step')
        
        # FDR1 ratio
        axes[1].plot(self.history['FDR1_ratio'])
        axes[1].axhline(y=1, color='r', linestyle='--', label='Equilibrium')
        axes[1].set_ylabel('FDR1 Ratio')
        axes[1].set_xlabel('Step')
        axes[1].legend()
        
        # Learning rate
        axes[2].plot(self.history['learning_rate'])
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_xlabel('Step')
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


# Example usage with a simple model

class SimpleModel:
    """Simple quadratic model for demonstration."""
    
    def __init__(self, dim: int = 10, noise_scale: float = 0.1):
        self.dim = dim
        self.noise_scale = noise_scale
        self.theta = np.random.randn(dim) * 0.1
        
    def loss(self, theta: np.ndarray) -> float:
        """Quadratic loss: f(θ) = 0.5 * θ^T H θ"""
        # Hessian with eigenvalues in [0.1, 10]
        H = np.diag(np.linspace(0.1, 10, self.dim))
        return 0.5 * theta @ H @ theta
    
    def grad(self, theta: np.ndarray, use_mini_batch: bool = True) -> np.ndarray:
        """Gradient with optional noise for mini-batch."""
        H = np.diag(np.linspace(0.1, 10, self.dim))
        grad = H @ theta
        
        if use_mini_batch:
            # Add noise to simulate mini-batch
            noise = np.random.randn(self.dim) * self.noise_scale
            grad += noise
            
        return grad


def demo_fdr():
    """Demonstrate FDR implementation on a simple quadratic model."""
    print("=" * 60)
    print("Fluctuation-Dissipation Relations Demo")
    print("=" * 60)
    
    # Setup
    dim = 20
    model = SimpleModel(dim=dim, noise_scale=0.05)
    initial_theta = np.random.randn(dim) * 0.1
    
    # Train with adaptive scheduling
    print("\nTraining with adaptive scheduling...")
    
    trainer = FDRTrainer(
        model=model,
        loss_fn=model.loss,
        grad_fn=model.grad,
        learning_rate=0.01,
        use_adaptive_schedule=True,
        fdr_threshold=0.02,
        fdr_decrease_factor=0.1
    )
    
    # Simulation
    theta = initial_theta.copy()
    n_steps = 1000
    
    for step in range(n_steps):
        # Get gradients
        grad_mini = model.grad(theta, use_mini_batch=True)
        grad_full = model.grad(theta, use_mini_batch=False) if step % 100 == 0 else None
        
        # Step
        is_epoch = (step + 1) % 100 == 0
        result = trainer.step(theta, grad_mini, grad_full, is_epoch_end=is_epoch)
        theta = result['theta_new']
        
        # Print progress
        if step % 100 == 0:
            metrics = result['metrics']
            print(f"Step {step}: Loss={model.loss(theta):.4f}, "
                  f"FDR1 ratio={metrics['FDR1_ratio']:.4f}, "
                  f"LR={trainer.scheduler.lr:.6f}")
    
    # Results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Final learning rate: {trainer.scheduler.lr:.6f}")
    print(f"Number of decreases: {trainer.scheduler.decrease_count}")
    print(f"Final loss: {model.loss(theta):.4f}")
    print(f"Parameter norm: {np.linalg.norm(theta):.4f}")
    
    # Show FDR1 saturation
    final_ratio = trainer.metrics.get_FDR1_ratio()
    print(f"Final FDR1 ratio ⟨O_L⟩/⟨O_R⟩: {final_ratio:.4f}")
    print(f"Equilibrated: {trainer.metrics.is_equilibrated(threshold=0.02)}")
    
    # Check FDR1 relation
    print("\nFDR1 Verification:")
    print(f"  ⟨θ·∇f⟩ ≈ {trainer.metrics.O_L_avg:.4f}")
    print(f"  0.5η⟨Tr(C̃)⟩ ≈ {trainer.metrics.O_R_avg:.4f}")
    print(f"  Ratio: {trainer.metrics.O_L_avg / trainer.metrics.O_R_avg:.4f} (should be ~1)")
    
    return trainer


def demo_fdr2_analysis():
    """
    Demonstrate FDR2 analysis for loss landscape.
    Simulates the behavior shown in Figure 3 of the paper.
    """
    print("\n" + "=" * 60)
    print("FDR2 Analysis Demo - Loss Landscape Characterization")
    print("=" * 60)
    
    dim = 10
    
    # Create models with different levels of anharmonicity
    print("\nAnalyzing harmonic vs. anharmonic landscapes...")
    
    # Harmonic landscape
    model_harmonic = SimpleModel(dim=dim, noise_scale=0.05)
    
    # Anharmonic landscape (add cubic term)
    class AnharmonicModel(SimpleModel):
        def loss(self, theta):
            # Add cubic anharmonicity
            harmonic = super().loss(theta)
            cubic = 0.1 * np.sum(theta**3)
            return harmonic + cubic
        
        def grad(self, theta, use_mini_batch=True):
            grad = super().grad(theta, use_mini_batch=False)
            # Add cubic gradient
            grad += 0.3 * theta**2
            if use_mini_batch:
                grad += np.random.randn(self.dim) * 0.05
            return grad
    
    model_anharmonic = AnharmonicModel(dim=dim, noise_scale=0.05)
    
    # Function to compute G(η) = ⟨(∇f)²⟩ for a range of learning rates
    def compute_G_eta(model, eta_values, n_steps=500, n_runs=5):
        G_values = []
        for eta in eta_values:
            G_run = []
            for run in range(n_runs):
                theta = np.random.randn(dim) * 0.1
                for step in range(n_steps):
                    grad = model.grad(theta, use_mini_batch=True)
                    theta = theta - eta * grad
                    if step > n_steps // 2:  # Discard transient
                        full_grad = model.grad(theta, use_mini_batch=False)
                        G_run.append(np.dot(full_grad, full_grad))
            G_values.append(np.mean(G_run))
        return np.array(G_values)
    
    # Test learning rates
    eta_values = np.logspace(-4, -1, 10)
    
    print(f"Testing learning rates: {eta_values}")
    
    # Compute for harmonic model
    print("\nHarmonic model:")
    G_harmonic = compute_G_eta(model_harmonic, eta_values)
    
    # Compute for anharmonic model
    print("Anharmonic model:")
    G_anharmonic = compute_G_eta(model_anharmonic, eta_values)
    
    # Analyze linearity
    print("\n" + "=" * 60)
    print("Linearity Analysis")
    print("=" * 60)
    
    # Fit linear regime (small eta)
    small_eta_idx = eta_values < 0.01
    if np.any(small_eta_idx):
        from scipy import stats
        
        # Harmonic
        slope_h, intercept_h, r_h, _, _ = stats.linregress(
            eta_values[small_eta_idx], G_harmonic[small_eta_idx]
        )
        print(f"Harmonic model: slope={slope_h:.3e}, R²={r_h**2:.4f}")
        
        # Anharmonic
        slope_a, intercept_a, r_a, _, _ = stats.linregress(
            eta_values[small_eta_idx], G_anharmonic[small_eta_idx]
        )
        print(f"Anharmonic model: slope={slope_a:.3e}, R²={r_a**2:.4f}")
    
    # Analyze nonlinearity (anharmonicity)
    print("\nAnharmonicity (deviation from linearity at higher η):")
    for i, eta in enumerate(eta_values):
        if eta > 0.01:
            linear_pred_h = G_harmonic[small_eta_idx][-1] * (eta / eta_values[small_eta_idx][-1])
            linear_pred_a = G_anharmonic[small_eta_idx][-1] * (eta / eta_values[small_eta_idx][-1])
            print(f"  η={eta:.4f}:")
            print(f"    Harmonic: G={G_harmonic[i]:.4f}, linear pred={linear_pred_h:.4f}, "
                  f"deviation={(G_harmonic[i]-linear_pred_h)/linear_pred_h*100:.1f}%")
            print(f"    Anharmonic: G={G_anharmonic[i]:.4f}, linear pred={linear_pred_a:.4f}, "
                  f"deviation={(G_anharmonic[i]-linear_pred_a)/linear_pred_a*100:.1f}%")
    
    return {
        'eta': eta_values,
        'G_harmonic': G_harmonic,
        'G_anharmonic': G_anharmonic
    }


def main():
    """Run all demonstrations."""
    # FDR1 demo with adaptive scheduling
    trainer = demo_fdr()
    
    # FDR2 landscape analysis
    results = demo_fdr2_analysis()
    
    print("\n" + "=" * 60)
    print("Key Takeaways from Fluctuation-Dissipation Relations")
    print("=" * 60)
    print("""
    FDR1: ⟨θ·∇f⟩ = 0.5η⟨Tr(C̃)⟩
    - Use to check stationarity/equilibration
    - Use for adaptive learning rate scheduling
    
    FDR2: ⟨(∇f)²⟩ = (η/2)⟨Tr(HC̃)⟩ + O(η²)
    - Small η: measures Hessian magnitude
    - Large η: reveals anharmonicity
    - Deviation from linearity indicates breakdown of harmonic approximation
    
    Adaptive Scheduler:
    - Monitor FDR1 ratio
    - Decrease learning rate when |⟨O_L⟩/⟨O_R⟩ - 1| < X
    - Reduces need for manual scheduling
    """)


if __name__ == "__main__":
    main()