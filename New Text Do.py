import numpy as np

class PatternSearchOptimizer:
    """
    An alternative to Gradient Descent that optimizes parameters 
    using coordinate exploration and pattern acceleration.
    """
    def __init__(self, step_size=0.5, step_reduction=0.5, tolerance=1e-5):
        self.step_size = step_size
        self.step_reduction = step_reduction
        self.tolerance = tolerance

    def _explore(self, loss_fn, x, current_loss, alpha):
        """Explores neighboring spaces along coordinate axes."""
        dims = len(x)
        x_new = np.copy(x)
        
        for i in range(dims):
            for sign in [1, -1]:
                x_test = np.copy(x_new)
                x_test[i] += sign * alpha
                test_loss = loss_fn(x_test)
                
                if test_loss < current_loss:
                    current_loss = test_loss
                    x_new = x_test
                    break # Accept step and move to next dimension
        return x_new, current_loss

    def minimize(self, loss_fn, x_init, max_iter=500):
        """Performs Hooke-Jeeves optimization to minimize loss_fn."""
        x = np.array(x_init, dtype=float)
        alpha = self.step_size
        current_loss = loss_fn(x)
        
        for iteration in range(max_iter):
            if alpha < self.tolerance:
                break
                
            # Step 1: Exploratory Move
            x_explored, new_loss = self._explore(loss_fn, x, current_loss, alpha)
            
            if new_loss < current_loss:
                # Step 2: Pattern Move (Accelerate in successful direction)
                pattern_vector = x_explored - x
                x_pattern = x_explored + pattern_vector
                pattern_loss = loss_fn(x_pattern)
                
                # Further explore around the pattern point
                x_pattern_explored, final_loss = self._explore(loss_fn, x_pattern, pattern_loss, alpha)
                
                if final_loss < new_loss:
                    x = x_pattern_explored
                    current_loss = final_loss
                else:
                    x = x_explored
                    current_loss = new_loss
            else:
                # Reduce step size if no improvements are found
                alpha *= self.step_reduction
                
        return x, current_loss

# ==========================================
# EXAMPLE USAGE: Optimizing a Loss Function
# ==========================================
if __name__ == "__main__":
    # Define a complex, non-differentiable loss landscape (with absolute values)
    def noisy_loss_function(theta):
        x, y = theta[0], theta[1]
        # Global minimum is at (3, 2) where loss = 0
        base_loss = (x - 3)**2 + (y - 2)**2
        noise = 2.0 * abs(np.sin(x)) if abs(x - 3) > 0.1 else 0
        return base_loss + noise

    # Initialize parameters randomly or sub-optimally
    initial_weights = [10.0, -5.0]
    
    optimizer = PatternSearchOptimizer(step_size=1.0, tolerance=1e-6)
    optimal_weights, final_loss = optimizer.minimize(noisy_loss_function, initial_weights)
    
    print(f"Initial Weights: {initial_weights}")
    print(f"Optimized Weights: {np.round(optimal_weights, 4)}")
    print(f"Final Minimum Loss: {round(final_loss, 6)}")

