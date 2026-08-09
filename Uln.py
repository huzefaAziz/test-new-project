import numpy as np

class ParticleSwarmOptimizer:
    def __init__(self, num_particles, num_weights, inertia=0.5, cognitive=1.5, social=1.5):
        """
        inertia: How much of the previous velocity to keep (prevents erratic jumps).
        cognitive: Weight given to the particle's own best historical position.
        social: Weight given to the swarm's absolute best position.
        """
        self.num_particles = num_particles
        self.num_weights = num_weights
        self.w = inertia
        self.c1 = cognitive
        self.c2 = social
        
        # Initialize random positions (weights) and zero velocities
        self.positions = np.random.uniform(-1.0, 1.0, (num_particles, num_weights))
        self.velocities = np.zeros((num_particles, num_weights))
        
        # Track historical records
        self.p_best_positions = np.copy(self.positions)
        self.p_best_scores = np.full(num_particles, float('inf'))
        self.g_best_position = np.zeros(num_weights)
        self.g_best_score = float('inf')

    def update(self, loss_function):
        """Evaluates the swarm and moves particles without using derivatives."""
        for i in range(self.num_particles):
            # 1. Compute the loss for the current particle configuration
            current_loss = loss_function(self.positions[i])
            
            # 2. Update Personal Best
            if current_loss < self.p_best_scores[i]:
                self.p_best_scores[i] = current_loss
                self.p_best_positions[i] = np.copy(self.positions[i])
                
            # 3. Update Global Best
            if current_loss < self.g_best_score:
                self.g_best_score = current_loss
                self.g_best_position = np.copy(self.positions[i])
        
        # 4. Update velocities and positions for the next iteration
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            
            # Physics-inspired step equation
            cognitive_velocity = self.c1 * r1 * (self.p_best_positions[i] - self.positions[i])
            social_velocity = self.c2 * r2 * (self.g_best_position - self.positions[i])
            
            self.velocities[i] = (self.w * self.velocities[i]) + cognitive_velocity + social_velocity
            self.positions[i] += self.velocities[i]

# --- Testing the Alternative Optimizer on a Linear Classification Task ---
if __name__ == "__main__":
    # Generate synthetic data (2 features, 100 samples)
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.array([1 if (2*x[0] - 3*x[1] + 0.5 > 0) else 0 for x in X])

    # Define a simple objective function (Mean Squared Error of a linear boundary)
    def cross_entropy_loss(weights):
        w = weights[:2]
        b = weights[2]
        predictions = 1 / (1 + np.exp(-(np.dot(X, w) + b)))
        return np.mean((predictions - y) ** 2)

    # Initialize optimizer: 30 particles tracking 3 parameters (2 weights, 1 bias)
    optimizer = ParticleSwarmOptimizer(num_particles=30, num_weights=3)

    print("Beginning Optimization Loop...")
    for generation in range(100):
        optimizer.update(cross_entropy_loss)
        if generation % 20 == 0:
            print(f"Generation {generation} -> Best Swarm Loss: {optimizer.g_best_score:.4f}")

    print("\nOptimization Complete!")
    print(f"Final Weights Found: {optimizer.g_best_position[:2]}")
    print(f"Final Bias Found: {optimizer.g_best_position[2]}")
