import networkx as nx
import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Callable, Any
import warnings
from scipy.special import softmax

class TruncatedPoissonFamily:
    """
    Implementation of the Truncated Poisson variational family from Definition 3.1.
    This family satisfies the unbounded with connected and bounded members property.
    """
    
    def __init__(self, delta: float = 0.95, lambda_param: float = 1.0):
        """
        Initialize the Truncated Poisson family.
        
        Args:
            delta: Quantile threshold for truncation (default: 0.95)
            lambda_param: Rate parameter for Poisson distribution
        """
        self.delta = delta
        self.lambda_param = lambda_param
        self._support_cache = None
        
    def poisson_pmf(self, k: int, lam: float) -> float:
        """Compute Poisson probability mass function."""
        if k < 0:
            return 0.0
        # Use math.factorial instead of np.math.factorial
        return np.exp(-lam) * (lam ** k) / math.factorial(k)
    
    def compute_quantile(self, lam: float, delta: float) -> int:
        """Compute the delta-quantile of Poisson distribution."""
        cumulative = 0.0
        k = 0
        while cumulative < delta:
            cumulative += self.poisson_pmf(k, lam)
            k += 1
        return k - 1
    
    def get_support(self) -> List[int]:
        """
        Get the support of the truncated Poisson distribution.
        Returns a list of integers from 0 to the quantile.
        """
        if self._support_cache is None:
            max_k = self.compute_quantile(self.lambda_param, self.delta)
            self._support_cache = list(range(max_k + 1))
        return self._support_cache
    
    def get_probabilities(self) -> Dict[int, float]:
        """
        Get probability mass function for the truncated distribution.
        """
        support = self.get_support()
        if not support:
            return {0: 1.0}
        
        # Compute unnormalized probabilities
        probs = {}
        for k in support:
            probs[k] = self.poisson_pmf(k, self.lambda_param)
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            for k in probs:
                probs[k] /= total
        
        return probs
    
    def get_max_truncation(self) -> int:
        """Get the maximum truncation level m(q(lambda))."""
        return max(self.get_support()) if self.get_support() else 0
    
    def update_lambda(self, new_lambda: float):
        """Update the lambda parameter and clear cache."""
        self.lambda_param = max(0.1, new_lambda)
        self._support_cache = None
    
    def sample(self, size: int = 1) -> np.ndarray:
        """Sample from the truncated Poisson distribution."""
        probs = self.get_probabilities()
        support = self.get_support()
        if not support:
            return np.zeros(size, dtype=int)
        
        probs_array = np.array([probs[k] for k in support])
        samples = np.random.choice(support, size=size, p=probs_array)
        return samples


class UDNLayer:
    """Represents a layer in the unbounded depth neural network."""
    
    def __init__(self, layer_id: int, input_dim: int, output_dim: int, 
                 activation: Optional[str] = 'relu'):
        """
        Initialize a neural network layer.
        
        Args:
            layer_id: Unique identifier for the layer
            input_dim: Input dimension
            output_dim: Output dimension
            activation: Activation function ('relu', 'tanh', 'sigmoid', or None)
        """
        self.layer_id = layer_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Initialize weights and biases with Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.weights = np.random.randn(input_dim, output_dim) * scale
        self.bias = np.zeros(output_dim)
        
        # Store gradients for optimization
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        z = x @ self.weights + self.bias
        
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        else:
            return z
    
    def backward(self, grad_output: np.ndarray, x: np.ndarray, 
                 learning_rate: float = 0.001) -> np.ndarray:
        """
        Backward pass for gradient computation.
        
        Args:
            grad_output: Gradient from next layer
            x: Input to this layer
            learning_rate: Learning rate for parameter update
        
        Returns:
            Gradient with respect to input
        """
        # Compute gradient of activation
        if self.activation == 'relu':
            z = x @ self.weights + self.bias
            grad_activation = grad_output * (z > 0).astype(float)
        elif self.activation == 'tanh':
            z = x @ self.weights + self.bias
            grad_activation = grad_output * (1 - np.tanh(z) ** 2)
        elif self.activation == 'sigmoid':
            z = x @ self.weights + self.bias
            sig = 1.0 / (1.0 + np.exp(-z))
            grad_activation = grad_output * sig * (1 - sig)
        else:
            grad_activation = grad_output
        
        # Compute gradients with respect to weights and bias
        grad_weights = x.T @ grad_activation
        grad_bias = np.sum(grad_activation, axis=0)
        
        # Update parameters
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        # Compute gradient with respect to input
        grad_input = grad_activation @ self.weights.T
        
        return grad_input


class OutputLayer:
    """Output layer for generating predictions at each truncation level."""
    
    def __init__(self, layer_id: int, input_dim: int, output_dim: int):
        """
        Initialize output layer.
        
        Args:
            layer_id: Unique identifier for the output layer
            input_dim: Input dimension (hidden state size)
            output_dim: Output dimension (number of classes or regression output)
        """
        self.layer_id = layer_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros(output_dim)
        
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through output layer."""
        return x @ self.weights + self.bias
    
    def backward(self, grad_output: np.ndarray, x: np.ndarray, 
                 learning_rate: float = 0.001) -> np.ndarray:
        """Backward pass for output layer."""
        grad_weights = x.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0)
        
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_output @ self.weights.T
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(x)


class UnboundedDepthNeuralNetwork:
    """
    Implementation of the Unbounded Depth Neural Network (UDN).
    
    This class implements the infinite-depth neural network with dynamic
    truncation as described in the paper.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dim: int = 32, prior_std: float = 1.0,
                 truncation_prior_rate: float = 0.5):
        """
        Initialize the UDN.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (number of classes)
            hidden_dim: Hidden layer dimension
            prior_std: Standard deviation for weight prior
            truncation_prior_rate: Rate parameter for Poisson prior on truncation
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.prior_std = prior_std
        self.truncation_prior_rate = truncation_prior_rate
        
        # Store layers lazily
        self.hidden_layers: Dict[int, UDNLayer] = {}
        self.output_layers: Dict[int, OutputLayer] = {}
        
        # Current maximum truncation level
        self.max_truncation = 0
        
        # Variational parameters
        self.truncation_distribution = TruncatedPoissonFamily(
            lambda_param=1.0
        )
        
        # NetworkX graph for computational graph representation
        self.computational_graph = nx.DiGraph()
        self._graph_initialized = False
        
    def _init_graph(self):
        """Initialize the computational graph."""
        if not self._graph_initialized:
            self.computational_graph.add_node('input', type='input')
            self._graph_initialized = True
        
    def get_layer(self, layer_id: int) -> UDNLayer:
        """Get or create a hidden layer."""
        self._init_graph()
        
        if layer_id not in self.hidden_layers:
            if layer_id == 1:
                input_dim = self.input_dim
            else:
                input_dim = self.hidden_dim
            
            # Different layer types for different depths (as in paper)
            if layer_id <= 3:
                activation = 'relu'
            elif layer_id <= 8:
                activation = 'relu'
            else:
                activation = 'relu'
            
            self.hidden_layers[layer_id] = UDNLayer(
                layer_id, input_dim, self.hidden_dim, activation
            )
            
            # Add to computational graph
            node_name = f'h{layer_id}'
            self.computational_graph.add_node(node_name, type='hidden', layer_id=layer_id)
            if layer_id == 1:
                self.computational_graph.add_edge('input', node_name)
            else:
                self.computational_graph.add_edge(f'h{layer_id-1}', node_name)
            
        return self.hidden_layers[layer_id]
    
    def get_output_layer(self, layer_id: int) -> OutputLayer:
        """Get or create an output layer."""
        self._init_graph()
        
        if layer_id not in self.output_layers:
            self.output_layers[layer_id] = OutputLayer(
                layer_id, self.hidden_dim, self.output_dim
            )
            
            # Add to computational graph
            node_name = f'o{layer_id}'
            self.computational_graph.add_node(node_name, type='output', layer_id=layer_id)
            self.computational_graph.add_edge(f'h{layer_id}', node_name)
            
        return self.output_layers[layer_id]
    
    def forward(self, x: np.ndarray, truncation: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through the network up to a truncation level.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            truncation: Truncation level. If None, uses the maximum support.
        
        Returns:
            Tuple of (outputs, hidden_states)
            outputs: List of outputs for each truncation level
            hidden_states: List of hidden states for each layer
        """
        if truncation is None:
            truncation = self.truncation_distribution.get_max_truncation()
        
        # Ensure we have the required layers
        for i in range(1, truncation + 1):
            self.get_layer(i)
            self.get_output_layer(i)
        
        # Forward pass through hidden layers
        h = x
        hidden_states = []
        
        for i in range(1, truncation + 1):
            h = self.hidden_layers[i].forward(h)
            hidden_states.append(h)
        
        # Get output for each truncation level
        outputs = []
        for i in range(1, truncation + 1):
            output = self.output_layers[i].forward(hidden_states[i-1])
            outputs.append(output)
        
        return outputs, hidden_states
    
    def compute_all_outputs(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Compute outputs for all truncation levels efficiently.
        Shares computation as described in Section 4 of the paper.
        
        Args:
            x: Input tensor
        
        Returns:
            List of outputs for each truncation level
        """
        max_truncation = self.truncation_distribution.get_max_truncation()
        outputs, _ = self.forward(x, max_truncation)
        return outputs
    
    def compute_elbo(self, X: np.ndarray, y: np.ndarray, 
                     compute_predictions: bool = True) -> Tuple[float, Optional[float]]:
        """
        Compute the Evidence Lower Bound (ELBO) as in equation (7) of the paper.
        
        Args:
            X: Input data (n_samples, input_dim)
            y: Target data (n_samples, output_dim)
            compute_predictions: Whether to compute predictive accuracy
        
        Returns:
            Tuple of (elbo_value, accuracy)
        """
        n_samples = X.shape[0]
        max_truncation = self.truncation_distribution.get_max_truncation()
        
        # Get truncation probabilities
        truncation_probs = self.truncation_distribution.get_probabilities()
        
        # Pre-compute all outputs for efficiency (linear complexity)
        outputs, hidden_states = self.forward(X, max_truncation)
        
        total_elbo = 0.0
        total_accuracy = 0.0 if compute_predictions else 0.0
        
        for truncation, prob in truncation_probs.items():
            if truncation == 0 or truncation > max_truncation:
                continue
            
            # Get output for this truncation level
            output = outputs[truncation - 1]
            
            # Compute log-likelihood
            # Using cross-entropy for classification
            log_likelihood = self._compute_log_likelihood(output, y)
            
            # Compute prior term for weights
            log_prior_weights = self._compute_log_prior_weights(truncation)
            
            # Compute variational approximation term
            log_variational_weights = self._compute_log_variational_weights(truncation)
            
            # Compute truncation prior and variational terms
            # ℓ-1 ~ Poisson(α) where α = truncation_prior_rate
            log_truncation_prior = -self.truncation_prior_rate + \
                (truncation - 1) * np.log(self.truncation_prior_rate) - \
                math.log(math.factorial(truncation - 1))
            
            log_truncation_variational = np.log(prob + 1e-10)  # Add small epsilon for stability
            
            # ELBO contribution for this truncation
            elbo_contrib = prob * (
                np.mean(log_likelihood) + 
                log_prior_weights - 
                log_variational_weights + 
                log_truncation_prior - 
                log_truncation_variational
            )
            
            total_elbo += elbo_contrib
            
            if compute_predictions:
                # Compute accuracy for this truncation
                pred = softmax(output, axis=1)
                acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1))
                total_accuracy += prob * acc
        
        return total_elbo, total_accuracy if compute_predictions else None
    
    def _compute_log_likelihood(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute log-likelihood (cross-entropy for classification)."""
        probs = softmax(output, axis=1)
        eps = 1e-10
        log_probs = np.log(np.clip(probs, eps, 1.0))
        return np.sum(y * log_probs, axis=1)
    
    def _compute_log_prior_weights(self, truncation: int) -> float:
        """Compute log of prior over weights up to truncation."""
        log_prior = 0.0
        for i in range(1, truncation + 1):
            if i in self.hidden_layers:
                layer = self.hidden_layers[i]
                # Gaussian prior with std prior_std
                log_prior += -0.5 * np.sum(layer.weights ** 2) / (self.prior_std ** 2)
                log_prior += -0.5 * np.sum(layer.bias ** 2) / (self.prior_std ** 2)
        return log_prior
    
    def _compute_log_variational_weights(self, truncation: int) -> float:
        """Compute log of variational approximation over weights."""
        log_var = 0.0
        for i in range(1, truncation + 1):
            if i in self.hidden_layers:
                layer = self.hidden_layers[i]
                # Mean-field Gaussian (log of 1 for unit variance)
                log_var += -0.5 * np.sum(layer.weights ** 2)
                log_var += -0.5 * np.sum(layer.bias ** 2)
        return log_var
    
    def update_variational_parameters(self, X: np.ndarray, y: np.ndarray,
                                     learning_rate: float = 0.001,
                                     update_truncation: bool = True):
        """
        Update variational parameters using gradient-based optimization.
        
        Args:
            X: Input data
            y: Target data
            learning_rate: Learning rate
            update_truncation: Whether to update truncation parameter
        """
        # Get current max truncation
        current_max = self.truncation_distribution.get_max_truncation()
        
        # Ensure we have layers up to current max
        for i in range(1, current_max + 1):
            self.get_layer(i)
            self.get_output_layer(i)
        
        # Compute outputs for all truncations
        outputs, hidden_states = self.forward(X, current_max)
        
        # Update output layers
        for i in range(1, current_max + 1):
            if i <= len(outputs):
                output = outputs[i-1]
                h = hidden_states[i-1]
                
                # Compute gradient (cross-entropy)
                probs = softmax(output, axis=1)
                grad = probs - y
                grad /= X.shape[0]
                
                self.output_layers[i].backward(grad, h, learning_rate)
        
        # Update hidden layers (backprop through all layers)
        # Gradients flow from output layers through all shared weights
        for i in range(current_max, 0, -1):
            if i <= len(hidden_states):
                h = hidden_states[i-1]
                x_prev = hidden_states[i-2] if i > 1 else X
                
                # Aggregate gradients from all output layers that use this hidden state
                grad = np.zeros_like(h)
                for j in range(i, current_max + 1):
                    if j <= len(outputs):
                        output = outputs[j-1]
                        probs = softmax(output, axis=1)
                        grad_j = probs - y
                        grad_j /= X.shape[0]
                        # Backprop through output layer
                        grad += self.output_layers[j].backward(grad_j, h, learning_rate)
                
                # Backprop through hidden layer
                self.hidden_layers[i].backward(grad, x_prev, learning_rate)
        
        # Update truncation distribution
        if update_truncation:
            # Simple heuristic: move lambda based on gradient direction
            # In practice, this would use the ELBO gradient
            current_lambda = self.truncation_distribution.lambda_param
            
            # Compute mean depth from current distribution
            probs = self.truncation_distribution.get_probabilities()
            mean_depth = sum(k * prob for k, prob in probs.items())
            
            # Update lambda to match desired depth (simplified)
            # In the full implementation, this would use the actual ELBO gradient
            target_lambda = mean_depth * 0.8 + 1.0
            new_lambda = current_lambda + 0.01 * (target_lambda - current_lambda)
            self.truncation_distribution.update_lambda(new_lambda)
    
    def predictive_distribution(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictive distribution as in equation (11) of the paper.
        
        Args:
            X: Input data
        
        Returns:
            Predictive distribution (ensemble of truncations)
        """
        probs = self.truncation_distribution.get_probabilities()
        max_truncation = self.truncation_distribution.get_max_truncation()
        
        outputs, _ = self.forward(X, max_truncation)
        
        # Ensemble prediction
        ensemble_output = np.zeros((X.shape[0], self.output_dim))
        
        for truncation, prob in probs.items():
            if truncation == 0 or truncation > max_truncation:
                continue
            
            output = outputs[truncation - 1]
            pred = softmax(output, axis=1)
            ensemble_output += prob * pred
        
        return ensemble_output
    
    def get_network_graph(self) -> nx.DiGraph:
        """Get the computational graph as a NetworkX DiGraph."""
        return self.computational_graph.copy()
    
    def get_layer_count(self) -> int:
        """Get the current number of layers."""
        return len(self.hidden_layers)
    
    def describe_architecture(self) -> Dict[str, Any]:
        """Get a description of the current architecture."""
        return {
            'total_layers': len(self.hidden_layers),
            'max_truncation': self.truncation_distribution.get_max_truncation(),
            'hidden_dim': self.hidden_dim,
            'truncation_lambda': self.truncation_distribution.lambda_param,
            'truncation_support': self.truncation_distribution.get_support(),
            'truncation_probs': self.truncation_distribution.get_probabilities()
        }


class DynamicVariationalInference:
    """
    Implementation of dynamic variational inference for the UDN.
    
    This implements Algorithm 1 from the paper, managing the dynamic creation
    and removal of variational parameters.
    """
    
    def __init__(self, model: UnboundedDepthNeuralNetwork, 
                 batch_size: int = 64, max_epochs: int = 100,
                 learning_rate: float = 0.001):
        """
        Initialize the dynamic variational inference algorithm.
        
        Args:
            model: UDN model
            batch_size: Batch size for stochastic optimization
            max_epochs: Maximum number of training epochs
            learning_rate: Learning rate
        """
        self.model = model
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        
        # Track training history
        self.history = {
            'elbo': [],
            'accuracy': [],
            'max_truncation': [],
            'lambda': []
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: bool = True):
        """
        Fit the UDN using dynamic variational inference.
        
        Args:
            X: Training data
            y: Training targets
            validation_data: Optional validation data
            verbose: Whether to print progress
        """
        n_samples = X.shape[0]
        
        for epoch in range(self.max_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            epoch_elbo = 0.0
            n_batches = 0
            
            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Get current max truncation (dynamic)
                max_trunc = self.model.truncation_distribution.get_max_truncation()
                
                # Ensure we have layers up to max truncation
                for i in range(1, max_trunc + 1):
                    self.model.get_layer(i)
                    self.model.get_output_layer(i)
                
                # Compute ELBO and update
                elbo, _ = self.model.compute_elbo(X_batch, y_batch, compute_predictions=False)
                
                # Update variational parameters
                self.model.update_variational_parameters(
                    X_batch, y_batch, self.learning_rate
                )
                
                epoch_elbo += elbo
                n_batches += 1
            
            # Record history
            avg_elbo = epoch_elbo / n_batches if n_batches > 0 else 0
            self.history['elbo'].append(avg_elbo)
            self.history['max_truncation'].append(
                self.model.truncation_distribution.get_max_truncation()
            )
            self.history['lambda'].append(
                self.model.truncation_distribution.lambda_param
            )
            
            # Compute validation accuracy if provided
            if validation_data is not None:
                X_val, y_val = validation_data
                pred = self.model.predictive_distribution(X_val)
                val_acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_val, axis=1))
                self.history['accuracy'].append(val_acc)
            
            if verbose and epoch % 10 == 0:
                val_str = f", Val Acc: {self.history['accuracy'][-1]:.4f}" if validation_data and self.history['accuracy'] else ""
                print(f"Epoch {epoch}: ELBO = {avg_elbo:.4f}, "
                      f"Max Truncation = {self.history['max_truncation'][-1]}{val_str}")
    
    def get_posterior(self) -> Dict[int, float]:
        """Get the posterior distribution over truncations."""
        return self.model.truncation_distribution.get_probabilities()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the posterior predictive distribution."""
        return self.model.predictive_distribution(X)


# Example usage and test function
def test_udn():
    """
    Test the UDN implementation with a simple dataset.
    """
    print("Testing Unbounded Depth Neural Network...")
    
    # Generate synthetic spiral data (simplified)
    np.random.seed(42)
    n_samples = 500
    omega = 20
    
    # Simple two-class spiral data
    X = np.random.randn(n_samples, 2)
    y = np.zeros((n_samples, 2))
    
    # Create simple pattern
    for i in range(n_samples):
        angle = np.arctan2(X[i, 1], X[i, 0])
        radius = np.sqrt(X[i, 0]**2 + X[i, 1]**2)
        if radius < 1.5:
            y[i, int((angle > 0))] = 1
        else:
            y[i, int((angle < 0))] = 1
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create UDN model
    model = UnboundedDepthNeuralNetwork(
        input_dim=2,
        output_dim=2,
        hidden_dim=16,
        prior_std=1.0,
        truncation_prior_rate=0.5
    )
    
    # Create inference engine
    inference = DynamicVariationalInference(
        model=model,
        batch_size=32,
        max_epochs=50,
        learning_rate=0.001
    )
    
    # Train the model
    print("Training UDN...")
    inference.fit(X_train, y_train, validation_data=(X_test, y_test))
    
    # Make predictions
    predictions = inference.predict(X_test)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
    
    # Get posterior distribution
    posterior = inference.get_posterior()
    
    print(f"\nResults:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Posterior over truncations: {posterior}")
    print(f"Model architecture: {model.describe_architecture()}")
    
    # Visualize computational graph
    graph = model.get_network_graph()
    print(f"Computational graph nodes: {list(graph.nodes())[:10]}...")  # Show first 10 nodes
    print(f"Computational graph edges: {list(graph.edges())[:10]}...")  # Show first 10 edges
    
    return model, inference


if __name__ == "__main__":
    # Run test
    model, inference = test_udn()