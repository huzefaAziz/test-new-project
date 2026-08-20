import numpy as np
from scipy.special import softmax
from typing import List, Tuple, Dict, Any, Optional, Union
import time

class DCNN:
    """
    Diffusion-Convolutional Neural Network implementation.
    """
    
    def __init__(self, 
                 n_hops: int = 2, 
                 n_features: Optional[int] = None,
                 n_classes: Optional[int] = None,
                 learning_rate: float = 0.05,
                 activation: str = 'tanh',
                 task_type: str = 'node',
                 random_seed: int = 42):
        """
        Initialize DCNN model.
        """
        np.random.seed(random_seed)
        
        self.n_hops = n_hops
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.task_type = task_type
        
        # Set activation function
        if activation == 'tanh':
            self.activation = np.tanh
            self.activation_derivative = lambda x: 1 - np.tanh(x) ** 2
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: (x > 0).astype(float)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize weights
        self.Wc = None
        self.Wd = None
        
    def _initialize_weights(self, n_features: int, n_classes: int):
        """Initialize weights with small random values."""
        self.n_features = n_features
        self.n_classes = n_classes
        
        # Initialize with mean 0, variance 0.01
        self.Wc = np.random.normal(0, 0.01, (self.n_hops, n_features))
        self.Wd = np.random.normal(0, 0.01, (n_classes, self.n_hops * n_features))
        
    def _compute_transition_matrix(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Compute degree-normalized transition matrix P.
        """
        # Compute degree matrix
        degrees = adjacency.sum(axis=1, keepdims=True)
        
        # Handle zero-degree nodes by adding self-loop
        degrees[degrees == 0] = 1
        
        # Normalize: P = D^{-1} * A
        P = adjacency / degrees
        
        return P
    
    def _compute_power_series(self, P: np.ndarray, n_hops: int) -> np.ndarray:
        """
        Compute power series of transition matrix: P^0, P^1, ..., P^(H-1)
        """
        N = P.shape[0]
        P_star = np.zeros((N, n_hops, N))
        
        # P^0 is identity matrix
        P_star[:, 0, :] = np.eye(N)
        
        # Compute powers
        P_power = np.eye(N)
        for h in range(1, n_hops):
            P_power = P_power @ P
            P_star[:, h, :] = P_power
            
        return P_star
    
    def _diffusion_convolution_node(self, 
                                   P_star: np.ndarray, 
                                   X: np.ndarray, 
                                   Wc: np.ndarray) -> np.ndarray:
        """
        Apply diffusion-convolution operation for node classification.
        """
        N, H, _ = P_star.shape
        F = X.shape[1]
        
        # Compute P_star @ X: (N x H x N) @ (N x F) -> (N x H x F)
        Z = np.einsum('nhn,nf->nhf', P_star, X)
        
        # Apply element-wise multiplication with weights and activation
        Z = self.activation(Z * Wc[np.newaxis, :, :])
        
        return Z
    
    def _diffusion_convolution_graph(self, 
                                    P_star: np.ndarray, 
                                    X: np.ndarray, 
                                    Wc: np.ndarray) -> np.ndarray:
        """
        Apply diffusion-convolution operation for graph classification.
        """
        N, H, _ = P_star.shape
        F = X.shape[1]
        
        # Compute mean over nodes: (1/N) * 1^T @ P_star @ X
        Z = np.einsum('nhn,nf->hf', P_star, X) / N
        
        # Apply element-wise multiplication with weights and activation
        Z = self.activation(Z * Wc)
        
        return Z
    
    def _hinge_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute multi-class hinge loss.
        
        Args:
            logits: B x C prediction logits
            labels: B x C one-hot labels
            
        Returns:
            loss: Scalar loss value
        """
        B, C = logits.shape
        
        # Ensure labels are one-hot encoded
        if len(labels.shape) == 1:
            # Convert to one-hot if needed
            labels_one_hot = np.zeros((B, C))
            labels_one_hot[np.arange(B), labels.astype(int)] = 1
        else:
            labels_one_hot = labels
        
        # Compute correct class scores
        correct_scores = np.sum(logits * labels_one_hot, axis=1, keepdims=True)
        
        # Compute margins: max(0, 1 - correct + scores for each class)
        # The correct class margin should be 0
        margins = np.maximum(0, 1 - correct_scores + logits)
        
        # Set correct class margin to 0
        margins = margins * (1 - labels_one_hot)
        
        return np.mean(np.sum(margins, axis=1))
    
    def _hinge_loss_gradient(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute gradient of multi-class hinge loss.
        
        Args:
            logits: B x C prediction logits
            labels: B x C one-hot labels
            
        Returns:
            gradient: B x C gradient
        """
        B, C = logits.shape
        
        # Ensure labels are one-hot encoded
        if len(labels.shape) == 1:
            labels_one_hot = np.zeros((B, C))
            labels_one_hot[np.arange(B), labels.astype(int)] = 1
        else:
            labels_one_hot = labels
        
        # Compute correct class scores
        correct_scores = np.sum(logits * labels_one_hot, axis=1, keepdims=True)
        
        # Compute margins
        margins = np.maximum(0, 1 - correct_scores + logits)
        
        # Gradient: 1 if margin > 0 for non-correct classes
        grad = (margins > 0).astype(float)
        
        # Set gradient to 0 for correct class
        grad = grad * (1 - labels_one_hot)
        
        # Negative gradient for correct class (sum of all positive margins)
        grad_correct = -np.sum(grad, axis=1, keepdims=True)
        grad = grad + grad_correct * labels_one_hot
        
        return grad / B
    
    def _edge_to_node_transform(self, 
                               adjacency: np.ndarray, 
                               features: np.ndarray,
                               edge_features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform graph for edge classification by converting edges to nodes.
        """
        N = features.shape[0]
        E = np.count_nonzero(np.triu(adjacency, 1))
        
        # Create incidence matrix B (E x N)
        B = np.zeros((E, N))
        edge_idx = 0
        for i in range(N):
            for j in range(i+1, N):
                if adjacency[i, j] > 0:
                    B[edge_idx, i] = 1
                    B[edge_idx, j] = 1
                    edge_idx += 1
        
        # Augmented adjacency: [[A, B^T], [B, 0]]
        augmented_adj = np.zeros((N + E, N + E))
        augmented_adj[:N, :N] = adjacency
        augmented_adj[:N, N:] = B.T
        augmented_adj[N:, :N] = B
        
        # Augmented features
        augmented_features = np.zeros((N + E, features.shape[1]))
        augmented_features[:N, :] = features
        
        return augmented_adj, augmented_features
    
    def fit(self, 
            adjacency: np.ndarray, 
            features: np.ndarray, 
            labels: np.ndarray,
            train_mask: np.ndarray,
            val_mask: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            early_stopping: bool = True,
            patience: int = 10,
            verbose: bool = True):
        """
        Train the DCNN model using AdaGrad.
        """
        N, F = features.shape
        
        # Convert labels to one-hot if needed
        if len(labels.shape) == 1:
            self.n_classes = len(np.unique(labels))
            y_one_hot = np.zeros((N, self.n_classes))
            y_one_hot[np.arange(N), labels.astype(int)] = 1
        else:
            y_one_hot = labels
            self.n_classes = labels.shape[1]
        
        # Initialize weights if not already done
        if self.Wc is None:
            self._initialize_weights(F, self.n_classes)
        
        # Compute transition matrix and power series
        P = self._compute_transition_matrix(adjacency)
        P_star = self._compute_power_series(P, self.n_hops)
        
        # Compute diffusion-convolution representation
        if self.task_type == 'node':
            Z = self._diffusion_convolution_node(P_star, features, self.Wc)
            Z_flat = Z.reshape(Z.shape[0], -1)
        elif self.task_type == 'graph':
            Z = self._diffusion_convolution_graph(P_star, features, self.Wc)
            Z_flat = Z.flatten()[np.newaxis, :]  # Add batch dimension
        else:
            raise ValueError(f"Task type {self.task_type} not supported")
        
        # AdaGrad accumulators
        grad_squared_Wc = np.zeros_like(self.Wc)
        grad_squared_Wd = np.zeros_like(self.Wd)
        epsilon = 1e-8
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_weights = (self.Wc.copy(), self.Wd.copy())
        patience_counter = 0
        val_losses = []
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle training data
            if self.task_type == 'node':
                train_indices = np.where(train_mask)[0]
                np.random.shuffle(train_indices)
            else:
                train_indices = np.array([0])  # For graph classification
            
            epoch_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(train_indices), batch_size):
                batch_indices = train_indices[i:i+batch_size]
                
                # Forward pass
                if self.task_type == 'node':
                    Z_flat_batch = Z_flat[batch_indices]
                    batch_labels = y_one_hot[batch_indices]
                else:
                    Z_flat_batch = Z_flat
                    batch_labels = y_one_hot[0:1]  # Single graph label
                
                logits = Z_flat_batch @ self.Wd.T
                
                # Compute loss
                loss = self._hinge_loss(logits, batch_labels)
                epoch_loss += loss
                num_batches += 1
                
                # Backward pass
                d_logits = self._hinge_loss_gradient(logits, batch_labels)
                d_Wd = d_logits.T @ Z_flat_batch
                d_Z_flat = d_logits @ self.Wd
                
                # Reshape gradient for Wc
                if self.task_type == 'node':
                    # For node classification
                    Z_batch = Z[batch_indices]
                    d_Z = d_Z_flat.reshape(Z_batch.shape)
                    
                    # Compute gradient for Wc
                    # dWc = sum over batch and nodes of (dZ * activation_derivative * Z/Wc)
                    d_Wc = np.sum(d_Z * self.activation_derivative(Z_batch) * 
                                 (Z_batch / (self.Wc[np.newaxis, :, :] + 1e-8)), 
                                 axis=(0, 1))
                    
                else:
                    # For graph classification
                    d_Z = d_Z_flat.reshape(Z.shape)
                    d_Wc = np.sum(d_Z * self.activation_derivative(Z) * 
                                 (Z / (self.Wc + 1e-8)), 
                                 axis=0)
                
                # Update weights with AdaGrad
                grad_squared_Wc += d_Wc ** 2
                grad_squared_Wd += d_Wd ** 2
                
                self.Wc -= self.learning_rate * d_Wc / (np.sqrt(grad_squared_Wc) + epsilon)
                self.Wd -= self.learning_rate * d_Wd / (np.sqrt(grad_squared_Wd) + epsilon)
            
            # Validation
            if val_mask is not None:
                val_loss = self._evaluate_loss(Z_flat, y_one_hot, val_mask)
                val_losses.append(val_loss)
                
                # Early stopping
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = (self.Wc.copy(), self.Wd.copy())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch}")
                            self.Wc, self.Wd = best_weights
                            break
            
            if verbose and epoch % 10 == 0:
                train_acc = self._compute_accuracy(Z_flat, y_one_hot, train_mask)
                val_acc = self._compute_accuracy(Z_flat, y_one_hot, val_mask) if val_mask is not None else None
                print(f"Epoch {epoch}, Loss: {epoch_loss/num_batches:.4f}, "
                      f"Train Acc: {train_acc:.4f}" + 
                      (f", Val Acc: {val_acc:.4f}" if val_acc is not None else ""))
    
    def _evaluate_loss(self, Z_flat: np.ndarray, y_one_hot: np.ndarray, mask: np.ndarray) -> float:
        """Evaluate loss on masked data."""
        if self.task_type == 'node':
            logits = Z_flat[mask] @ self.Wd.T
            loss = self._hinge_loss(logits, y_one_hot[mask])
        else:
            logits = Z_flat @ self.Wd.T
            loss = self._hinge_loss(logits, y_one_hot[0:1])
        return loss
    
    def _compute_accuracy(self, Z_flat: np.ndarray, y_one_hot: np.ndarray, mask: np.ndarray) -> float:
        """Compute accuracy on masked data."""
        if self.task_type == 'node':
            logits = Z_flat[mask] @ self.Wd.T
            predictions = np.argmax(logits, axis=1)
            true_labels = np.argmax(y_one_hot[mask], axis=1)
        else:
            logits = Z_flat @ self.Wd.T
            predictions = np.argmax(logits, axis=1) if len(logits.shape) > 1 else np.argmax(logits)
            true_labels = np.argmax(y_one_hot[0])
        return np.mean(predictions == true_labels)
    
    def predict(self, 
                adjacency: np.ndarray, 
                features: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions.
        """
        P = self._compute_transition_matrix(adjacency)
        P_star = self._compute_power_series(P, self.n_hops)
        
        if self.task_type == 'node':
            Z = self._diffusion_convolution_node(P_star, features, self.Wc)
            Z_flat = Z.reshape(Z.shape[0], -1)
        else:
            Z = self._diffusion_convolution_graph(P_star, features, self.Wc)
            Z_flat = Z.flatten()[np.newaxis, :]
        
        if mask is not None and self.task_type == 'node':
            Z_flat = Z_flat[mask]
        
        logits = Z_flat @ self.Wd.T
        return np.argmax(logits, axis=1)
    
    def predict_proba(self, 
                     adjacency: np.ndarray, 
                     features: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make probabilistic predictions using softmax.
        """
        P = self._compute_transition_matrix(adjacency)
        P_star = self._compute_power_series(P, self.n_hops)
        
        if self.task_type == 'node':
            Z = self._diffusion_convolution_node(P_star, features, self.Wc)
            Z_flat = Z.reshape(Z.shape[0], -1)
        else:
            Z = self._diffusion_convolution_graph(P_star, features, self.Wc)
            Z_flat = Z.flatten()[np.newaxis, :]
        
        if mask is not None and self.task_type == 'node':
            Z_flat = Z_flat[mask]
        
        logits = Z_flat @ self.Wd.T
        return softmax(logits, axis=1)


# ============ Helper Functions for Data Loading ============

def generate_synthetic_graph(n_nodes: int = 100, n_features: int = 10, n_classes: int = 3):
    """
    Generate a synthetic graph dataset.
    
    Returns:
        adjacency: n_nodes x n_nodes adjacency matrix
        features: n_nodes x n_features feature matrix
        labels: n_nodes label vector
    """
    # Create random adjacency matrix (sparse, undirected)
    adjacency = np.random.rand(n_nodes, n_nodes) < 0.1
    adjacency = adjacency.astype(float)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0)
    
    # Create random features
    features = np.random.randn(n_nodes, n_features)
    
    # Create random labels (with some structure based on features)
    # This creates linearly separable data with some noise
    W = np.random.randn(n_features, n_classes)
    logits = features @ W
    probs = softmax(logits, axis=1)
    labels = np.array([np.random.choice(n_classes, p=probs[i]) for i in range(n_nodes)])
    
    return adjacency, features, labels

def create_masks(N: int, train_ratio: float = 0.5, val_ratio: float = 0.25, test_ratio: float = 0.25):
    """
    Create train/val/test masks.
    """
    indices = np.random.permutation(N)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    
    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    
    return train_mask, val_mask, test_mask


# ============ Example Usage ============

def example_node_classification():
    """Example of node classification on synthetic data."""
    print("=" * 60)
    print("DCNN Node Classification Example")
    print("=" * 60)
    
    # Generate synthetic data
    print("\nGenerating synthetic graph data...")
    n_nodes = 500
    n_features = 20
    n_classes = 3
    
    adjacency, features, labels = generate_synthetic_graph(n_nodes, n_features, n_classes)
    
    # Create masks
    train_mask, val_mask, test_mask = create_masks(n_nodes, train_ratio=0.5, val_ratio=0.25)
    
    print(f"Dataset size: {n_nodes} nodes")
    print(f"Features: {n_features}")
    print(f"Classes: {n_classes}")
    print(f"Edges: {np.count_nonzero(adjacency) // 2}")
    
    # Initialize model
    model = DCNN(
        n_hops=2,
        n_features=n_features,
        n_classes=n_classes,
        learning_rate=0.05,
        activation='tanh',
        task_type='node'
    )
    
    # Train model
    print("\n" + "-" * 40)
    print("Training DCNN...")
    print("-" * 40)
    
    start_time = time.time()
    model.fit(
        adjacency=adjacency,
        features=features,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        epochs=30,
        batch_size=32,
        early_stopping=True,
        patience=5,
        verbose=True
    )
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds")
    
    # Evaluate
    print("\n" + "-" * 40)
    print("Evaluation Results:")
    print("-" * 40)
    
    train_pred = model.predict(adjacency, features, train_mask)
    train_acc = np.mean(train_pred == labels[train_mask])
    print(f"Train Accuracy: {train_acc:.4f}")
    
    val_pred = model.predict(adjacency, features, val_mask)
    val_acc = np.mean(val_pred == labels[val_mask])
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    test_pred = model.predict(adjacency, features, test_mask)
    test_acc = np.mean(test_pred == labels[test_mask])
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Get probabilities
    test_probs = model.predict_proba(adjacency, features, test_mask)
    print(f"Test predictions shape: {test_probs.shape}")
    print(f"Sample probabilities: {test_probs[0]}")
    
    return model

def example_graph_classification():
    """Example of graph classification on synthetic data."""
    print("\n" + "=" * 60)
    print("DCNN Graph Classification Example")
    print("=" * 60)
    
    # Create multiple synthetic graphs
    n_graphs = 50
    n_classes = 2
    n_features = 10
    
    print(f"\nGenerating {n_graphs} synthetic graphs...")
    
    # Store all graphs in a list
    graph_data = []
    graph_labels = []
    
    for i in range(n_graphs):
        n_nodes = np.random.randint(20, 50)
        adj, features, labels = generate_synthetic_graph(n_nodes, n_features, n_classes)
        graph_data.append((adj, features))
        graph_labels.append(np.random.randint(0, 2))  # Random binary labels
    
    print(f"Average graph size: {np.mean([len(d[0]) for d in graph_data]):.1f} nodes")
    
    # For simplicity, we'll train on the first graph and test on the rest
    # Note: In practice, you'd batch graphs properly
    print("\nNote: Graph classification is demonstrated but full batching would require more complex data loading")
    
    # Train on one graph
    train_idx = 0
    train_adj, train_features = graph_data[train_idx]
    train_label = graph_labels[train_idx]
    
    # Create masks for graph classification (all nodes in the graph)
    N = train_adj.shape[0]
    train_mask = np.ones(N, dtype=bool)
    
    # Initialize model for graph classification
    model = DCNN(
        n_hops=2,
        n_features=n_features,
        n_classes=n_classes,
        learning_rate=0.05,
        activation='tanh',
        task_type='graph'
    )
    
    print("\n" + "-" * 40)
    print("Training DCNN for graph classification...")
    print("-" * 40)
    
    model.fit(
        adjacency=train_adj,
        features=train_features,
        labels=np.array([train_label]),
        train_mask=train_mask,
        epochs=30,
        batch_size=1,
        verbose=True
    )
    
    return model

if __name__ == "__main__":
    # Run node classification example
    node_model = example_node_classification()
    
    # Run graph classification example
    graph_model = example_graph_classification()
    
    print("\n" + "=" * 60)
    print("DCNN Implementation Complete")
    print("=" * 60)