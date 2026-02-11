"""
Mandelbrot Set Visualization using Graph Neural Network with Kernel PCA
Uses sklearn's KernelPCA (no PyTorch)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, diags
import warnings
warnings.filterwarnings('ignore')


class GraphNeuralNetworkKernelPCA:
    """
    Graph Neural Network using Kernel PCA for message passing
    """
    
    def __init__(self, n_components=50, kernel='rbf', gamma=0.1, n_iter=3, sample_size=5000, batch_size=10000):
        """
        Initialize GNN with Kernel PCA
        
        Args:
            n_components: Number of components for Kernel PCA
            kernel: Kernel type ('rbf', 'poly', 'sigmoid', 'cosine')
            gamma: Kernel coefficient for RBF
            n_iter: Number of message passing iterations
            sample_size: Number of samples to use for fitting KernelPCA (to avoid memory issues)
            batch_size: Batch size for transforming large datasets
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.n_iter = n_iter
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.kpca = None
        self.scaler = None
        self.sample_indices = None
        self.current_iteration = 0
        
    def create_adjacency_matrix(self, height, width):
        """
        Create sparse adjacency matrix for pixel graph (4-connected neighbors)
        """
        n_nodes = height * width
        rows = []
        cols = []
        
        for i in range(height):
            for j in range(width):
                node_idx = i * width + j
                
                # Add edges to 4-connected neighbors
                neighbors = [
                    (i-1, j),  # up
                    (i+1, j),  # down
                    (i, j-1),  # left
                    (i, j+1),  # right
                ]
                
                for ni, nj in neighbors:
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_idx = ni * width + nj
                        rows.append(node_idx)
                        cols.append(neighbor_idx)
        
        # Create sparse matrix (symmetric, so we add both directions)
        data = np.ones(len(rows), dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
        
        # Make symmetric (since we only added one direction)
        adj = adj + adj.T
        adj.data = np.ones(adj.nnz, dtype=np.float32)  # Set all to 1
        
        # Normalize adjacency matrix (row normalization)
        degree = np.array(adj.sum(axis=1)).flatten()
        degree[degree == 0] = 1  # Avoid division by zero
        degree_inv = 1.0 / degree
        degree_inv_diag = diags(degree_inv, format='csr')
        adj_normalized = degree_inv_diag @ adj
        
        return adj_normalized
    
    
    def message_passing(self, features, adjacency, iteration):
        """
        Message passing using Kernel PCA transformation
        Uses sampling to avoid memory issues with large datasets
        Refits scaler and KernelPCA each iteration to handle changing dimensions
        """
        # Aggregate neighbor features (sparse matrix multiplication)
        aggregated = adjacency @ features
        
        # Convert to dense if needed (aggregated is dense after sparse @ dense)
        if hasattr(aggregated, 'toarray'):
            aggregated = aggregated.toarray()
        
        # Combine with self features
        combined = np.hstack([features, aggregated])
        
        # Sample a subset for fitting to avoid memory issues
        n_samples = combined.shape[0]
        n_features = combined.shape[1]
        sample_size = min(self.sample_size, n_samples)
        
        # Randomly sample indices
        sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
        combined_sample = combined[sample_indices]
        
        # Create new scaler and KernelPCA for this iteration (dimensions may have changed)
        scaler = StandardScaler()
        combined_sample_scaled = scaler.fit_transform(combined_sample)
        
        kpca = KernelPCA(
            n_components=min(self.n_components, combined_sample_scaled.shape[1]),
            kernel=self.kernel,
            gamma=self.gamma,
            fit_inverse_transform=False  # Don't need inverse for transform
        )
        
        # Fit on sample
        print(f"  Iteration {iteration + 1}: Fitting KernelPCA on {sample_size} samples ({n_features} features)...")
        kpca.fit(combined_sample_scaled)
        print(f"  Iteration {iteration + 1}: Transforming all {n_samples} samples in batches...")
        
        # Transform all data in batches to avoid memory issues
        combined_scaled = scaler.transform(combined)
        transformed = self._transform_in_batches_with_kpca(combined_scaled, kpca)
        
        return transformed
    
    def _transform_in_batches_with_kpca(self, X, kpca):
        """
        Transform data in batches using a specific KernelPCA instance
        """
        n_samples = X.shape[0]
        if n_samples <= self.batch_size:
            return kpca.transform(X)
        
        # Process in batches
        batches = []
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        for i in range(0, n_samples, self.batch_size):
            batch_idx = i // self.batch_size + 1
            batch = X[i:i+self.batch_size]
            batch_transformed = kpca.transform(batch)
            batches.append(batch_transformed)
            if batch_idx % 5 == 0 or batch_idx == n_batches:
                print(f"    Transformed batch {batch_idx}/{n_batches} ({min(batch_idx*self.batch_size, n_samples)}/{n_samples} samples)")
        
        return np.vstack(batches)
    
    def forward(self, node_features, adjacency):
        """
        Forward pass through GNN layers
        """
        features = node_features.copy()
        
        for iteration in range(self.n_iter):
            print(f"\n--- GNN Layer {iteration + 1}/{self.n_iter} ---")
            # Message passing with Kernel PCA
            features = self.message_passing(features, adjacency, iteration)
            
            # Apply activation (ReLU-like)
            features = np.maximum(features, 0)
            
            # Normalize features
            norm = np.linalg.norm(features, axis=1, keepdims=True)
            norm[norm == 0] = 1
            features = features / norm
        
        return features


def compute_mandelbrot_set(width, height, max_iter=100, x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5):
    """
    Compute Mandelbrot set escape times
    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    
    # Initialize arrays
    Z = np.zeros_like(C)
    escape_time = np.zeros(C.shape, dtype=np.int32)
    
    # Compute Mandelbrot iterations
    for i in range(max_iter):
        mask = np.abs(Z) <= 2.0
        Z[mask] = Z[mask] ** 2 + C[mask]
        escape_time[mask] = i
    
    return escape_time, C


def create_node_features(escape_time, complex_values):
    """
    Create initial node features from Mandelbrot set
    """
    height, width = escape_time.shape
    
    # Normalize escape time
    escape_norm = escape_time.astype(np.float32) / escape_time.max()
    
    # Extract real and imaginary parts
    real_part = np.real(complex_values).astype(np.float32)
    imag_part = np.imag(complex_values).astype(np.float32)
    
    # Normalize complex parts
    real_norm = (real_part - real_part.min()) / (real_part.max() - real_part.min() + 1e-8)
    imag_norm = (imag_part - imag_part.min()) / (imag_part.max() - imag_part.min() + 1e-8)
    
    # Create feature matrix: [escape_time, real_part, imag_part, magnitude]
    magnitude = np.abs(complex_values).astype(np.float32)
    mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    
    features = np.stack([
        escape_norm.flatten(),
        real_norm.flatten(),
        imag_norm.flatten(),
        mag_norm.flatten()
    ], axis=1)
    
    return features


def visualize_results(original, processed, gnn_output, width, height):
    """
    Visualize original Mandelbrot set, processed version, and GNN output
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original Mandelbrot set
    axes[0].imshow(original, cmap='hot', origin='lower')
    axes[0].set_title('Original Mandelbrot Set', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Processed version
    axes[1].imshow(processed, cmap='viridis', origin='lower')
    axes[1].set_title('GNN Processed (Kernel PCA)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # GNN output features visualization (first component)
    gnn_vis = gnn_output[:, 0].reshape(height, width)
    axes[2].imshow(gnn_vis, cmap='plasma', origin='lower')
    axes[2].set_title('GNN Output Features (Component 0)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('mandelbrot_gnn_kernelpca.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'mandelbrot_gnn_kernelpca.png'")
    plt.show()


def main():
    """
    Main function to run Mandelbrot set GNN with Kernel PCA
    """
    print("=" * 60)
    print("Mandelbrot Set Graph Neural Network with Kernel PCA")
    print("=" * 60)
    
    # Parameters
    width, height = 400, 400
    max_iter = 100
    
    print(f"\nComputing Mandelbrot set ({width}x{height})...")
    escape_time, complex_values = compute_mandelbrot_set(width, height, max_iter)
    
    print("Creating graph structure...")
    gnn = GraphNeuralNetworkKernelPCA(
        n_components=32,
        kernel='rbf',
        gamma=0.01,
        n_iter=3,
        sample_size=5000,  # Fit KernelPCA on 5000 samples to avoid memory issues
        batch_size=10000   # Transform in batches of 10000
    )
    
    # Create adjacency matrix for pixel graph
    adjacency = gnn.create_adjacency_matrix(height, width)
    n_edges = adjacency.nnz // 2  # Divide by 2 since it's symmetric
    print(f"Graph created: {adjacency.shape[0]} nodes, {n_edges} edges")
    print(f"Adjacency matrix memory: {adjacency.data.nbytes / 1024**2:.2f} MB (sparse)")
    
    # Create node features
    print("Creating node features...")
    node_features = create_node_features(escape_time, complex_values)
    print(f"Node features shape: {node_features.shape}")
    
    # Forward pass through GNN
    print("\nRunning Graph Neural Network with Kernel PCA...")
    print(f"  - Kernel: {gnn.kernel}")
    print(f"  - Components: {gnn.n_components}")
    print(f"  - Iterations: {gnn.n_iter}")
    
    gnn_output = gnn.forward(node_features, adjacency)
    print(f"GNN output shape: {gnn_output.shape}")
    
    # Create processed visualization
    # Use first component of GNN output
    processed = gnn_output[:, 0].reshape(height, width)
    
    # Normalize for visualization
    processed = (processed - processed.min()) / (processed.max() - processed.min() + 1e-8)
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_results(escape_time, processed, gnn_output, width, height)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
