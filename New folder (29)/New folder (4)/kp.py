import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import SparsePCA


def mandelbrot_set(width, height, x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5, max_iter=100):
    """
    Generate Mandelbrot set data.
    Returns the iteration counts for each pixel.
    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    
    Z = np.zeros_like(C, dtype=complex)
    iterations = np.zeros(C.shape, dtype=int)
    
    for i in range(max_iter):
        mask = np.abs(Z) <= 2.0
        Z[mask] = Z[mask] ** 2 + C[mask]
        iterations[mask] = i
    
    return iterations


def prepare_data_for_sparse_pca(mandelbrot_data, patch_size=8):
    """
    Prepare Mandelbrot set data for SparsePCA by creating patches.
    This mimics how neural networks process image data in patches.
    """
    height, width = mandelbrot_data.shape
    
    # Normalize data to [0, 1]
    normalized = mandelbrot_data.astype(float) / mandelbrot_data.max()
    
    # Extract patches
    patches = []
    for i in range(0, height - patch_size + 1, patch_size // 2):
        for j in range(0, width - patch_size + 1, patch_size // 2):
            patch = normalized[i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())
    
    return np.array(patches), normalized


def sparse_pca_neural_network(X, n_components=64, alpha=0.1):
    """
    Use SparsePCA as a neural network alternative.
    SparsePCA learns sparse feature representations similar to sparse autoencoders.
    """
    print(f"Training SparsePCA neural network alternative with {n_components} components...")
    spca = SparsePCA(n_components=n_components, alpha=alpha, max_iter=1000, random_state=42)
    X_transformed = spca.fit_transform(X)
    
    print(f"Learned {len(spca.components_)} sparse components")
    print(f"Input shape: {X.shape}, Output shape: {X_transformed.shape}")
    
    return spca, X_transformed


def reconstruct_mandelbrot(spca, X_transformed, original_shape, patch_size=8):
    """
    Reconstruct the Mandelbrot set from sparse PCA representation.
    """
    # Transform back to original space
    X_reconstructed = spca.inverse_transform(X_transformed)
    
    # Reconstruct image from patches
    height, width = original_shape
    reconstructed = np.zeros(original_shape)
    patch_count = np.zeros(original_shape)
    
    idx = 0
    for i in range(0, height - patch_size + 1, patch_size // 2):
        for j in range(0, width - patch_size + 1, patch_size // 2):
            patch = X_reconstructed[idx].reshape(patch_size, patch_size)
            reconstructed[i:i+patch_size, j:j+patch_size] += patch
            patch_count[i:i+patch_size, j:j+patch_size] += 1
            idx += 1
    
    # Average overlapping patches
    patch_count[patch_count == 0] = 1
    reconstructed = reconstructed / patch_count
    
    return reconstructed


def visualize_results(original, reconstructed, components, spca):
    """
    Visualize the original Mandelbrot set, reconstructed version, and learned components.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original Mandelbrot set
    axes[0, 0].imshow(original, cmap='hot', origin='lower')
    axes[0, 0].set_title('Original Mandelbrot Set', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Reconstructed Mandelbrot set
    axes[0, 1].imshow(reconstructed, cmap='hot', origin='lower')
    axes[0, 1].set_title('Reconstructed (SparsePCA Neural Network)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Difference
    difference = np.abs(original - reconstructed)
    axes[1, 0].imshow(difference, cmap='viridis', origin='lower')
    axes[1, 0].set_title('Reconstruction Error', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Visualize some learned sparse components
    n_show = min(16, len(spca.components_))
    component_grid = int(np.sqrt(n_show))
    
    # Reshape components back to patch size
    patch_size = int(np.sqrt(len(spca.components_[0])))
    
    component_image = np.zeros((component_grid * patch_size, component_grid * patch_size))
    for i in range(component_grid):
        for j in range(component_grid):
            idx = i * component_grid + j
            if idx < n_show:
                component_image[i*patch_size:(i+1)*patch_size, 
                               j*patch_size:(j+1)*patch_size] = \
                    spca.components_[idx].reshape(patch_size, patch_size)
    
    axes[1, 1].imshow(component_image, cmap='RdBu_r', origin='lower')
    axes[1, 1].set_title(f'Learned Sparse Components (showing {n_show}/{len(spca.components_)})', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('mandelbrot_sparse_pca_neural_network.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'mandelbrot_sparse_pca_neural_network.png'")
    plt.show()


def main():
    """
    Main function to run the SparsePCA neural network alternative on Mandelbrot set.
    """
    print("=" * 60)
    print("SparsePCA Neural Network Alternative for Mandelbrot Set")
    print("=" * 60)
    
    # Generate Mandelbrot set
    print("\n1. Generating Mandelbrot set...")
    width, height = 400, 400
    mandelbrot_data = mandelbrot_set(width, height, max_iter=100)
    
    # Prepare data for SparsePCA (create patches)
    print("\n2. Preparing data patches for neural network processing...")
    X_patches, normalized_mandelbrot = prepare_data_for_sparse_pca(mandelbrot_data, patch_size=8)
    print(f"   Created {X_patches.shape[0]} patches of size {X_patches.shape[1]}")
    
    # Train SparsePCA as neural network alternative
    print("\n3. Training SparsePCA neural network alternative...")
    n_components = 64  # Number of sparse features to learn
    alpha = 0.1  # Sparsity control parameter
    spca, X_transformed = sparse_pca_neural_network(X_patches, n_components=n_components, alpha=alpha)
    
    # Reconstruct Mandelbrot set
    print("\n4. Reconstructing Mandelbrot set from learned representation...")
    reconstructed = reconstruct_mandelbrot(spca, X_transformed, normalized_mandelbrot.shape, patch_size=8)
    
    # Visualize results
    print("\n5. Visualizing results...")
    visualize_results(normalized_mandelbrot, reconstructed, X_transformed, spca)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics:")
    print(f"  Original data shape: {mandelbrot_data.shape}")
    print(f"  Number of patches: {X_patches.shape[0]}")
    print(f"  Patch size: {int(np.sqrt(X_patches.shape[1]))}x{int(np.sqrt(X_patches.shape[1]))}")
    print(f"  Learned components: {len(spca.components_)}")
    print(f"  Compression ratio: {X_patches.shape[1] / n_components:.2f}x")
    print(f"  Mean reconstruction error: {np.mean(np.abs(normalized_mandelbrot - reconstructed)):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()