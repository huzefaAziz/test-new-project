import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA

def mandelbrot_iteration(c, max_iter=100):
    """
    Calculate Mandelbrot set iteration count for a complex number c.
    Returns the number of iterations before divergence.
    """
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def generate_mandelbrot_data(width=800, height=600, max_iter=100, 
                            x_min=-2.5, x_max=1.5, y_min=-2.0, y_max=2.0):
    """
    Generate Mandelbrot set data as a 2D array using vectorized operations.
    """
    # Create coordinate grids
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    
    # Convert to complex plane
    C = X + 1j * Y
    
    # Vectorized Mandelbrot calculation
    z = np.zeros_like(C)
    mandelbrot_data = np.zeros(C.shape, dtype=int)
    
    for n in range(max_iter):
        mask = np.abs(z) <= 2
        mandelbrot_data[mask] = n
        z[mask] = z[mask] * z[mask] + C[mask]
    
    return mandelbrot_data, X, Y

def extract_patches(data, patch_size=8):
    """
    Extract overlapping patches from the image.
    Each patch becomes a sample with patch_size*patch_size features.
    """
    height, width = data.shape
    patches = []
    patch_positions = []
    
    stride = patch_size // 2  # Overlapping patches
    
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            patch = data[i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())
            patch_positions.append((i, j))
    
    return np.array(patches), patch_positions, patch_size, stride

def reconstruct_from_patches(patches, patch_positions, patch_size, stride, height, width):
    """
    Reconstruct image from patches using averaging for overlapping regions.
    """
    reconstructed = np.zeros((height, width))
    count = np.zeros((height, width))
    
    for patch, (i, j) in zip(patches, patch_positions):
        patch_2d = patch.reshape(patch_size, patch_size)
        h_end = min(i + patch_size, height)
        w_end = min(j + patch_size, width)
        p_h = h_end - i
        p_w = w_end - j
        
        reconstructed[i:h_end, j:w_end] += patch_2d[:p_h, :p_w]
        count[i:h_end, j:w_end] += 1
    
    # Avoid division by zero
    count[count == 0] = 1
    reconstructed = reconstructed / count
    
    return reconstructed

def apply_incremental_pca(data, n_components=50, patch_size=8, batch_size=1000):
    """
    Apply IncrementalPCA to compress and reconstruct Mandelbrot set data.
    This acts as a neural network alternative for dimensionality reduction.
    Uses image patches as features to learn patterns in the Mandelbrot set.
    """
    height, width = data.shape
    
    # Extract patches from the image
    print(f"Extracting {patch_size}x{patch_size} patches...")
    patches, patch_positions, patch_size, stride = extract_patches(data, patch_size)
    n_features = patches.shape[1]
    
    print(f"Extracted {len(patches)} patches with {n_features} features each")
    
    # Initialize IncrementalPCA
    n_comp = min(n_components, n_features, len(patches))
    ipca = IncrementalPCA(n_components=n_comp, batch_size=batch_size)
    
    # Fit the model incrementally
    print(f"Fitting IncrementalPCA with {n_comp} components...")
    n_samples = len(patches)
    
    # Partial fit on batches
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = patches[i:end_idx]
        ipca.partial_fit(batch)
    
    # Transform to lower-dimensional space
    print("Transforming patches to lower-dimensional space...")
    transformed_patches = []
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = patches[i:end_idx]
        transformed = ipca.transform(batch)
        transformed_patches.append(transformed)
    
    transformed_patches = np.vstack(transformed_patches)
    
    # Reconstruct patches back to original space
    print("Reconstructing patches...")
    reconstructed_patches = []
    for i in range(0, len(transformed_patches), batch_size):
        end_idx = min(i + batch_size, len(transformed_patches))
        batch = transformed_patches[i:end_idx]
        reconstructed = ipca.inverse_transform(batch)
        reconstructed_patches.append(reconstructed)
    
    reconstructed_patches = np.vstack(reconstructed_patches)
    
    # Reconstruct image from patches
    print("Reconstructing image from patches...")
    reconstructed_2d = reconstruct_from_patches(
        reconstructed_patches, patch_positions, patch_size, stride, height, width
    )
    
    return reconstructed_2d, ipca, transformed_patches

def visualize_mandelbrot(original_data, reconstructed_data, X, Y):
    """
    Visualize original and reconstructed Mandelbrot sets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original Mandelbrot set
    im1 = axes[0].imshow(original_data, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                         cmap='hot', origin='lower', interpolation='bilinear')
    axes[0].set_title('Original Mandelbrot Set', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Real')
    axes[0].set_ylabel('Imaginary')
    plt.colorbar(im1, ax=axes[0])
    
    # Reconstructed Mandelbrot set using IncrementalPCA
    im2 = axes[1].imshow(reconstructed_data, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                         cmap='hot', origin='lower', interpolation='bilinear')
    axes[1].set_title('Reconstructed Mandelbrot Set (IncrementalPCA)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Real')
    axes[1].set_ylabel('Imaginary')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('mandelbrot_incremental_pca.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'mandelbrot_incremental_pca.png'")
    plt.show()

def main():
    """
    Main function to generate Mandelbrot set and apply IncrementalPCA.
    """
    print("=" * 60)
    print("Mandelbrot Set using IncrementalPCA (Neural Network Alternative)")
    print("=" * 60)
    
    # Parameters
    width, height = 400, 300  # Reduced for faster computation
    max_iter = 100
    n_components = 50  # Number of principal components
    batch_size = 1000  # Batch size for IncrementalPCA
    
    # Generate Mandelbrot set data
    print("\nGenerating Mandelbrot set data...")
    mandelbrot_data, X, Y = generate_mandelbrot_data(
        width=width, height=height, max_iter=max_iter
    )
    print(f"Generated Mandelbrot set: {height}x{width} pixels")
    
    # Apply IncrementalPCA
    print("\nApplying IncrementalPCA...")
    patch_size = 8
    reconstructed_data, ipca, transformed = apply_incremental_pca(
        mandelbrot_data, n_components=n_components, patch_size=patch_size, batch_size=batch_size
    )
    
    # Calculate compression ratio and explained variance
    original_size = mandelbrot_data.size
    n_patches = transformed.shape[0]
    patch_features = patch_size * patch_size
    compressed_size = transformed.size  # n_patches * n_components
    compression_ratio = (n_patches * patch_features) / compressed_size
    explained_variance = np.sum(ipca.explained_variance_ratio_)
    
    print(f"\nCompression Statistics:")
    print(f"  Original size: {original_size} pixels")
    print(f"  Number of patches: {n_patches}")
    print(f"  Patch size: {patch_size}x{patch_size} = {patch_features} features")
    print(f"  Compressed size: {compressed_size} values ({n_patches} patches × {ipca.n_components} components)")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_mandelbrot(mandelbrot_data, reconstructed_data, X, Y)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
