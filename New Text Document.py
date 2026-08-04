import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.ndimage import zoom
import time

class MandelbrotKMeans:
    """
    K-Means clustering for Mandelbrot set approximation.
    Clusters image patches and reconstructs using cluster centroids.
    """
    
    def __init__(self, n_clusters=16, patch_size=8, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.patch_size = patch_size
        self.max_iters = max_iters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.patch_centroids = None
        
    def generate_mandelbrot(self, width=512, height=512, 
                           x_range=(-2.5, 1.5), y_range=(-1.5, 1.5),
                           max_iter=256):
        """
        Generate Mandelbrot set image.
        """
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Create coordinate grid
        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        
        # Mandelbrot iteration
        Z = np.zeros_like(C, dtype=complex)
        escape_time = np.zeros(C.shape, dtype=np.float32)
        mask = np.ones(C.shape, dtype=bool)
        
        for i in range(max_iter):
            Z[mask] = Z[mask] ** 2 + C[mask]
            mask_new = np.abs(Z) > 2
            escape_time[mask_new & mask] = i
            mask = mask & ~mask_new
            if not mask.any():
                break
        
        # Points inside set (never escaped)
        escape_time[~mask] = max_iter
        
        # Normalize to [0, 1] for better visualization
        self.mandelbrot_image = escape_time / max_iter
        
        # Also create binary mask for the set
        self.mandelbrot_binary = (escape_time >= max_iter).astype(np.float32)
        
        return self.mandelbrot_image, self.mandelbrot_binary
    
    def extract_patches(self, image):
        """
        Extract overlapping patches from image.
        """
        h, w = image.shape
        ps = self.patch_size
        
        # Pad image to handle borders
        pad_h = (ps - h % ps) % ps
        pad_w = (ps - w % ps) % ps
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        # Extract non-overlapping patches
        patches = []
        positions = []
        
        for i in range(0, padded.shape[0] - ps + 1, ps):
            for j in range(0, padded.shape[1] - ps + 1, ps):
                patch = padded[i:i+ps, j:j+ps]
                patches.append(patch.flatten())
                positions.append((i, j))
        
        return np.array(patches), positions, padded.shape
    
    def fit_clusters(self, image=None):
        """
        Fit K-Means to image patches.
        """
        if image is None:
            image = self.mandelbrot_image
        
        # Extract patches
        print(f"Extracting patches of size {self.patch_size}x{self.patch_size}...")
        patches, self.patch_positions, self.padded_shape = self.extract_patches(image)
        print(f"Extracted {len(patches)} patches")
        
        # Scale patches
        patches_scaled = self.scaler.fit_transform(patches)
        
        # Fit K-Means
        print(f"Clustering {len(patches)} patches into {self.n_clusters} clusters...")
        start_time = time.time()
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iters,
            random_state=self.random_state,
            n_init=10,
            verbose=0
        )
        self.labels = self.kmeans.fit_predict(patches_scaled)
        self.patch_centroids = self.kmeans.cluster_centers_
        
        # Inverse transform centroids
        self.centroid_original = self.scaler.inverse_transform(self.patch_centroids)
        
        print(f"Clustering complete in {time.time() - start_time:.2f}s")
        
        return self.labels, self.centroid_original
    
    def reconstruct_image(self, width=None, height=None):
        """
        Reconstruct image from cluster centroids.
        """
        if width is None:
            width = self.padded_shape[1]
        if height is None:
            height = self.padded_shape[0]
        
        ps = self.patch_size
        
        # Create empty image
        reconstructed = np.zeros((height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)
        
        # Place each reconstructed patch
        for idx, (i, j) in enumerate(self.patch_positions):
            centroid = self.centroid_original[self.labels[idx]]
            patch = centroid.reshape(ps, ps)
            
            reconstructed[i:i+ps, j:j+ps] += patch
            count[i:i+ps, j:j+ps] += 1
        
        # Average overlapping regions
        count[count == 0] = 1
        reconstructed = reconstructed / count
        
        # Crop to original size
        if self.padded_shape != (height, width):
            h, w = self.mandelbrot_image.shape
            reconstructed = reconstructed[:h, :w]
        
        return reconstructed
    
    def compute_error(self, original, reconstructed):
        """
        Compute reconstruction error metrics.
        """
        mse = np.mean((original - reconstructed) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Structural similarity (simplified)
        mean_original = np.mean(original)
        mean_reconstructed = np.mean(reconstructed)
        std_original = np.std(original)
        std_reconstructed = np.std(reconstructed)
        cov = np.mean((original - mean_original) * (reconstructed - mean_reconstructed))
        ssim = (2 * mean_original * mean_reconstructed + 0.01) * (2 * cov + 0.03) / \
               ((mean_original**2 + mean_reconstructed**2 + 0.01) * 
                (std_original**2 + std_reconstructed**2 + 0.03))
        
        # Binary accuracy (inside vs outside set)
        binary_original = (original > 0.5).astype(int)
        binary_reconstructed = (reconstructed > 0.5).astype(int)
        binary_accuracy = np.mean(binary_original == binary_reconstructed)
        
        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim,
            'binary_accuracy': binary_accuracy,
            'compression_ratio': len(self.patch_positions) / self.n_clusters
        }


def visualize_approximation(model, width=400, height=400):
    """
    Visualize original vs approximated Mandelbrot set.
    """
    # Generate original
    original, binary = model.generate_mandelbrot(width=width, height=height)
    
    # Fit and reconstruct
    model.fit_clusters(original)
    reconstructed = model.reconstruct_image()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original Mandelbrot
    im1 = axes[0, 0].imshow(original, cmap='hot', aspect='equal')
    axes[0, 0].set_title('Original Mandelbrot Set')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], label='Escape Time')
    
    # Reconstructed
    im2 = axes[0, 1].imshow(reconstructed, cmap='hot', aspect='equal')
    axes[0, 1].set_title(f'K-Means Reconstruction ({model.n_clusters} clusters, {model.patch_size}x{model.patch_size} patches)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], label='Escape Time')
    
    # Difference
    diff = np.abs(original - reconstructed)
    im3 = axes[0, 2].imshow(diff, cmap='plasma', aspect='equal')
    axes[0, 2].set_title('Reconstruction Error')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], label='Error')
    
    # Binary comparison
    binary_orig = (original > 0.5).astype(int)
    binary_recon = (reconstructed > 0.5).astype(int)
    
    axes[1, 0].imshow(binary_orig, cmap='gray', aspect='equal')
    axes[1, 0].set_title('Original Binary (Mandelbrot Set)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(binary_recon, cmap='gray', aspect='equal')
    axes[1, 1].set_title('Reconstructed Binary')
    axes[1, 1].axis('off')
    
    # Metrics
    metrics = model.compute_error(original, reconstructed)
    axes[1, 2].axis('off')
    text = f"PSNR: {metrics['psnr']:.2f} dB\n"
    text += f"MSE: {metrics['mse']:.6f}\n"
    text += f"SSIM: {metrics['ssim']:.4f}\n"
    text += f"Binary Accuracy: {metrics['binary_accuracy']:.4f}\n"
    text += f"Compression: {metrics['compression_ratio']:.1f}x\n"
    text += f"Patches: {len(model.patch_positions)}\n"
    text += f"Clusters: {model.n_clusters}"
    axes[1, 2].text(0.1, 0.5, text, fontsize=14, verticalalignment='center')
    axes[1, 2].set_title('Performance Metrics')
    
    plt.tight_layout()
    plt.show()
    
    return metrics


def experiment_parameters():
    """
    Experiment with different patch sizes and cluster counts.
    """
    results = []
    
    # Test different patch sizes
    patch_sizes = [4, 8, 16, 32]
    n_clusters_list = [8, 16, 32, 64]
    
    for ps in patch_sizes:
        for n_clusters in n_clusters_list:
            print(f"\n{'='*50}")
            print(f"Patch Size: {ps}x{ps}, Clusters: {n_clusters}")
            print(f"{'='*50}")
            
            model = MandelbrotKMeans(
                n_clusters=n_clusters,
                patch_size=ps,
                random_state=42
            )
            
            # Generate small image for testing
            model.generate_mandelbrot(width=256, height=256)
            model.fit_clusters()
            reconstructed = model.reconstruct_image()
            
            metrics = model.compute_error(model.mandelbrot_image, reconstructed)
            metrics['patch_size'] = ps
            metrics['n_clusters'] = n_clusters
            results.append(metrics)
            
            print(f"PSNR: {metrics['psnr']:.2f} dB")
            print(f"Binary Acc: {metrics['binary_accuracy']:.4f}")
    
    return results


def multi_scale_approximation(model, scales=[1, 2, 4]):
    """
    Multi-scale approximation for better quality.
    """
    original = model.mandelbrot_image
    
    # Create coarse approximations at different scales
    approximations = []
    
    for scale in scales:
        # Downsample
        h, w = original.shape
        new_h, new_w = h // scale, w // scale
        downsampled = zoom(original, (1/scale, 1/scale), order=1)
        
        # Fit K-Means on downsampled
        model.fit_clusters(downsampled)
        
        # Reconstruct at original size
        recon = model.reconstruct_image(width=w, height=h)
        approximations.append(recon)
    
    # Ensemble reconstruction
    final = np.mean(approximations, axis=0)
    
    return final


# ---------- Main Execution ----------
if __name__ == "__main__":
    print("K-Means Mandelbrot Set Approximation (Image Patch Compression)")
    print("=" * 70)
    
    # Create model
    model = MandelbrotKMeans(
        n_clusters=32,
        patch_size=8,
        max_iters=100,
        random_state=42
    )
    
    # Generate Mandelbrot
    print("Generating Mandelbrot set...")
    model.generate_mandelbrot(width=512, height=512)
    
    # Visualize approximation
    print("\nVisualizing approximation...")
    metrics = visualize_approximation(model, width=400, height=400)
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Patch Size: {model.patch_size}x{model.patch_size}")
    print(f"Number of Clusters: {model.n_clusters}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"Binary Accuracy: {metrics['binary_accuracy']:.4f}")
    print(f"Compression Ratio: {metrics['compression_ratio']:.1f}x")
    
    # Multi-scale demo
    print("\n" + "=" * 70)
    print("Multi-Scale Approximation Demo")
    print("=" * 70)
    model.generate_mandelbrot(width=512, height=512)
    multi_scale = multi_scale_approximation(model, scales=[1, 2, 4])
    
    # Visualize multi-scale result
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(model.mandelbrot_image, cmap='hot', aspect='equal')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(multi_scale, cmap='hot', aspect='equal')
    axes[1].set_title('Multi-Scale Reconstruction')
    axes[1].axis('off')
    
    axes[2].imshow(np.abs(model.mandelbrot_image - multi_scale), cmap='plasma', aspect='equal')
    axes[2].set_title('Error')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Parameter experiments
    print("\n" + "=" * 70)
    print("Running parameter experiments...")
    print("=" * 70)
    exp_results = experiment_parameters()
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSNR vs parameters
    for ps in [4, 8, 16, 32]:
        mask = [r['patch_size'] == ps for r in exp_results]
        psnr_values = [r['psnr'] for r in exp_results if r['patch_size'] == ps]
        n_clusters = [r['n_clusters'] for r in exp_results if r['patch_size'] == ps]
        axes[0].plot(n_clusters, psnr_values, 'o-', label=f'{ps}x{ps}')
    
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR vs Parameters')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Binary accuracy vs parameters
    for ps in [4, 8, 16, 32]:
        mask = [r['patch_size'] == ps for r in exp_results]
        bin_acc = [r['binary_accuracy'] for r in exp_results if r['patch_size'] == ps]
        n_clusters = [r['n_clusters'] for r in exp_results if r['patch_size'] == ps]
        axes[1].plot(n_clusters, bin_acc, 's-', label=f'{ps}x{ps}')
    
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Binary Accuracy')
    axes[1].set_title('Shape Accuracy vs Parameters')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()