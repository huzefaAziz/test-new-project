"""
Mandelbrot Set with K-Means Neural Network Clustering
======================================================

This script applies the K-Means Neural Network architecture (from kmeans_nn_architecture.html)
to the Mandelbrot set. Instead of traditional smooth coloring, the Mandelbrot escape-time data
is clustered using K-Means, producing a segmented fractal visualization where regions with
similar escape behavior are grouped and colored by cluster assignment.

Architecture (mirrors the HTML NN):
  Input Layer  →  Distance Layer  →  Centroid Layer  →  Assignment Layer
  (features)      (‖x - cₖ‖²)       (cluster centers)   (argmin → label)

Features per pixel: [real, imag, escape_iterations, |z_final|]
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys
import time
import sys


# ─────────────────────────── Configuration ────────────────────────────

# Mandelbrot parameters
WIDTH, HEIGHT = 1200, 900
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25
MAX_ITER = 256

# K-Means parameters (matching the HTML architecture)
K = 8                # Number of clusters
KMEANS_MAX_ITER = 100  # Max K-Means iterations
CONVERGENCE_THRESHOLD = 0.1  # Centroid movement threshold

# Output
OUTPUT_FILE = "mandelbrot_kmeans.png"


# ───────────────────── Mandelbrot Set Computation ─────────────────────

def compute_mandelbrot(width, height, x_min, x_max, y_min, y_max, max_iter):
    """
    Compute the Mandelbrot set escape-time values and final |z| magnitudes.
    Returns:
        iterations: 2D array of escape iteration counts
        magnitudes: 2D array of |z| at escape (or at max_iter)
        real_coords: 2D array of real part of c
        imag_coords: 2D array of imaginary part of c
    """
    print(f"Computing Mandelbrot set ({width}x{height}, max_iter={max_iter})...")
    t0 = time.time()

    # Create coordinate grids
    real = np.linspace(x_min, x_max, width)
    imag = np.linspace(y_min, y_max, height)
    real_coords, imag_coords = np.meshgrid(real, imag)
    c = real_coords + 1j * imag_coords

    # Initialize z and tracking arrays
    z = np.zeros_like(c, dtype=np.complex128)
    iterations = np.zeros(c.shape, dtype=np.float64)
    magnitudes = np.zeros(c.shape, dtype=np.float64)
    mask = np.ones(c.shape, dtype=bool)  # True = still iterating

    for i in range(max_iter):
        z[mask] = z[mask] ** 2 + c[mask]
        escaped = mask & (np.abs(z) > 2.0)
        iterations[escaped] = i + 1
        magnitudes[escaped] = np.abs(z[escaped])
        mask[escaped] = False

    # Points that never escaped get max_iter
    iterations[mask] = max_iter
    magnitudes[mask] = np.abs(z[mask])

    elapsed = time.time() - t0
    print(f"  Mandelbrot computed in {elapsed:.2f}s")
    print(f"  Interior points (max_iter): {np.sum(iterations == max_iter)}")
    print(f"  Escaped points: {np.sum(iterations < max_iter)}")

    return iterations, magnitudes, real_coords, imag_coords


# ───────────────────── K-Means NN Architecture ────────────────────────

class DataPoint:
    """Mirrors the DataPoint class from the HTML K-Means NN."""
    __slots__ = ['features', 'cluster']

    def __init__(self, features):
        self.features = np.array(features, dtype=np.float64)
        self.cluster = -1


class Centroid:
    """Mirrors the Centroid class from the HTML K-Means NN."""
    __slots__ = ['position', 'prev_position', 'index']

    def __init__(self, position, index):
        self.position = np.array(position, dtype=np.float64)
        self.prev_position = self.position.copy()
        self.index = index


class KMeansNeuralNetwork:
    """
    K-Means Neural Network Architecture
    ====================================
    Implements the same algorithm as the HTML visualization:

    Layer 1 - Input Layer:
        Receives feature vectors [real, imag, normalized_iterations, normalized_magnitude]

    Layer 2 - Distance Calculation Layer:
        Computes Euclidean distance from each input to each centroid:
        D_k = sqrt(sum((x_i - c_ki)^2))

    Layer 3 - Centroid Layer:
        Stores current cluster center positions, updated each iteration
        by averaging all points assigned to each cluster.

    Layer 4 - Assignment (Output) Layer:
        Applies argmin over distances to assign cluster labels:
        cluster = argmin_k(D_k)

    Convergence: when all centroid movements < threshold.
    """

    def __init__(self, k, max_iter=100, convergence_threshold=0.1):
        self.k = k
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold
        self.centroids = []
        self.clusters = []
        self.iteration = 0
        self.converged = False
        self.inertia_history = []

    def _initialize_centroids(self, data):
        """K-Means++ initialization for better starting centroids."""
        n = data.shape[0]
        dim = data.shape[1]

        # Pick the first centroid randomly
        idx = np.random.randint(n)
        centers = [data[idx].copy()]

        for i in range(1, self.k):
            # Compute distances to nearest existing centroid
            dists = np.full(n, np.inf)
            for c in centers:
                d = np.sum((data - c) ** 2, axis=1)
                dists = np.minimum(dists, d)

            # Choose next centroid with probability proportional to distance^2
            probs = dists / dists.sum()
            idx = np.random.choice(n, p=probs)
            centers.append(data[idx].copy())

        self.centroids = [Centroid(c, i) for i, c in enumerate(centers)]
        print(f"  Initialized {self.k} centroids (K-Means++ method)")

    def _distance_layer(self, data):
        """
        Distance Calculation Layer: compute Euclidean distance
        from every data point to every centroid.
        D[i, k] = ||data[i] - centroid_k||
        """
        centroid_positions = np.array([c.position for c in self.centroids])
        # Vectorized distance computation
        # data: (N, dim), centroid_positions: (K, dim)
        diff = data[:, np.newaxis, :] - centroid_positions[np.newaxis, :, :]  # (N, K, dim)
        distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, K)
        return distances

    def _assignment_layer(self, distances):
        """
        Assignment (Output) Layer: argmin over distances.
        cluster_label = argmin_k(D[i, k])
        """
        return np.argmin(distances, axis=1)

    def _update_centroids(self, data, labels):
        """
        Centroid Update: recalculate each centroid as mean of assigned points.
        Returns True if any centroid moved more than the threshold.
        """
        moved = False
        for centroid in self.centroids:
            centroid.prev_position = centroid.position.copy()
            mask = labels == centroid.index
            if np.any(mask):
                new_pos = data[mask].mean(axis=0)
                movement = np.sqrt(np.sum((new_pos - centroid.position) ** 2))
                if movement > self.convergence_threshold:
                    moved = True
                centroid.position = new_pos
        return moved

    def _calculate_inertia(self, data, labels):
        """
        Calculate inertia (Sum of Squared Errors / SSE):
        SSE = sum_i ||x_i - c_{label_i}||^2
        """
        inertia = 0.0
        for centroid in self.centroids:
            mask = labels == centroid.index
            if np.any(mask):
                inertia += np.sum((data[mask] - centroid.position) ** 2)
        return inertia

    def fit(self, data):
        """
        Run the full K-Means NN forward pass iteratively until convergence.
        """
        print(f"\nRunning K-Means Neural Network (K={self.k})...")
        t0 = time.time()

        self._initialize_centroids(data)

        labels = np.full(data.shape[0], -1, dtype=np.int32)

        for it in range(self.max_iter):
            self.iteration = it + 1

            # Forward pass through the network layers
            distances = self._distance_layer(data)       # Layer 2: Distance
            labels = self._assignment_layer(distances)    # Layer 4: Assignment (argmin)
            moved = self._update_centroids(data, labels)  # Update Layer 3: Centroids

            inertia = self._calculate_inertia(data, labels)
            self.inertia_history.append(inertia)

            if it % 10 == 0 or not moved:
                print(f"  Iteration {it + 1:3d} | Inertia (SSE): {inertia:,.2f} | "
                      f"{'Converged!' if not moved else 'Running...'}")

            if not moved:
                self.converged = True
                break

        # Build final cluster lists
        self.clusters = [np.where(labels == i)[0] for i in range(self.k)]
        active = sum(1 for c in self.clusters if len(c) > 0)

        elapsed = time.time() - t0
        print(f"\n  K-Means completed in {elapsed:.2f}s")
        print(f"  Final iteration: {self.iteration}")
        print(f"  Final inertia: {self.inertia_history[-1]:,.2f}")
        print(f"  Convergence: {'Yes' if self.converged else 'No (max iterations reached)'}")
        print(f"  Active clusters: {active}/{self.k}")
        for i, cluster in enumerate(self.clusters):
            print(f"    Cluster {i + 1}: {len(cluster):,} points")

        return labels


# ─────────────────────── Feature Extraction ───────────────────────────

def extract_features(iterations, magnitudes, real_coords, imag_coords, max_iter):
    """
    Extract and normalize feature vectors for every pixel.
    Features: [real, imag, normalized_iterations, log_magnitude]
    These map to the Input Layer of the K-Means NN.
    """
    print("\nExtracting features for K-Means input layer...")
    h, w = iterations.shape
    n = h * w

    # Flatten arrays
    iters_flat = iterations.flatten()
    mags_flat = magnitudes.flatten()
    real_flat = real_coords.flatten()
    imag_flat = imag_coords.flatten()

    # Normalize features to [0, 1] range
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    # Log-transform magnitudes for better distribution
    log_mags = np.log1p(mags_flat)

    features = np.column_stack([
        normalize(real_flat),
        normalize(imag_flat),
        normalize(iters_flat),
        normalize(log_mags)
    ])

    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Features: [real, imag, iterations, log_magnitude]")
    return features


# ─────────────────────── Visualization ────────────────────────────────

def generate_cluster_colors(k):
    """Generate K visually distinct colors (matching HTML palette style)."""
    base_colors = [
        (255, 107, 107),  # #FF6B6B - Red
        (78, 205, 196),   # #4ECDC4 - Teal
        (69, 183, 209),   # #45B7D1 - Blue
        (255, 160, 122),  # #FFA07A - Salmon
        (152, 216, 200),  # #98D8C8 - Mint
        (247, 220, 111),  # #F7DC6F - Yellow
        (187, 143, 206),  # #BB8FCE - Purple
        (133, 193, 226),  # #85C1E2 - Light Blue
    ]

    if k <= len(base_colors):
        return base_colors[:k]

    # Generate additional colors via HSV if needed
    colors = list(base_colors)
    for i in range(k - len(base_colors)):
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio for spacing
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def render_mandelbrot(labels, iterations, width, height, k, max_iter,
                      centroids, clusters, inertia_history):
    """
    Render the Mandelbrot set colored by K-Means cluster assignments.
    Interior (max_iter) points are rendered as dark/black.
    """
    print("\nRendering Mandelbrot K-Means visualization...")

    cluster_colors = generate_cluster_colors(k)
    labels_2d = labels.reshape(height, width)
    iters_2d = iterations.reshape(height, width) if isinstance(iterations, np.ndarray) and iterations.ndim == 1 else iterations

    # Create image
    img = Image.new('RGB', (width, height))
    pixels = np.zeros((height, width, 3), dtype=np.uint8)

    for cluster_idx in range(k):
        mask = labels_2d == cluster_idx
        base_r, base_g, base_b = cluster_colors[cluster_idx]

        # Get iteration values for shading within cluster
        cluster_iters = iters_2d[mask]
        if len(cluster_iters) == 0:
            continue

        iter_min = cluster_iters.min()
        iter_max = cluster_iters.max()
        iter_range = max(iter_max - iter_min, 1)

        # Shade within each cluster by iteration count for depth
        # Points closer to max_iter are darker, points that escaped early are brighter
        norm_iters = (cluster_iters - iter_min) / iter_range

        # Apply shading: mix base color with brightness factor
        brightness = 0.3 + 0.7 * norm_iters  # Range [0.3, 1.0]

        pixels[mask, 0] = np.clip(base_r * brightness, 0, 255).astype(np.uint8)
        pixels[mask, 1] = np.clip(base_g * brightness, 0, 255).astype(np.uint8)
        pixels[mask, 2] = np.clip(base_b * brightness, 0, 255).astype(np.uint8)

    # Make interior points (Mandelbrot set proper) very dark
    interior = iters_2d >= max_iter
    pixels[interior] = [10, 10, 15]

    img = Image.fromarray(pixels)

    # ── Add info overlay ──
    draw = ImageDraw.Draw(img)

    # Semi-transparent info box in top-left
    box_w, box_h = 360, 200
    overlay = Image.new('RGBA', (box_w, box_h), (0, 0, 0, 180))
    img_rgba = img.convert('RGBA')
    img_rgba.paste(overlay, (10, 10), overlay)

    draw = ImageDraw.Draw(img_rgba)

    try:
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_body = ImageFont.truetype("arial.ttf", 13)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_body = font_title
        font_small = font_title

    y_off = 18
    draw.text((20, y_off), "K-Means NN × Mandelbrot Set", fill=(255, 255, 255), font=font_title)
    y_off += 28

    info_lines = [
        f"Resolution: {width} × {height}",
        f"Max Iterations: {max_iter}",
        f"Clusters (K): {k}",
        f"K-Means Iterations: {len(inertia_history)}",
        f"Final Inertia (SSE): {inertia_history[-1]:,.2f}" if inertia_history else "N/A",
        f"Converged: {'Yes' if len(inertia_history) < KMEANS_MAX_ITER else 'No'}",
        f"Active Clusters: {sum(1 for c in clusters if len(c) > 0)}/{k}",
    ]

    for line in info_lines:
        draw.text((20, y_off), line, fill=(200, 220, 255), font=font_body)
        y_off += 18

    # ── Cluster legend ──
    y_off += 5
    legend_x = 20
    for i in range(k):
        color = cluster_colors[i]
        count = len(clusters[i]) if i < len(clusters) else 0
        # Draw color swatch
        draw.rectangle([legend_x, y_off + 530, legend_x + 14, y_off + 544], fill=color, outline=(255, 255, 255))
        draw.text((legend_x + 18, y_off + 530), f"C{i+1}: {count:,}px", fill=(200, 200, 200), font=font_small)
        legend_x += 110
        if legend_x > width - 120:
            legend_x = 20
            y_off += 18

    # Convert back to RGB for saving
    img_final = img_rgba.convert('RGB')
    return img_final


# ─────────────────────────── Main ─────────────────────────────────────

def main():
    print("=" * 65)
    print("  Mandelbrot Set × K-Means Neural Network Architecture")
    print("=" * 65)
    print(f"\n  Image:  {WIDTH} × {HEIGHT}")
    print(f"  Region: [{X_MIN}, {X_MAX}] × [{Y_MIN}, {Y_MAX}]")
    print(f"  Max iterations: {MAX_ITER}")
    print(f"  K-Means clusters: {K}")
    print()

    # Step 1: Compute Mandelbrot set
    iterations, magnitudes, real_coords, imag_coords = compute_mandelbrot(
        WIDTH, HEIGHT, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER
    )

    # Step 2: Extract features (→ Input Layer of K-Means NN)
    features = extract_features(iterations, magnitudes, real_coords, imag_coords, MAX_ITER)

    # Step 3: Run K-Means Neural Network
    kmeans_nn = KMeansNeuralNetwork(
        k=K,
        max_iter=KMEANS_MAX_ITER,
        convergence_threshold=CONVERGENCE_THRESHOLD
    )
    labels = kmeans_nn.fit(features)

    # Step 4: Render visualization
    img = render_mandelbrot(
        labels, iterations.flatten(), WIDTH, HEIGHT, K, MAX_ITER,
        kmeans_nn.centroids, kmeans_nn.clusters, kmeans_nn.inertia_history
    )

    # Step 5: Save output
    img.save(OUTPUT_FILE, quality=95)
    print(f"\nSaved: {OUTPUT_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
