"""
Mandelbrot set with sklearn Pipeline: KernelPCA + KMeans (neural-network alternative).
Fit on subset; predict full grid via chunked KPCA transform + KMeans (no colour approximation).
Uses: Pipeline, KernelPCA, KMeans, matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

def mandelbrot_escape(c: np.ndarray, max_iter: int = 80) -> np.ndarray:
    """Compute escape iteration count for each complex point c."""
    z = np.zeros_like(c, dtype=np.complex128)
    escape = np.zeros(c.shape, dtype=np.int32)
    for i in range(max_iter):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] ** 2 + c[mask]
        escape[mask] = i
    return escape


def main():
    # Mandelbrot grid
    width, height = 320, 240
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    # Raw Mandelbrot features: escape count + real/imag
    escape = mandelbrot_escape(C)
    # Shape (H, W) -> (n_samples, n_features); use escape + coord as features
    n_samples = escape.size
    X_flat = np.column_stack([
        escape.ravel(),
        X.ravel(),
        Y.ravel(),
        np.abs(C).ravel(),
    ])

    # Pipeline: KernelPCA -> KMeans (neural-network alternative)
    # Fit on a subset to avoid 76k×76k Gram matrix (~44 GiB); predict on full grid in chunks
    fit_size = 6000  # Gram matrix 6000×6000 ≈ 288 MiB
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(n_samples, size=min(fit_size, n_samples), replace=False)
    X_fit = X_flat[fit_idx]

    model = Pipeline([
        ("kpca", KernelPCA(n_components=16, kernel="rbf")),
        ("kmeans", KMeans(n_clusters=64)),
    ])
    model.fit(X_fit)

    # Predict full grid: KernelPCA transform in chunks + KMeans (exact, no colour approximation)
    chunk_size = 10_000
    labels = np.empty(n_samples, dtype=np.int32)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = X_flat[start:end]
        trans = model.named_steps["kpca"].transform(chunk)
        labels[start:end] = model.named_steps["kmeans"].predict(trans)
    labels_2d = labels.reshape(height, width)

    # Plot: Mandelbrot colored by KernelPCA + KMeans clusters
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(escape, extent=(x_min, x_max, y_min, y_max), origin="lower",
                   cmap="viridis", aspect="auto")
    axes[0].set_title("Mandelbrot (escape iteration)")
    axes[0].set_xlabel("Re(c)")
    axes[0].set_ylabel("Im(c)")

    axes[1].imshow(labels_2d, extent=(x_min, x_max, y_min, y_max), origin="lower",
                   cmap="tab20", aspect="auto")
    axes[1].set_title("KernelPCA + KMeans clusters (64)")
    axes[1].set_xlabel("Re(c)")
    axes[1].set_ylabel("Im(c)")

    plt.tight_layout()
    plt.savefig("mandelbrot_ml.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
