"""
Mandelbrot set + Kernel PCA + Neural Network
Generates Mandelbrot data, reduces dimensions with Kernel PCA, then classifies
(in-set vs out-of-set) with a small neural network.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def mandelbrot_escape(c: complex, max_iter: int = 100) -> int:
    """Return iteration count at which z escapes (or max_iter if in set)."""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter


def generate_mandelbrot_data(
    n_samples: int = 5000,
    x_range: tuple = (-2.5, 1.0),
    y_range: tuple = (-1.25, 1.25),
    max_iter: int = 100,
    in_set_threshold: int = 100,
    seed: int = 42,
):
    """Sample (x,y) in the plane and label: 1 = in set, 0 = out of set."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, size=n_samples)
    y = rng.uniform(*y_range, size=n_samples)
    labels = np.zeros(n_samples, dtype=int)
    escape_counts = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        c = complex(x[i], y[i])
        esc = mandelbrot_escape(c, max_iter)
        escape_counts[i] = esc
        labels[i] = 1 if esc >= in_set_threshold else 0
    # Features: (x, y) and optionally escape count for richer representation
    X = np.column_stack([x, y, escape_counts])
    return X, labels


def run_pipeline(seed, n_samples=3000):
    """Generate data, fit scaler/KPCA/NN, return everything needed to plot."""
    X, y = generate_mandelbrot_data(n_samples=n_samples, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    n_components = min(10, X_train_s.shape[1], X_train_s.shape[0] - 1)
    kpca = KernelPCA(n_components=n_components, kernel="rbf", gamma=0.5, random_state=42)
    X_train_kpca = kpca.fit_transform(X_train_s)
    X_test_kpca = kpca.transform(X_test_s)
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
    )
    nn.fit(X_train_kpca, y_train)
    y_pred = nn.predict(X_test_kpca)
    return X, y, X_test_kpca, y_test, y_pred


def main():
    N_SAMPLES = 3000
    THROTTLE_SEC = 0.6

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.tight_layout()

    last_update = [0.0]

    def update_plots(X, y, X_test_kpca, y_test, y_pred):
        X_test_kpca_2 = X_test_kpca[:, :2]
        for ax in axes:
            ax.clear()
        # 1) Mandelbrot (x, y), true labels
        ax = axes[0]
        for lbl, color, name in [(0, "C0", "out-of-set"), (1, "C1", "in-set")]:
            m = y == lbl
            ax.scatter(X[m, 0], X[m, 1], c=color, s=5, alpha=0.6, label=name)
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")
        ax.set_title("Mandelbrot data (true labels)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        # 2) Kernel PCA, true labels
        ax = axes[1]
        for lbl, color, name in [(0, "C0", "out-of-set"), (1, "C1", "in-set")]:
            m = y_test == lbl
            ax.scatter(X_test_kpca_2[m, 0], X_test_kpca_2[m, 1], c=color, s=10, alpha=0.6, label=name)
        ax.set_xlabel("Kernel PCA 1")
        ax.set_ylabel("Kernel PCA 2")
        ax.set_title("Kernel PCA space (true labels)")
        ax.legend(loc="upper right", fontsize=8)
        # 3) Kernel PCA, NN predictions
        ax = axes[2]
        for lbl, color, name in [(0, "C0", "pred out"), (1, "C1", "pred in-set")]:
            m = y_pred == lbl
            ax.scatter(X_test_kpca_2[m, 0], X_test_kpca_2[m, 1], c=color, s=10, alpha=0.6, label=name)
        ax.set_xlabel("Kernel PCA 1")
        ax.set_ylabel("Kernel PCA 2")
        ax.set_title("Kernel PCA space (NN predictions)")
        ax.legend(loc="upper right", fontsize=8)
        fig.canvas.draw_idle()

    def on_mouse_move(event):
        if event.inaxes is None:
            return
        now = time.perf_counter()
        if now - last_update[0] < THROTTLE_SEC:
            return
        last_update[0] = now
        # New seed from mouse position + time so each move gives new data
        seed = int((event.xdata or 0) * 1000 + (event.ydata or 0) * 1000 + now * 1000) % (2 ** 31)
        fig.suptitle(f"Regenerating (seed={seed})...", fontsize=10)
        fig.canvas.draw_idle()
        X, y, X_test_kpca, y_test, y_pred = run_pipeline(seed, n_samples=N_SAMPLES)
        acc = accuracy_score(y_test, y_pred)
        fig.suptitle(f"Seed={seed}  |  Test accuracy: {acc:.3f}", fontsize=10)
        update_plots(X, y, X_test_kpca, y_test, y_pred)

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    # Initial data
    print("Generating initial Mandelbrot data (move mouse to regenerate)...")
    X, y, X_test_kpca, y_test, y_pred = run_pipeline(42, n_samples=N_SAMPLES)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {acc:.4f}")
    fig.suptitle(f"Seed=42  |  Test accuracy: {acc:.3f}  |  Move mouse to generate new data", fontsize=10)
    update_plots(X, y, X_test_kpca, y_test, y_pred)
    plt.show()


if __name__ == "__main__":
    main()
