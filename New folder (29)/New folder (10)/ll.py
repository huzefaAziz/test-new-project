import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def mandelbrot_set(width, height, x_min=-2.5, x_max=1.0, y_min=-1.25, y_max=1.25, max_iters=80):
    """Generate Mandelbrot set: 1 = in set, 0 = escaped. Returns (X_flat, y_flat) for ML."""
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X_grid, Y_grid = np.meshgrid(x, y)
    C = X_grid + 1j * Y_grid
    z = np.zeros_like(C)
    escape = np.zeros(C.shape, dtype=int)
    for i in range(max_iters):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] ** 2 + C[mask]
        escape[mask] = i
    # Binary: in set (never escaped) vs escaped
    in_set = (np.abs(z) <= 2).astype(np.float32)
    # Features: real, imag, escape iteration
    X_flat = np.column_stack([
        X_grid.ravel(), Y_grid.ravel(), escape.ravel()
    ]).astype(np.float32)
    y_flat = in_set.ravel()
    return X_flat, y_flat


class mandelbrotdiffusionAI(nn.Module):
    def __init__(self, input_size, output_size):
        super(mandelbrotdiffusionAI, self).__init__()
        # KMeans for clustering Mandelbrot features; KNN used separately (see fit_mandelbrot)
        self.knn = KNeighborsClassifier(n_neighbors=8)
        self.ai = Pipeline([("kmeans", KMeans(n_clusters=64, random_state=0))])
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.model(x)

    def fit_mandelbrot(self, width=64, height=64, **mandelbrot_kwargs):
        """Generate Mandelbrot data; fit KMeans pipeline and KNN classifier."""
        X, y = mandelbrot_set(width, height, **mandelbrot_kwargs)
        self.ai.fit(X)  # KMeans clustering (no labels)
        self.knn.fit(X, y.astype(int))  # KNN classification (in set vs escaped)
        return X, y

    def predict_mandelbrot(self, X):
        """Predict class (in set / escaped) for Mandelbrot-style features (numpy)."""
        return self.knn.predict(X)


if __name__ == "__main__":
    # Mandelbrot set AI: generate data and run pipeline + NN
    width, height = 200, 200
    X, y = mandelbrot_set(width, height, max_iters=50)
    n_samples, input_size = X.shape
    output_size = 2  # in set vs escaped
    ai = mandelbrotdiffusionAI(input_size=input_size, output_size=output_size)
    ai.fit_mandelbrot(width, height, max_iters=50)
    x_t = torch.from_numpy(X).float()
    y_long = torch.from_numpy(y.astype(np.int64)).long()

    # Train NN in one shot (zero epoch loops): L-BFGS does all steps internally
    optimizer = torch.optim.LBFGS(ai.parameters(), lr=1.0, max_iter=100)

    def closure():
        optimizer.zero_grad()
        out = ai(x_t)
        loss = nn.functional.cross_entropy(out, y_long)
        loss.backward()
        return loss

    optimizer.step(closure)
    ai.eval()
    with torch.no_grad():
        out = ai(x_t)
    print("Mandelbrot set AI")
    print("  Samples:", n_samples, "| NN output shape:", out.shape)
    print("  Predict (first 5):", ai.predict_mandelbrot(X[:5]))

    # Plot: ground truth, KNN prediction, NN output
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    ax1.imshow(y.reshape(height, width), extent=(-2.5, 1.0, -1.25, 1.25), origin="lower", cmap="hot")
    ax1.set_title("Mandelbrot set (ground truth)")
    ax1.set_xlabel("Re(c)")
    ax1.set_ylabel("Im(c)")

    pred = ai.predict_mandelbrot(X)
    ax2.imshow(pred.reshape(height, width), extent=(-2.5, 1.0, -1.25, 1.25), origin="lower", cmap="hot")
    ax2.set_title("KNN prediction (in set vs escaped)")
    ax2.set_xlabel("Re(c)")
    ax2.set_ylabel("Im(c)")

    out_np = out.detach().numpy()
    out_class = out_np.argmax(axis=1).reshape(height, width)
    ax3.imshow(out_class, extent=(-2.5, 1.0, -1.25, 1.25), origin="lower", cmap="hot")
    ax3.set_title("NN output (argmax)")
    ax3.set_xlabel("Re(c)")
    ax3.set_ylabel("Im(c)")
    plt.tight_layout()
    plt.show()
