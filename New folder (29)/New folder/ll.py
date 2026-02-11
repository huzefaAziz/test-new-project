import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def mandelbrot_escape(c, max_iter=80):
    """Escape-time for grid of complex c. Returns iteration count (max_iter if in set)."""
    z = np.zeros_like(c, dtype=complex)
    out = np.full(c.shape, max_iter, dtype=int)
    for i in range(max_iter):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] ** 2 + c[mask]
        escaped = (np.abs(z) > 2) & (out == max_iter)
        out[escaped] = i
    return out


# Grid in the complex plane (Mandelbrot region)
n = 200
x = np.linspace(-2.5, 1.0, n)
y = np.linspace(-1.25, 1.25, n)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y

escape = mandelbrot_escape(C)
in_set = (escape >= 80).astype(int).ravel()  # 1 = in set, 0 = escaped

# Features: (real, imag, escape_time normalized)
features = np.column_stack([
    X.ravel(), Y.ravel(),
    np.clip(escape.ravel() / 80.0, 0, 1)
])
y_label = in_set

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)
print("PCA explained variance ratio:", pca.explained_variance_ratio_)

# Logistic regression: predict in-set vs escaped
clf = LogisticRegression(max_iter=500, random_state=42)
clf.fit(features, y_label)
acc = clf.score(features, y_label)
print(f"LogisticRegression accuracy (train): {acc:.4f}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].imshow(escape, extent=[x.min(), x.max(), y.min(), y.max()],
               origin="lower", cmap="hot", aspect="auto")
axes[0].set_title("Mandelbrot set (escape time)")
axes[0].set_xlabel("Re(c)")
axes[0].set_ylabel("Im(c)")

axes[1].scatter(X_pca[y_label == 0, 0], X_pca[y_label == 0, 1],
                c="blue", s=1, alpha=0.5, label="escaped")
axes[1].scatter(X_pca[y_label == 1, 0], X_pca[y_label == 1, 1],
                c="red", s=1, alpha=0.5, label="in set")
axes[1].set_title("PCA of (Re, Im, escape)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend(markerscale=5)

pred = clf.predict(features)
axes[2].imshow(pred.reshape(n, n), extent=[x.min(), x.max(), y.min(), y.max()],
               origin="lower", cmap="coolwarm", aspect="auto")
axes[2].set_title("LR prediction (in set vs escaped)")
axes[2].set_xlabel("Re(c)")
axes[2].set_ylabel("Im(c)")

plt.tight_layout()
plt.savefig("mandelbrot_ml.png", dpi=120)
plt.show()
