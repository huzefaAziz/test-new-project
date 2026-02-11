import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- 1. Mandelbrot feature generator ---
def mandelbrot_set(width=200, height=200, max_iter=200):
    xs = np.linspace(-2.0, 1.0, width)
    ys = np.linspace(-1.5, 1.5, height)
    X = []
    coords = []

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            c = x + 1j*y
            z = 0j
            it = 0
            while abs(z) <= 2 and it < max_iter:
                z = z*z + c
                it += 1
            X.append([it/max_iter, np.real(c), np.imag(c), abs(z)])
            coords.append((i,j))
    return np.array(X), np.array(coords)

X, coords = mandelbrot_set()

# --- 2. Wrap KNeighborsClassifier so it works in a Pipeline ---
class KNNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def fit(self, X, y=None):
        # We just need labels for KNN, generate dummy labels
        dummy_y = np.zeros(X.shape[0])
        self.knn.fit(X, dummy_y)
        return self

    def transform(self, X):
        # Output distances to neighbors as features
        # Note: kneighbors returns distances + indices, we take distances
        distances, _ = self.knn.kneighbors(X)
        return distances

# --- 3. Pipeline ---
model = Pipeline([
    ("knn", KNNTransformer(n_neighbors=8)),
    ("kmeans", KMeans(n_clusters=64, random_state=0))
])

# --- 4. Fit-transform ---
Z = model.fit_transform(X)

# --- 5. Access kmeans labels ---
labels = model.named_steps["kmeans"].labels_

# --- 6. Optional visualization ---
plt.imshow(labels.reshape(200,200), cmap="tab20")
plt.title("Mandelbrot Clusters")
plt.colorbar()
plt.show()
