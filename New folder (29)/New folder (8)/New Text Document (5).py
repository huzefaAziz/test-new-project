import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

def mandelbrot_points(width=300, height=300, max_iter=100):
    xs = np.linspace(-2.0, 1.0, width)
    ys = np.linspace(-1.5, 1.5, height)

    X = []
    for x in xs:
        for y in ys:
            c = complex(x, y)
            z = 0
            for i in range(max_iter):
                z = z*z + c
                if abs(z) > 2:
                    break
            X.append([x, y, i])  # spatial + iteration depth

    return np.array(X)

class KNNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=16):
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        self.knn.fit(X)
        return self

    def transform(self, X):
        distances, _ = self.knn.kneighbors(X)
        return distances
model = Pipeline([
    ("knn", KNNTransformer(n_neighbors=16)),
    ("kmeans", KMeans(n_clusters=64, random_state=0))
])
X = mandelbrot_points(200, 200)
Z = model.fit_transform(X)

plt.scatter(X[:, 0], X[:, 1], c=Z.argmax(axis=1), s=1, cmap="turbo")
plt.title("Mandelbrot latent clusters (KNN → KMeans)")
plt.axis("off")
plt.show()
