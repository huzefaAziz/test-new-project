import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier

class KNNLayer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=8):
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        # fake labels so classifier can fit
        y_fake = np.zeros(len(X))
        self.knn.fit(X, y_fake)
        return self

    def transform(self, X):
        # use neighbor distances as features
        distances, _ = self.knn.kneighbors(X)
        return distances
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
def mandelbrot_features(w=200, h=200, max_iter=200):
    xs = np.linspace(-2, 1, w)
    ys = np.linspace(-1.5, 1.5, h)

    X = []
    for x in xs:
        for y in ys:
            c = x + 1j*y
            z = 0
            it = 0
            while abs(z) <= 2 and it < max_iter:
                z = z*z + c
                it += 1
            X.append([it/max_iter, x, y])

    return np.array(X)

X = mandelbrot_features()

model = Pipeline([
    ("knn", KNNLayer(n_neighbors=8)),
    ("kmeans", KMeans(n_clusters=64, random_state=0))
])
Z = model.fit_transform(X)

print(Z.shape)   # (N, 64)
import matplotlib.pyplot as plt

plt.imshow(Z[:, 0].reshape(200, 200), cmap="inferno")
plt.colorbar()
plt.show()
