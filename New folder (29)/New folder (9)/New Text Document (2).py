import numpy as np

def mandelbrot_set(width=300, height=300, max_iter=200):
    xs = np.linspace(-2.0, 1.0, width)
    ys = np.linspace(-1.5, 1.5, height)

    X = []
    coords = []

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            c = x + 1j * y
            z = 0j
            it = 0
            while abs(z) <= 2 and it < max_iter:
                z = z*z + c
                it += 1

            # feature vector (you can extend this)
            X.append([
                it / max_iter,
                np.real(c),
                np.imag(c),
                abs(z)
            ])

            coords.append((i, j))

    return np.array(X), np.array(coords)
X, coords = mandelbrot_set(
    width=200,
    height=200,
    max_iter=300
)
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import FunctionTransformer

def knn_distances(X):
    nn = NearestNeighbors(n_neighbors=8)
    nn.fit(X)
    D, _ = nn.kneighbors(X)
    return D

model = Pipeline([
    ("knn", FunctionTransformer(knn_distances, validate=False)),
    ("kmeans", KMeans(n_clusters=64, n_init=10, random_state=0))
])

Z = model.fit_transform(X)

import matplotlib.pyplot as plt

plt.imshow(
    Z[:, 0].reshape(200, 200),
    cmap="inferno"
)
plt.colorbar()
plt.title("Mandelbrot latent neuron 0")
plt.show()
