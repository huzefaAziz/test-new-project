import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
def mandelbrot_set(width=300, height=300, max_iter=100):
    xs = np.linspace(-2.0, 1.0, width)
    ys = np.linspace(-1.5, 1.5, height)

    X = []
    iters = []

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            c = x + 1j * y
            z = 0
            k = 0
            while abs(z) <= 2 and k < max_iter:
                z = z*z + c
                k += 1

            X.append([x, y])
            iters.append(k)

    return np.array(X), np.array(iters)

X, y = mandelbrot_set()
model = Pipeline([
    ("scale", StandardScaler()),

    # KNN as a distance-based embedding (this replaces activation)
    ("knn", KNeighborsTransformer(
        n_neighbors=16,
        mode="distance"
    )),

    # KMeans as neuron prototypes
    ("kmeans", KMeans(
        n_clusters=64,
        n_init=10,
        random_state=0
    ))
])
Z = model.fit_transform(X)
print(Z.shape)   # (num_points, 64)
plt.figure(figsize=(6, 6))
plt.scatter(Z[:, 0], Z[:, 1], c=y, s=1, cmap="hot")
plt.colorbar(label="Escape iterations")
plt.title("Mandelbrot set in sklearn latent space")
plt.show()
