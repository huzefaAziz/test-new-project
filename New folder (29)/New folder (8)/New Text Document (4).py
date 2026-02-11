import numpy as np

def mandelbrot_features(width=300, height=300, max_iter=200):
    x = np.linspace(-2.0, 1.0, width)
    y = np.linspace(-1.5, 1.5, height)

    X = []
    coords = []

    for i in range(width):
        for j in range(height):
            c = x[i] + 1j * y[j]
            z = 0
            it = 0
            while abs(z) <= 2 and it < max_iter:
                z = z*z + c
                it += 1

            # feature vector
            X.append([
                it,
                abs(z),
                np.real(c),
                np.imag(c)
            ])
            coords.append((i, j))

    return np.array(X), coords
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

model = Pipeline([
    ("knn", NearestNeighbors(n_neighbors=8)),
    ("kmeans", KMeans(n_clusters=64, n_init=10))
])
X, coords = mandelbrot_features(width=200, height=200)

# KNN produces a distance-based embedding
distances, _ = model.named_steps["knn"].fit(X).kneighbors(X)

# KMeans clusters that embedding
labels = model.named_steps["kmeans"].fit_predict(distances)

import matplotlib.pyplot as plt

img = np.zeros((200, 200))

for (i, j), label in zip(coords, labels):
    img[i, j] = label

plt.imshow(img, cmap="twilight")
plt.axis("off")
plt.title("Mandelbrot Set — KNN → KMeans Latent Clusters")
plt.show()
