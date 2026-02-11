import numpy as np

def mandelbrot_features(width=300, height=300, max_iter=100):
    xs = np.linspace(-2.0, 1.0, width)
    ys = np.linspace(-1.5, 1.5, height)

    X = []
    coords = []

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            c = complex(x, y)
            z = 0
            it = 0
            while abs(z) <= 2 and it < max_iter:
                z = z*z + c
                it += 1

            # feature vector
            X.append([
                x, y,
                it,
                abs(z)
            ])
            coords.append((i, j))

    return np.array(X), coords
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsTransformer
from sklearn.cluster import KMeans

X, coords = mandelbrot_features()

model = Pipeline([
    ("knn", KNeighborsTransformer(
        n_neighbors=20,
        mode="distance"   # activation = distance
    )),
    ("kmeans", KMeans(
        n_clusters=64,
        n_init=10,
        random_state=0
    ))
])

Z = model.fit_transform(X)

import matplotlib.pyplot as plt

img = np.zeros((300, 300))

for idx, (i, j) in enumerate(coords):
    img[i, j] = Z[idx].argmin()  # closest cluster

plt.imshow(img)
plt.axis("off")
plt.show()
