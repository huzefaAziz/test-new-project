import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def mandelbrot_features(width=300, height=300, max_iter=200):
    x = np.linspace(-2.0, 1.0, width)
    y = np.linspace(-1.5, 1.5, height)

    X = []
    coords = []

    for i, re in enumerate(x):
        for j, im in enumerate(y):
            c = re + 1j * im
            z = 0
            traj = []

            for k in range(max_iter):
                z = z*z + c
                traj.append(abs(z))
                if abs(z) > 2:
                    break

            X.append([
                k,                # escape time
                abs(z),            # magnitude
                np.mean(traj),
                np.std(traj)
            ])
            coords.append((i, j))

    return np.array(X), coords
X, coords = mandelbrot_features()
model = Pipeline([
    ("scale", StandardScaler()),

    # Nearest-neighbor memory embedding
    ("knn", KNeighborsTransformer(
        n_neighbors=8,
        mode="distance"
    )),

    # Concept abstraction layer
    ("kmeans", KMeans(
        n_clusters=64,
        n_init=10,
        random_state=0
    ))
])
Z = model.fit_transform(X)
labels = model.named_steps["kmeans"].labels_
width = height = int(np.sqrt(len(labels)))
image = np.zeros((width, height))

for (i, j), label in zip(coords, labels):
    image[i, j] = label

plt.figure(figsize=(6, 6))
plt.imshow(image, cmap="twilight")
plt.axis("off")
plt.title("Mandelbrot via KNN Memory → KMeans Concepts")
plt.show()
