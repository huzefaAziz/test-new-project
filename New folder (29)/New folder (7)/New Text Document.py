import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
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
                k,               # escape time
                abs(z),           # final magnitude
                np.mean(traj),
                np.std(traj)
            ])
            coords.append((i, j))

    return np.array(X), coords

X, coords = mandelbrot_features()

scaler = StandardScaler()
Xn = scaler.fit_transform(X)

memory = NearestNeighbors(
    n_neighbors=5,
    algorithm="auto",
    metric="euclidean"
)

memory.fit(Xn)
distances, indices = memory.kneighbors(Xn)

width = height = int(np.sqrt(len(X)))
image = np.zeros((width, height))

for (i, j), d in zip(coords, distances.mean(axis=1)):
    image[i, j] = d

plt.figure(figsize=(6, 6))
plt.imshow(image, cmap="inferno")
plt.axis("off")
plt.title("Mandelbrot via Nearest-Neighbor Memory Network")
plt.show()
