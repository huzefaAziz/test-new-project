import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
def mandelbrot_features(N=300, max_iter=80):
    xs = np.linspace(-2, 1, N)
    ys = np.linspace(-1.5, 1.5, N)

    X = []
    for x in xs:
        for y in ys:
            c = x + 1j*y
            z = 0
            for k in range(max_iter):
                z = z*z + c
                if abs(z) > 2:
                    break
            X.append([k, abs(z)])

    return np.array(X)
X = mandelbrot_features()

memory = NearestNeighbors(
    n_neighbors=32,
    algorithm="ball_tree"
).fit(X)
def one_step_diffusion_fix(X, memory, noise=0.15):
    X_noisy = X + noise * np.random.randn(*X.shape)
    _, idx = memory.kneighbors(X_noisy)
    return np.mean(X[idx], axis=1)
Z = one_step_diffusion_fix(X, memory)
N = int(np.sqrt(len(Z)))
img = Z[:, 0].reshape(N, N)

plt.imshow(img, cmap="inferno")
plt.axis("off")
plt.title("One-Step k-NN Diffusion Fix (Mandelbrot)")
plt.show()
