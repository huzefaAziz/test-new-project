from functools import cache
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Cat Image (1200x1200)
# -------------------------------
cat_img = plt.imread("card.jpg")
if cat_img.ndim == 2:
    cat_arr = np.stack([cat_img] * 3, axis=-1)
else:
    cat_arr = np.array(cat_img)[:, :, :3].copy()
if cat_arr.max() > 1:
    cat_arr = cat_arr / 255.0
cat_arr = np.clip(cat_arr, 0, 1)
# Resize to 1200x1200
h, w = 1200, 1200
if cat_arr.shape[0] != h or cat_arr.shape[1] != w:
    y_idx = np.linspace(0, cat_arr.shape[0] - 1, h).astype(int)
    x_idx = np.linspace(0, cat_arr.shape[1] - 1, w).astype(int)
    cat_arr = cat_arr[np.ix_(y_idx, x_idx)]

# -------------------------------
# 2. Mandelbrot Fractal (1200x1200) - values only, no separate image (vectorized)
# -------------------------------
def mandelbrot_grid(width, height, x_min=-2.0, x_max=1.0, y_min=-1.25, y_max=1.25, max_iter=80):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    cx, cy = np.meshgrid(x, y)
    c = cx + 1j * cy
    z = np.zeros_like(c)
    out = np.zeros((height, width))
    for n in range(max_iter):
        mask = np.abs(z) <= 2
        out[mask] = n / max_iter
        z[mask] = z[mask] * z[mask] + c[mask]
    out[np.abs(z) <= 2] = 1.0
    return out

mandel = mandelbrot_grid(w, h)

# Colormap: fractal value -> RGB (hot-style: black -> red -> yellow -> white)
def fractal_to_rgb(m):
    m = np.clip(m, 0, 1)
    r = np.clip(2 * m, 0, 1)
    g = np.clip(2 * m - 0.5, 0, 1)
    b = np.clip(2 * m - 1.0, 0, 1)
    return np.stack([r, g, b], axis=-1)

mandel_rgb = fractal_to_rgb(mandel)

# -------------------------------
# 3. Add fractal INTO the cat (no separate Mandelbrot image)
# -------------------------------
# Where the cat is brighter (face, fur), show more fractal so the pattern is *inside* the cat
lum = np.dot(cat_arr, [0.299, 0.587, 0.114])
lum = np.stack([lum, lum, lum], axis=-1)
alpha = np.clip(0.35 + 0.5 * lum, 0, 1)  # more fractal inside cat (bright) areas
fractal_inside_cat = np.clip((1 - alpha) * cat_arr + alpha * mandel_rgb, 0, 1)

# -------------------------------
# 4. Model: only KMeans (on the cat+fractal blend)
# -------------------------------
X = fractal_inside_cat.reshape(-1, 3)
model = Pipeline([
    ("kmeans", KMeans(n_clusters=64, random_state=0))
])
labels = model.fit_predict(X)
centroids = model.named_steps["kmeans"].cluster_centers_

# -------------------------------
# 5. Output: cat with fractal inside (quantized by KMeans, one image only)
# -------------------------------
dream_rgb = np.clip(centroids[labels].reshape(h, w, 3), 0, 1)

# -------------------------------
# 6. Show only the cat+fractal result (no Mandelbrot image)
# -------------------------------
plt.figure(figsize=(12, 12))
plt.imshow(dream_rgb)
plt.axis("off")
plt.tight_layout()
plt.savefig("deepdream_cat_mandelbrot.png", bbox_inches="tight", pad_inches=0, dpi=100)
plt.show()
