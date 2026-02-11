import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# -------------------------------
# 1. Load Cat Image
# -------------------------------
cat_img = Image.open("card.jpg").convert("RGB").resize((1200, 1200))
cat_arr = np.array(cat_img) / 255.0
h, w, c = cat_arr.shape

# -------------------------------
# 2. Generate Mandelbrot Fractal
# -------------------------------
def mandelbrot(width, height, max_iter=100):
    x = np.linspace(-2.0, 1.0, width)
    y = np.linspace(-1.5, 1.5, height)
    fractal = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            z = 0
            c_comp = complex(x[j], y[i])
            iter_count = 0
            while abs(z) <= 2 and iter_count < max_iter:
                z = z*z + c_comp
                iter_count += 1
            fractal[i, j] = iter_count
    return fractal / fractal.max()

mandel_map = mandelbrot(w, h)

# -------------------------------
# 3. Prepare Features for Pipeline
# -------------------------------
# Flatten image and add fractal as extra channel
features = np.dstack((cat_arr, mandel_map))
features_flat = features.reshape(-1, c+1)

# -------------------------------
# 4. Build Pipeline
# -------------------------------
model = Pipeline([
    ("knn", KNeighborsClassifier(n_neighbors=5)),
    ("kmeans", KMeans(n_clusters=64, random_state=42))
])

# Fit KMeans directly on features (KNN can be used for neighbor similarity)
model.named_steps["kmeans"].fit(features_flat)
labels = model.named_steps["kmeans"].labels_
centroids = model.named_steps["kmeans"].cluster_centers_

# -------------------------------
# 5. DeepDream-style Iterative Amplification
# -------------------------------
dream_flat = features_flat.copy()
iterations = 10
for _ in range(iterations):
    # Map each pixel to its cluster centroid (amplify patterns)
    dream_flat = centroids[labels]
    # Add slight neighbor mixing for swirl effect
    dream_img = dream_flat[:, :3].reshape(h, w, 3)
    dream_img = np.roll(dream_img, shift=2, axis=1)
    dream_flat[:, :3] = dream_img.reshape(-1,3)

# -------------------------------
# 6. Display Result
# -------------------------------
final_img = dream_flat[:, :3].reshape(h, w, 3)
plt.figure(figsize=(10,10))
plt.imshow(np.clip(final_img, 0, 1))
plt.axis('off')
plt.show()
