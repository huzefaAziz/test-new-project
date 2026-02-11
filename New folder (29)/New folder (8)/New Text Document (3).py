import numpy as np
import matplotlib.pyplot as plt

A, B, C, D = 4, 3, 64, 64
X = np.random.rand(A, B, C, D)

# Output image (no reduction)
img = np.zeros((A * C, B * D))

for a in range(A):
    for b in range(B):
        for c in range(C):
            for d in range(D):
                x = a * C + c
                y = b * D + d
                img[x, y] = X[a, b, c, d]
print(img.shape)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
