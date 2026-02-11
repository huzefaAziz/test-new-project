from functools import cache
import numpy as np
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline

# Recursive Fibonacci
@cache
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# Example input X
X = np.arange(20).reshape(-1, 1)

# Forward pipeline: fib -> one-hot
pipeline = Pipeline([
    ("fib", FunctionTransformer(lambda x: np.vectorize(fib)(x), validate=False)),
    ("onehot", OneHotEncoder(drop='if_binary', sparse_output=False))
])

# Fit and transform
X_encoded = pipeline.fit_transform(X)
print("Encoded:\n", X_encoded)

# ----------------------------
# Decoder: reverse the pipeline
# ----------------------------

# Step 1: reverse one-hot encoding
onehot = pipeline.named_steps["onehot"]
fib_values = onehot.inverse_transform(X_encoded)

# Step 2: reverse Fibonacci (find n such that fib(n) = value)
# We'll brute-force since recursive Fibonacci is not invertible analytically
max_n = 100  # max possible n to search
fib_cache = {fib(i): i for i in range(max_n)}  # precompute fib -> n mapping

# Map back
X_decoded = np.array([[fib_cache[val[0]]] for val in fib_values])
print("Decoded (original X):\n", X_decoded)
