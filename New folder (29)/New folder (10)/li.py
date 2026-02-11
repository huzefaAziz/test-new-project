import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
# recursive fibonacci
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# generate fibonacci inputs
n =100
fib_values = np.arange(n).reshape(-1, 1)
pipeline = Pipeline([
    ("fib", preprocessing.FunctionTransformer(
        lambda x: np.vectorize(fib)(x),
        validate=False
    )),
    ("scale", preprocessing.StandardScaler())
])

fib_scaled = pipeline.fit_transform(fib_values)
print(fib_scaled)
