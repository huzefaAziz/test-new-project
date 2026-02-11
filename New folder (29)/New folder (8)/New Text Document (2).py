from scipy.optimize import root
import numpy as np

def sigma(x):
    return np.tanh(x)

def implicit_neuron(h, x, W, U, decay):
    return sigma(W @ h + U @ x) - decay * h

dim = 8
x = np.array([0.3, -0.1, 0.9])

W = np.random.randn(dim, dim) * 0.3
U = np.random.randn(dim, len(x)) * 0.3
decay = 0.2

sol = root(
    implicit_neuron,
    x0=np.zeros(dim),
    args=(x, W, U, decay)
)

print(sol.x)
