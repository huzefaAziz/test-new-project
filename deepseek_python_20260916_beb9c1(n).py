"""
Polymorphic implementation of ConjVI from arXiv:2102.08880v4.
"""
from __future__ import annotations
import abc
import numpy as np
from typing import Optional, Tuple


# ----------------------------------------------------------------------
# 1. Problem abstraction
# ----------------------------------------------------------------------
class MDPProblem(abc.ABC):
    @property
    @abc.abstractmethod
    def gamma(self) -> float: ...
    @property
    @abc.abstractmethod
    def B(self) -> np.ndarray: ...
    @property
    @abc.abstractmethod
    def state_grid(self) -> np.ndarray: ...
    @property
    @abc.abstractmethod
    def input_grid(self) -> np.ndarray: ...
    @property
    @abc.abstractmethod
    def disturbance_support(self) -> np.ndarray: ...
    @property
    @abc.abstractmethod
    def disturbance_pmf(self) -> np.ndarray: ...
    @abc.abstractmethod
    def f_s(self, x: np.ndarray) -> np.ndarray: ...
    @abc.abstractmethod
    def C_s(self, x: np.ndarray) -> np.ndarray: ...
    @abc.abstractmethod
    def C_i(self, u: np.ndarray) -> np.ndarray: ...

    def dynamics(self, x, u, w):
        return self.f_s(x) + self.B @ u + w

    def stage_cost(self, x, u):
        return self.C_s(x) + self.C_i(u)


# ----------------------------------------------------------------------
# 2. Conjugate transform abstraction
# ----------------------------------------------------------------------
class ConjugateTransform(abc.ABC):
    @abc.abstractmethod
    def conjugate(self, f: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (dual_grid, f*) with dual_grid shape (n_dual, 1)."""
        ...


class LLTConjugate(ConjugateTransform):
    """
    Brute-force 1-D Legendre–Fenchel conjugate on a fixed dual grid.
    (A real LLT would replace the inner `max` loop with Lucet's O(N) sweep.)
    """
    def __init__(self, dual_grid: np.ndarray):
        self.dual_grid = np.asarray(dual_grid, dtype=float).reshape(-1, 1)

    def conjugate(self, f: np.ndarray, grid: np.ndarray):
        g = np.asarray(grid, dtype=float).ravel()
        f = np.asarray(f, dtype=float).ravel()
        s = self.dual_grid.ravel()
        # f*(s) = max_x ( s*x - f(x) )      -- discrete 1-D conjugate
        f_star = np.array([np.max(si * g - f) for si in s])
        return self.dual_grid.copy(), f_star


# ----------------------------------------------------------------------
# 3. Value-iteration abstraction
# ----------------------------------------------------------------------
class ValueIterationAlgorithm(abc.ABC):
    def __init__(self, problem: MDPProblem, tol: float = 1e-6, max_iter: int = 1000):
        self.problem = problem
        self.tol = tol
        self.max_iter = max_iter

    @abc.abstractmethod
    def dp_operator(self, J: np.ndarray) -> np.ndarray: ...

    def solve(self, J0: Optional[np.ndarray] = None) -> np.ndarray:
        J = np.zeros(len(self.problem.state_grid)) if J0 is None else J0.copy()
        for _ in range(self.max_iter):
            J_new = self.dp_operator(J)
            if np.max(np.abs(J_new - J)) < self.tol:
                return J_new
            J = J_new
        return J


# ----------------------------------------------------------------------
# 4. Primal VI
# ----------------------------------------------------------------------
class PrimalVI(ValueIterationAlgorithm):
    def dp_operator(self, J):
        prob = self.problem
        X, U = prob.state_grid, prob.input_grid
        W, P = prob.disturbance_support, prob.disturbance_pmf
        gamma = prob.gamma
        J_new = np.empty(len(X))
        for i, x in enumerate(X):
            costs = prob.C_s(x) + prob.C_i(U)             # (N_u, 1)
            for j, u in enumerate(U):
                ev = 0.0
                for w, p in zip(W, P):
                    xn = prob.dynamics(x, u, w)
                    idx = np.argmin(np.linalg.norm(X - xn, axis=1))
                    ev += p * J[idx]
                costs[j] += gamma * ev
            J_new[i] = np.min(costs)
        return J_new


# ----------------------------------------------------------------------
# 5. Conjugate VI (ConjVI)
# ----------------------------------------------------------------------
class ConjVI(ValueIterationAlgorithm):
    def __init__(self, problem, conj_transform, tol=1e-6, max_iter=1000):
        super().__init__(problem, tol, max_iter)
        self.conj = conj_transform
        self._Ci_star = None

    def _ensure_input_conjugate(self):
        if self._Ci_star is not None:
            return
        U = self.problem.input_grid
        Ci_vals = np.asarray(self.problem.C_i(U), dtype=float).ravel()
        self._Ci_star = self.conj.conjugate(Ci_vals, U)   # (dual_grid, Ci*)

    def dp_operator(self, J):
        prob = self.problem
        X = prob.state_grid
        W, P = prob.disturbance_support, prob.disturbance_pmf
        gamma = prob.gamma

        self._ensure_input_conjugate()
        dual_grid, Ci_star = self._Ci_star          # (n_dual,1) and (n_dual,)

        # --- Step 1:  epsilon(x) = gamma * E_w J(x + w) ---
        epsilon = np.zeros(len(X))
        for i, x in enumerate(X):
            ev = 0.0
            for w, p in zip(W, P):
                xn = x + w
                idx = np.argmin(np.linalg.norm(X - xn, axis=1))
                ev += p * J[idx]
            epsilon[i] = gamma * ev

        # --- Step 2:  phi = Ci*(.) + epsilon*(.)   (both length n_dual) ---
        _, epsilon_star = self.conj.conjugate(epsilon, X)
        phi = Ci_star + epsilon_star

        # --- Step 3:  J_new(x) = C_s(x) + phi*( f_s(x) ) ---
        s = dual_grid.ravel()
        J_new = np.empty(len(X))
        for i, x in enumerate(X):
            z = float(np.asarray(prob.f_s(x)).ravel()[0])
            phi_star_val = np.max(s * z - phi)          # 1-D conjugate of phi at z
            J_new[i] = float(np.asarray(prob.C_s(x)).ravel()[0]) + phi_star_val
        return J_new


# ----------------------------------------------------------------------
# 6. Toy problem
# ----------------------------------------------------------------------
class ToyProblem(MDPProblem):
    def __init__(self):
        self._gamma = 0.95
        self._B = np.array([[1.0]])
        self._X = np.linspace(-2, 2, 21).reshape(-1, 1)
        self._U = np.linspace(-1, 1, 11).reshape(-1, 1)
        self._W = np.array([[-0.1], [0.0], [0.1]])
        self._P = np.array([0.25, 0.5, 0.25])

    @property
    def gamma(self): return self._gamma
    @property
    def B(self): return self._B
    @property
    def state_grid(self): return self._X
    @property
    def input_grid(self): return self._U
    @property
    def disturbance_support(self): return self._W
    @property
    def disturbance_pmf(self): return self._P

    def f_s(self, x): return 0.9 * x
    def C_s(self, x): return x ** 2
    def C_i(self, u): return 0.1 * u ** 2


# ----------------------------------------------------------------------
# 7. Driver
# ----------------------------------------------------------------------
if __name__ == "__main__":
    problem = ToyProblem()

    # ---- Fixed dual grid of 41 points in [-3, 3] ----
    dual_grid = np.linspace(-3.0, 3.0, 41)
    llt = LLTConjugate(dual_grid)

    vi = PrimalVI(problem)
    conj_vi = ConjVI(problem, llt)

    J_vi = vi.solve()
    J_conj = conj_vi.solve()

    print("Primal VI  (first 5):", J_vi[:5])
    print("ConjVI     (first 5):", J_conj[:5])
    print("Max |VI - ConjVI|   :", np.max(np.abs(J_vi - J_conj)))