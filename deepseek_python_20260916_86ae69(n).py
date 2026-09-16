"""
Polymorphic ConjVI implementation (arXiv:2102.08880v4) with Matplotlib plots.
"""
from __future__ import annotations
import abc
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


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
    """Brute-force 1-D Legendre–Fenchel conjugate on a fixed dual grid."""

    def __init__(self, dual_grid: np.ndarray):
        self.dual_grid = np.asarray(dual_grid, dtype=float).reshape(-1, 1)

    def conjugate(self, f, grid):
        g = np.asarray(grid, dtype=float).ravel()
        f = np.asarray(f, dtype=float).ravel()
        s = self.dual_grid.ravel()
        # f*(s) = max_x ( s*x - f(x) )
        f_star = np.array([np.max(si * g - f) for si in s])
        return self.dual_grid.copy(), f_star


# ----------------------------------------------------------------------
# 3. Value-iteration abstraction
# ----------------------------------------------------------------------
class ValueIterationAlgorithm(abc.ABC):
    def __init__(self, problem: MDPProblem, tol: float = 1e-6, max_iter: int = 200):
        self.problem = problem
        self.tol = tol
        self.max_iter = max_iter
        self.history: List[float] = []          # <- NEW: residual per iteration

    @abc.abstractmethod
    def dp_operator(self, J: np.ndarray) -> np.ndarray: ...

    def solve(self, J0: Optional[np.ndarray] = None) -> np.ndarray:
        J = np.zeros(len(self.problem.state_grid)) if J0 is None else J0.copy()
        self.history.clear()
        for _ in range(self.max_iter):
            J_new = self.dp_operator(J)
            res = float(np.max(np.abs(J_new - J)))
            self.history.append(res)
            if res < self.tol:
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
            costs = prob.C_s(x) + prob.C_i(U)
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
# 5. Conjugate VI
# ----------------------------------------------------------------------
class ConjVI(ValueIterationAlgorithm):
    def __init__(self, problem, conj_transform, tol=1e-6, max_iter=200):
        super().__init__(problem, tol, max_iter)
        self.conj = conj_transform
        self._Ci_star = None

    def _ensure_input_conjugate(self):
        if self._Ci_star is not None:
            return
        U = self.problem.input_grid
        Ci_vals = np.asarray(self.problem.C_i(U), dtype=float).ravel()
        self._Ci_star = self.conj.conjugate(Ci_vals, U)

    def dp_operator(self, J):
        prob = self.problem
        X = prob.state_grid
        W, P = prob.disturbance_support, prob.disturbance_pmf
        gamma = prob.gamma

        self._ensure_input_conjugate()
        dual_grid, Ci_star = self._Ci_star

        # Step 1:  ε(x) = γ · E_w J(x+w)
        epsilon = np.zeros(len(X))
        for i, x in enumerate(X):
            ev = 0.0
            for w, p in zip(W, P):
                xn = x + w
                idx = np.argmin(np.linalg.norm(X - xn, axis=1))
                ev += p * J[idx]
            epsilon[i] = gamma * ev

        # Step 2:  φ = C_i*(·) + ε*(·)
        _, epsilon_star = self.conj.conjugate(epsilon, X)
        phi = Ci_star + epsilon_star

        # Step 3:  J_new(x) = C_s(x) + φ*( f_s(x) )
        s = dual_grid.ravel()
        J_new = np.empty(len(X))
        for i, x in enumerate(X):
            z = float(np.asarray(prob.f_s(x)).ravel()[0])
            phi_star_val = np.max(s * z - phi)
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
# 7. Driver + Matplotlib visualisation
# ----------------------------------------------------------------------
def main():
    problem = ToyProblem()
    dual_grid = np.linspace(-3.0, 3.0, 41)
    llt = LLTConjugate(dual_grid)

    vi = PrimalVI(problem, tol=1e-8, max_iter=200)
    conj_vi = ConjVI(problem, llt, tol=1e-8, max_iter=200)

    J_vi = vi.solve()
    J_conj = conj_vi.solve()

    x = problem.state_grid.ravel()
    diff = np.abs(J_vi - J_conj)

    print(f"Primal VI  first 5: {J_vi[:5]}")
    print(f"ConjVI     first 5: {J_conj[:5]}")
    print(f"Max |VI - ConjVI| : {diff.max():.3e}")
    print(f"VI iterations     : {len(vi.history)}")
    print(f"ConjVI iterations : {len(conj_vi.history)}")

    # ---------- Plots ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("ConjVI vs Primal VI  (arXiv:2102.08880v4)",
                 fontsize=14, fontweight="bold")

    # (a) Value functions
    ax = axes[0, 0]
    ax.plot(x, J_vi, "o-", label="Primal VI", color="tab:blue")
    ax.plot(x, J_conj, "s--", label="ConjVI", color="tab:orange")
    ax.set_xlabel("state x")
    ax.set_ylabel("J(x)")
    ax.set_title("(a) Optimal value function")
    ax.grid(alpha=0.3)
    ax.legend()

    # (b) Absolute difference
    ax = axes[0, 1]
    ax.semilogy(x, np.maximum(diff, 1e-16), "d-", color="tab:red")
    ax.set_xlabel("state x")
    ax.set_ylabel("|J_VI(x) − J_ConjVI(x)|")
    ax.set_title("(b) Pointwise absolute difference")
    ax.grid(alpha=0.3, which="both")

    # (c) Convergence history
    ax = axes[1, 0]
    ax.semilogy(vi.history, "-", label="Primal VI", color="tab:blue")
    ax.semilogy(conj_vi.history, "--", label="ConjVI", color="tab:orange")
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$\|J_{k+1} - J_k\|_\infty$")
    ax.set_title("(c) Convergence residual")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # (d) Value function on log scale vs |x|
    ax = axes[1, 1]
    ax.plot(x, J_vi, "-", color="tab:blue", label="Primal VI")
    ax.plot(x, J_conj, "--", color="tab:orange", label="ConjVI")
    ax.plot(x, x ** 2, ":", color="gray", label=r"$C_s(x)=x^2$")
    ax.set_xlabel("state x")
    ax.set_ylabel("J(x)")
    ax.set_title("(d) Value function vs state cost")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("conjvi_results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()