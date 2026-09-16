from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


# ----------------------------------------------------------------------
# 1. Logconcave distribution interface
# ----------------------------------------------------------------------
class LogConcaveDistribution(ABC):
    @abstractmethod
    def V(self, x: np.ndarray) -> float: ...
    @abstractmethod
    def grad_V(self, x: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def sample_prior(self) -> np.ndarray: ...


# ----------------------------------------------------------------------
# 2. Concrete distributions
# ----------------------------------------------------------------------
class GaussianDistribution(LogConcaveDistribution):
    def __init__(self, dim: int):
        self.dim = dim

    def V(self, x: np.ndarray) -> float:
        return 0.5 * float(np.dot(x, x))

    def grad_V(self, x: np.ndarray) -> np.ndarray:
        return x

    def sample_prior(self) -> np.ndarray:
        return np.random.randn(self.dim)


class UniformBall(LogConcaveDistribution):
    def __init__(self, dim: int):
        self.dim = dim

    def V(self, x: np.ndarray) -> float:
        return 0.0 if np.linalg.norm(x) <= 1.0 else np.inf

    def grad_V(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Indicator function has no gradient.")

    def sample_prior(self) -> np.ndarray:
        while True:
            x = np.random.uniform(-1, 1, self.dim)
            if np.linalg.norm(x) <= 1.0:
                return x


class GeneralLogConcave(LogConcaveDistribution):
    def __init__(self, dim: int, V_func, grad_V_func):
        self.dim = dim
        self._V = V_func
        self._grad_V = grad_V_func

    def V(self, x: np.ndarray) -> float:
        return self._V(x)

    def grad_V(self, x: np.ndarray) -> np.ndarray:
        return self._grad_V(x)

    def sample_prior(self) -> np.ndarray:
        raise NotImplementedError


# ----------------------------------------------------------------------
# 3. Gaussian tilt — now itself a LogConcaveDistribution
# ----------------------------------------------------------------------
class GaussianTilt(LogConcaveDistribution):
    """
    Tilted measure
        π_t(dx) ∝ exp(−‖x‖² / (2t)) π(dx).

    It IS-A LogConcaveDistribution, so any Sampler can operate on it
    through the exact same interface as on the base distribution.
    """

    def __init__(self, base: LogConcaveDistribution, t: float):
        if t <= 0:
            raise ValueError("Tilt parameter t must be positive.")
        self.base = base
        self.t = t

    def V(self, x: np.ndarray) -> float:
        return self.base.V(x) + float(np.dot(x, x)) / (2.0 * self.t)

    def grad_V(self, x: np.ndarray) -> np.ndarray:
        return self.base.grad_V(x) + x / self.t

    def sample_prior(self) -> np.ndarray:
        # A tilted distribution generally has no closed-form prior
        # we can sample from, but we can delegate to the base if
        # it exists.
        return self.base.sample_prior()


# ----------------------------------------------------------------------
# 4. Samplers (polymorphic)
# ----------------------------------------------------------------------
class Sampler(ABC):
    @abstractmethod
    def step(self, x: np.ndarray, dist: LogConcaveDistribution) -> np.ndarray: ...

    def sample(
        self,
        dist: LogConcaveDistribution,
        x0: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        x = x0.copy()
        for _ in range(n_steps):
            x = self.step(x, dist)
        return x


class BallWalk(Sampler):
    def __init__(self, delta: float = 0.1):
        self.delta = delta

    def step(self, x: np.ndarray, dist: LogConcaveDistribution) -> np.ndarray:
        dim = x.shape[0]
        u = np.random.randn(dim)
        u /= np.linalg.norm(u)
        y = x + self.delta * u

        # Metropolis acceptance on the log-density −V
        log_alpha = -dist.V(y) + dist.V(x)
        if np.log(np.random.rand()) < log_alpha:
            return y
        return x


class HitAndRun(Sampler):
    def __init__(self, line_length: float = 1.0):
        self.line_length = line_length

    def step(self, x: np.ndarray, dist: LogConcaveDistribution) -> np.ndarray:
        dim = x.shape[0]
        direction = np.random.randn(dim)
        direction /= np.linalg.norm(direction)

        t = np.random.uniform(-self.line_length, self.line_length)
        y = x + t * direction

        log_alpha = -dist.V(y) + dist.V(x)
        if np.log(np.random.rand()) < log_alpha:
            return y
        return x


# ----------------------------------------------------------------------
# 5. Gaussian Cooling annealer
# ----------------------------------------------------------------------
class GaussianCooling:
    def __init__(
        self,
        base_dist: LogConcaveDistribution,
        sampler: Sampler,
        t_start: float = 10.0,
        t_end: float = 0.1,
        n_stages: int = 20,
        steps_per_stage: int = 100,
    ):
        self.base = base_dist
        self.sampler = sampler
        self.t_start = t_start
        self.t_end = t_end
        self.n_stages = n_stages
        self.steps_per_stage = steps_per_stage

    def run(self, x0: Optional[np.ndarray] = None) -> np.ndarray:
        dim = self.base.sample_prior().shape[0] if x0 is None else x0.shape[0]
        x = np.zeros(dim) if x0 is None else x0.copy()

        ts = np.geomspace(self.t_start, self.t_end, self.n_stages)
        for t in ts:
            # GaussianTilt IS-A LogConcaveDistribution, so the
            # sampler can consume it directly without any adaptation.
            tilted = GaussianTilt(self.base, t)
            x = self.sampler.sample(tilted, x, self.steps_per_stage)

        return x


# ----------------------------------------------------------------------
# 6. Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dim = 5
    dist = GaussianDistribution(dim)
    walk = BallWalk(delta=0.2)

    cooler = GaussianCooling(
        base_dist=dist,
        sampler=walk,
        t_start=5.0,
        t_end=0.1,
        n_stages=10,
        steps_per_stage=50,
    )

    sample = cooler.run(x0=np.zeros(dim))
    print("Sample from the target distribution:", sample)