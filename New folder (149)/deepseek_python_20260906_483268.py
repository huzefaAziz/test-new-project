import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# ------------------------------------------------------------
# 1. Cyclotomic Ring O_K = Z[ζ_q] for q ≡ 3 mod 4
# ------------------------------------------------------------

@dataclass
class CyclotomicElement:
    q: int
    coeffs: List[int]  # length q-1

    def __post_init__(self):
        if len(self.coeffs) != self.q - 1:
            raise ValueError(f"Expected {self.q-1} coefficients, got {len(self.coeffs)}")

    def __add__(self, other):
        if self.q != other.q:
            raise ValueError("Different cyclotomic fields")
        n = self.q - 1
        return CyclotomicElement(self.q, [self.coeffs[i] + other.coeffs[i] for i in range(n)])

    def __neg__(self):
        return CyclotomicElement(self.q, [-c for c in self.coeffs])

    def __mul__(self, other):
        """Multiplication in Z[ζ_q] with reduction using 1+ζ+...+ζ^(q-1)=0."""
        if isinstance(other, int):
            # Scalar multiplication
            return CyclotomicElement(self.q, [c * other for c in self.coeffs])
        if not isinstance(other, CyclotomicElement):
            raise TypeError(f"Unsupported type: {type(other)}")
        if self.q != other.q:
            raise ValueError("Different cyclotomic fields")
        n = self.q - 1
        conv = [0] * (2 * n - 1)
        for i in range(n):
            for j in range(n):
                conv[i + j] += self.coeffs[i] * other.coeffs[j]
        result = [0] * n
        for i in range(2 * n - 1):
            if i < n:
                result[i] += conv[i]
            else:
                for j in range(n):
                    result[j] -= conv[i]
        return CyclotomicElement(self.q, result)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def norm_squared(self) -> float:
        return self.q * sum(c * c for c in self.coeffs)


# ------------------------------------------------------------
# 2. Rank-2 Module over O_K
# ------------------------------------------------------------

@dataclass
class ModuleVector:
    q: int
    x1: CyclotomicElement
    x2: CyclotomicElement

    def norm_squared(self) -> float:
        return self.x1.norm_squared() + self.x2.norm_squared()


@dataclass
class RankTwoModule:
    q: int
    m1: ModuleVector
    m2: ModuleVector

    def contains(self, v: ModuleVector) -> bool:
        raise NotImplementedError("Membership test requires solving over O_K")


# ------------------------------------------------------------
# 3. X3C Problem and Reduction
# ------------------------------------------------------------

@dataclass
class X3CInstance:
    A: List[List[int]]  # M rows, n columns; each column has exactly three 1s
    n: int
    M: int

    def __post_init__(self):
        self.M = len(self.A)
        self.n = len(self.A[0]) if self.M > 0 else 0
        for j in range(self.n):
            if sum(self.A[i][j] for i in range(self.M)) != 3:
                raise ValueError(f"Column {j} does not have exactly 3 ones")


def x3c_to_svp_instance(x3c: X3CInstance) -> Tuple[int, ModuleVector, ModuleVector, int]:
    """Polynomial-time reduction from X3C to SVP on rank-2 cyclotomic modules."""
    q = _choose_prime(x3c)
    alphas = _choose_distinct_field_elements(q, x3c.n)
    k = _choose_k(q, x3c)
    c_elem = _choose_coset_representative(q, k)
    U, V = _construct_checker_constants(q, x3c, alphas)
    Gamma = _choose_Gamma(q, x3c)

    # π = 1 - ζ
    pi = CyclotomicElement(q, [1] + [-1] * (q - 2))
    pi_k = _power(pi, k)

    # m1 = (U·π^k, 0)
    U_pi_k = U * pi_k
    m1 = ModuleVector(q, U_pi_k, CyclotomicElement(q, [0] * (q - 1)))

    # m2 = (U·c - V, -Γ)
    U_c = U * c_elem
    U_c_minus_V = U_c + (-V)          # U·c - V
    Gamma_elem = CyclotomicElement(q, [Gamma] + [0] * (q - 2))
    m2 = ModuleVector(q, U_c_minus_V, -Gamma_elem)   # second coord = -Γ

    B0 = _compute_B0(q, x3c, U, V)
    S = B0 + (Gamma ** 2) * (q - 1)

    return q, m1, m2, S


# ------------------------------------------------------------
# 4. Helper Functions (sketches – not fully implemented)
# ------------------------------------------------------------

def _choose_prime(x3c: X3CInstance) -> int:
    candidate = max(7, x3c.n + x3c.M + 1)
    while True:
        if candidate % 4 == 3 and _is_prime(candidate):
            return candidate
        candidate += 1

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def _choose_distinct_field_elements(q: int, n: int) -> List[int]:
    return list(range(1, n + 1))

def _choose_k(q: int, x3c: X3CInstance) -> int:
    return 2 * (x3c.M + x3c.n) + 10

def _choose_coset_representative(q: int, k: int) -> CyclotomicElement:
    coeffs = [random.randint(0, 1) for _ in range(q - 1)]
    return CyclotomicElement(q, coeffs)

def _construct_checker_constants(q: int, x3c: X3CInstance,
                                 alphas: List[int]) -> Tuple[CyclotomicElement, CyclotomicElement]:
    # Placeholder – in reality this uses Hasse derivatives and interpolation
    coeffs_U = [random.randint(0, 1) for _ in range(q - 1)]
    coeffs_V = [random.randint(0, 1) for _ in range(q - 1)]
    return CyclotomicElement(q, coeffs_U), CyclotomicElement(q, coeffs_V)

def _choose_Gamma(q: int, x3c: X3CInstance) -> int:
    return (x3c.M + x3c.n) ** 2 + 10

def _power(base: CyclotomicElement, exp: int) -> CyclotomicElement:
    result = CyclotomicElement(base.q, [1] + [0] * (base.q - 2))
    for _ in range(exp):
        result = result * base
    return result

def _compute_B0(q: int, x3c: X3CInstance,
                U: CyclotomicElement, V: CyclotomicElement) -> int:
    # Placeholder – should be derived from the checker identity
    return (q - 1) * (x3c.M + 1)


# ------------------------------------------------------------
# 5. Corrected Example: Valid X3C instance
# ------------------------------------------------------------

if __name__ == "__main__":
    # Universe of size M=3, one set containing all three elements.
    # This is a YES instance (choose that set).
    A = [
        [1],
        [1],
        [1]
    ]
    x3c = X3CInstance(A, n=1, M=3)

    # Run the reduction (will use random choices for the checker constants)
    q, m1, m2, S = x3c_to_svp_instance(x3c)

    print(f"q = {q}")
    print(f"m1 = ({m1.x1.coeffs}, {m1.x2.coeffs})")
    print(f"m2 = ({m2.x1.coeffs}, {m2.x2.coeffs})")
    print(f"S = {S}")
    print(f"λ₁(M)² ≤ S  ⇔  X3C instance is YES")