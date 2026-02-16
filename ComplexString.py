from typing import Sequence, Union

class StringMixin:
    def upper(self):
        return str(self).upper()

class ComplexString(StringMixin, complex):
    """Complex number with string mixin; usable as scalar in Hilbert space."""
    pass


# ----- Hilbert space (finite-dimensional over C) -----

Scalar = Union[complex, ComplexString]

def _to_complex(z: Scalar) -> complex:
    return complex(z) if isinstance(z, ComplexString) else z

class HilbertSpace:
    """Finite-dimensional Hilbert space H over C: inner product conjugate-linear in 1st arg."""
    def __init__(self, dimension: int):
        self.dimension = dimension

    def inner(self, u: Sequence[Scalar], v: Sequence[Scalar]) -> complex:
        """<u|v> = sum_i conjugate(u_i) * v_i (conjugate-linear in first argument)."""
        if len(u) != self.dimension or len(v) != self.dimension:
            raise ValueError("vector length must match space dimension")
        return sum(_to_complex(u[i]).conjugate() * _to_complex(v[i]) for i in range(self.dimension))

    def norm(self, v: Sequence[Scalar]) -> float:
        """||v|| = sqrt(<v|v>)."""
        return (self.inner(v, v).real) ** 0.5

    def normalize(self, v: Sequence[Scalar]) -> list[complex]:
        """Return v / ||v|| (unit vector)."""
        n = self.norm(v)
        if n == 0:
            raise ValueError("cannot normalize zero vector")
        return [_to_complex(z) / n for z in v]

    def project(self, v: Sequence[Scalar], onto: Sequence[Scalar]) -> list[complex]:
        """Project v onto unit vector 'onto': <onto|v> * onto (onto should be normalized)."""
        return [_to_complex(self.inner(onto, v) * z) for z in onto]


# ----- Example: use ComplexString as scalars in H -----
if __name__ == "__main__":
    # ComplexString as Hilbert space scalars
    print(ComplexString(1, 2).upper())   # "(1+2J)"
    print(ComplexString("3+4j").upper()) # "(3+4J)"

    H = HilbertSpace(2)
    u = [ComplexString(1, 0), ComplexString(0, 1)]   # |u> = (1, i)
    v = [ComplexString(1, 0), ComplexString(0, -1)]   # |v> = (1, -i)
    print("inner <u|v> =", H.inner(u, v))            # 1*1 + (-i)*(-i) = 1 - 1 = 0 (orthogonal)
    print("||u|| =", H.norm(u))                      # sqrt(2)
    w = H.normalize(u)
    print("normalized u:", w)