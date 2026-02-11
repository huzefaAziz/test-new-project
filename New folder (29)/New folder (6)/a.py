# Binary: one 1 + n zeros = 2^n. Reuse zeros infinitely (e.g. 100000 = 32).
# No colour, no approximation.

def binary_one_plus_zeros(n_zeros):
    """Single 1 + n zeros in binary = 2^n. One bit represents value with reused zeros."""
    return 1 << n_zeros  # 1000...0 (n zeros) = 2^n; e.g. 1<<5 = 32


class CompressedFib:
    """
    Recursive Fibonacci using structural reuse (fast doubling).
    Supports huge numbers without converting full integer to string.
    No colour, no approximation.
    """

    def __init__(self, n):
        self.n = n

    def compute(self):
        return self._fib(self.n)[0]

    def _fib(self, k):
        if k == 0:
            return (0, 1)
        a, b = self._fib(k >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        if k & 1:
            return (d, c + d)
        return (c, d)

    def first_digits(self, num_digits=10):
        """First num_digits without converting entire huge integer to string."""
        from math import log10, floor
        n_val = self.compute()
        if n_val == 0:
            return "0"
        digits = floor(log10(n_val)) + 1
        factor = 10 ** (digits - num_digits)
        return str(n_val // factor)

    def last_digits(self, num_digits=10):
        """Last num_digits using modulo."""
        n_val = self.compute()
        return str(n_val % (10 ** num_digits)).zfill(num_digits)


if __name__ == "__main__":
    # Example: binary 100000 = 32 (one 1, five zeros)
    print("Binary 1+zeros:", "100000 =", binary_one_plus_zeros(5))

    fib_large = CompressedFib(100000)
    print("First 15 digits:", fib_large.first_digits(15))
