"""
Binary Circular Recall Multiplication
=====================================
Core rule: multiplier = 9 (configurable).
Each recalled '1' bit triggers one addition of the multiplier.
The binary pattern lives in a circular memory that can be rotated,
so the same pattern can be recalled from different starting positions.
"""

from collections import deque
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. BinaryCircularMemory class
# ============================================================
class BinaryCircularMemory:
    """A circular buffer that stores a binary pattern and supports
    rotation, recall, and 1-bit counting."""

    def __init__(self, pattern: str = ""):
        self._buffer = deque()
        if pattern:
            self.store(pattern)

    # ---- 2. Store a binary pattern ----
    def store(self, pattern: str) -> None:
        """Store a binary string (e.g. '101') into the circular memory."""
        if not all(c in "01" for c in pattern):
            raise ValueError("Pattern must contain only '0' and '1'.")
        self._buffer = deque(pattern)

    @property
    def pattern(self) -> str:
        return "".join(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"BinaryCircularMemory(pattern='{self.pattern}')"

    # ---- 3. Recall / search function ----
    def recall(self, start: int = 0, length: int | None = None) -> str:
        """Recall a sub-pattern starting at position `start` (wraps around).
        If length is None, recall the whole pattern from that start."""
        if len(self._buffer) == 0:
            return ""
        n = len(self._buffer)
        start = start % n
        length = length if length is not None else n
        recalled = []
        for i in range(length):
            recalled.append(self._buffer[(start + i) % n])
        return "".join(recalled)

    # ---- 4. Count recalled 1-bits ----
    def count_ones(self, start: int = 0, length: int | None = None) -> int:
        """Count how many '1' bits are in the recalled segment."""
        return self.recall(start, length).count("1")

    # ---- 6. Circular rotation ----
    def rotate(self, steps: int = 1) -> str:
        """Rotate the circular memory by `steps` positions to the left.
        Returns the new pattern."""
        self._buffer.rotate(-steps)
        return self.pattern

    def rotated_view(self, steps: int) -> str:
        """Non-destructive rotation: returns what the pattern would look
        like after rotating by `steps`, without mutating the buffer."""
        d = deque(self._buffer)
        d.rotate(-steps)
        return "".join(d)

    # ---- 7. Visualization ----
    def visualize(self, title: str = "Circular Binary Memory", ax=None):
        """Draw the binary pattern arranged on a circle."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        n = len(self._buffer)
        if n == 0:
            ax.text(0, 0, "empty", ha="center", va="center")
            ax.set_title(title)
            return fig

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # Start from the top (pi/2) and go clockwise
        angles = np.pi / 2 - angles
        xs = np.cos(angles)
        ys = np.sin(angles)

        # Draw the circle
        circle = plt.Circle((0, 0), 1.0, fill=False,
                            color="gray", linestyle="--")
        ax.add_patch(circle)

        # Draw each bit
        for i, bit in enumerate(self._buffer):
            color = "crimson" if bit == "1" else "steelblue"
            ax.scatter(xs[i], ys[i], s=1400, color=color,
                       edgecolor="black", zorder=3)
            ax.text(xs[i], ys[i], bit, ha="center", va="center",
                    fontsize=16, fontweight="bold", color="white",
                    zorder=4)
            # index label outside
            ax.text(1.25 * xs[i], 1.25 * ys[i], f"[{i}]",
                    ha="center", va="center", fontsize=9, color="gray")

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)
        return fig


# ============================================================
# 5. Multiplication via REPEATED ADDITION (no Python '*')
# ============================================================
def recall_multiply(num_ones: int, multiplier: int = 9,
                    verbose: bool = True) -> int:
    """Compute num_ones * multiplier using repeated addition.
    Each recalled '1' bit contributes one `multiplier` to the sum.

    This is intentionally NOT using Python's '*' operator — the whole
    point of the algorithm is that multiplication emerges from
    recall-triggered additions.
    """
    if num_ones < 0:
        raise ValueError("num_ones must be non-negative")

    result = 0
    steps = []
    for i in range(num_ones):
        result += multiplier          # one addition per recalled '1'
        steps.append(result)

    if verbose:
        if num_ones == 0:
            print(f"  recall_multiply: 0 ones × {multiplier} = 0")
        else:
            additions = " + ".join([str(multiplier)] * num_ones)
            print(f"  recall_multiply: {additions} = {result}")
            print(f"    running totals: {steps}")
    return result


# ============================================================
# High-level: recall a pattern and multiply
# ============================================================
def recall_and_multiply(mem: BinaryCircularMemory,
                        start: int = 0,
                        length: int | None = None,
                        multiplier: int = 9,
                        verbose: bool = True) -> int:
    """Recall a segment from circular memory, count its 1-bits,
    and multiply that count by `multiplier` via repeated addition."""
    recalled = mem.recall(start, length)
    ones = recalled.count("1")
    if verbose:
        print(f"  recalled pattern : '{recalled}'")
        print(f"  number of 1s     : {ones}")
    return recall_multiply(ones, multiplier, verbose=verbose)


# ============================================================
# 8. Demonstration with 1, 11, 101, 111, 1101
# ============================================================
def demo_basic():
    print("=" * 60)
    print("DEMO: Binary Circular Recall Multiplication (multiplier=9)")
    print("=" * 60)

    patterns = ["1", "11", "101", "111", "1101"]
    for p in patterns:
        print(f"\n--- pattern: '{p}' ---")
        mem = BinaryCircularMemory(p)
        result = recall_and_multiply(mem, multiplier=9)
        print(f"  => final result = {result}")


def demo_rotation():
    print("\n" + "=" * 60)
    print("DEMO: Circular rotation — same pattern, different starts")
    print("=" * 60)

    mem = BinaryCircularMemory("1101")
    print(f"Stored pattern: '{mem.pattern}'")
    for start in range(len(mem)):
        print(f"\n[recall from start={start}]")
        recall_and_multiply(mem, start=start, length=len(mem),
                            multiplier=9)


def demo_visualization():
    print("\n" + "=" * 60)
    print("DEMO: Visualization of circular memory")
    print("=" * 60)

    patterns = ["1", "11", "101", "111", "1101"]
    fig, axes = plt.subplots(1, len(patterns), figsize=(4 * len(patterns), 4))
    for ax, p in zip(axes, patterns):
        mem = BinaryCircularMemory(p)
        mem.visualize(title=f"pattern '{p}'", ax=ax)
    plt.tight_layout()
    plt.show()


# ============================================================
# 9. Using the algorithm to accelerate recursive Fibonacci
# ============================================================
# Idea: the "circular recall" becomes a memoization ring.
# Before computing fib(n), we RECALL from the circular memory.
# If the value is already there, we skip the recursion.
# Otherwise we compute, STORE it, and continue.
# This turns the exponential recursive Fibonacci into linear time.
# ============================================================
class FibonacciCircularCache:
    """A circular memory used as a memoization ring for Fibonacci.
    Each slot stores (index, value). A 'recall' checks whether
    fib(n) is already cached."""

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.slots: dict[int, int] = {}   # index -> fib(index)
        self.ring = deque(maxlen=capacity)  # ordered history
        self.recall_hits = 0
        self.recall_misses = 0

    def recall(self, n: int) -> int | None:
        """Try to recall fib(n). Returns the value or None."""
        if n in self.slots:
            self.recall_hits += 1
            return self.slots[n]
        self.recall_misses += 1
        return None

    def store(self, n: int, value: int) -> None:
        """Store a freshly computed fib(n) into the ring."""
        self.slots[n] = value
        self.ring.append(n)

    def stats(self) -> str:
        return (f"recall hits={self.recall_hits}, "
                f"misses={self.recall_misses}, "
                f"cache size={len(self.slots)}")


def fib_recursive_cached(n: int, cache: FibonacciCircularCache,
                         multiplier: int = 9,
                         verbose: bool = True) -> int:
    """Recursive Fibonacci accelerated by the circular recall cache.

    The multiplier is woven in as a 'recall cost': every time we
    recall a cached value we conceptually pay 1 × multiplier via
    repeated addition — this keeps the 'binary recall multiplication'
    spirit alive inside the Fibonacci computation.
    """
    # Try to RECALL first
    cached = cache.recall(n)
    if cached is not None:
        if verbose:
            print(f"  fib({n}) -> RECALL HIT = {cached}")
            # Recall cost: 1 one-bit × multiplier
            recall_and_multiply(
                BinaryCircularMemory("1"),
                multiplier=multiplier, verbose=False)
        return cached

    if verbose:
        print(f"  fib({n}) -> RECALL MISS, computing...")

    # Base cases
    if n <= 1:
        result = n
    else:
        a = fib_recursive_cached(n - 1, cache, multiplier, verbose)
        b = fib_recursive_cached(n - 2, cache, multiplier, verbose)
        result = a + b

    # STORE the computed value into the circular cache
    cache.store(n, result)
    return result


def demo_fibonacci():
    print("\n" + "=" * 60)
    print("DEMO: Fibonacci accelerated by circular recall cache")
    print("=" * 60)

    cache = FibonacciCircularCache(capacity=64)

    # First compute fib(0)..fib(10) with verbose output
    print("\n--- computing fib(0) to fib(10) ---")
    for n in range(11):
        print(f"\n>> fib({n}):")
        value = fib_recursive_cached(n, cache, multiplier=9, verbose=True)
        print(f"   fib({n}) = {value}")

    print(f"\nCache stats after first run: {cache.stats()}")

    # Now ask for fib(10) again — everything should be RECALL HITS
    print("\n--- re-asking fib(10) (should all be recall hits) ---")
    print(f">> fib(10):")
    value = fib_recursive_cached(10, cache, multiplier=9, verbose=True)
    print(f"   fib(10) = {value}")

    print(f"\nFinal cache stats: {cache.stats()}")

    # Scale up: fib(30) — would be impossibly slow without the cache
    print("\n--- scaling up: fib(30) ---")
    value = fib_recursive_cached(30, cache, multiplier=9, verbose=False)
    print(f"   fib(30) = {value}")
    print(f"   Cache stats: {cache.stats()}")


# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    demo_basic()
    demo_rotation()
    demo_fibonacci()
    demo_visualization()   # comment out if running headless