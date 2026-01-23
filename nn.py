"""
Turing Machine Simulator with Semi-Infinite Tape
Computes Fibonacci sequence recursively using fast doubling method
"""

class SemiInfiniteTape:
    """A semi-infinite tape that only supports non-negative positions (0, 1, 2, ...)"""
    
    def __init__(self):
        self.tape = {}  # Sparse representation: position -> value
    
    def read(self, position):
        """Read value at position. Returns 0 if position is empty."""
        if position < 0:
            raise IndexError("Semi-infinite tape: negative positions not allowed")
        return self.tape.get(position, 0)
    
    def write(self, position, value):
        """Write value at position."""
        if position < 0:
            raise IndexError("Semi-infinite tape: negative positions not allowed")
        self.tape[position] = value
    
    def get_occupied_range(self):
        """Get the range of occupied positions."""
        if not self.tape:
            return (0, 0)
        positions = sorted(self.tape.keys())
        return (positions[0], positions[-1])


class TuringMachine:
    """Turing Machine with semi-infinite tape for computing Fibonacci"""
    
    def __init__(self):
        self.tape = SemiInfiniteTape()
        self.head = 0
        self.state = 'START'
    
    def compute_fibonacci_recursive(self, n, base_pos=0):
        """
        Compute Fibonacci(n) recursively using fast doubling method.
        Uses semi-infinite tape for storage without memoization.
        
        Fast doubling formulas:
        F(2k) = F(k) * (2*F(k+1) - F(k))
        F(2k+1) = F(k+1)^2 + F(k)^2
        
        Args:
            n: Fibonacci number to compute
            base_pos: Base position on tape for storing intermediate results
            
        Returns:
            (F(n), F(n+1)) tuple
        """
        # Base cases: F(0) = 0, F(1) = 1
        if n == 0:
            self.tape.write(base_pos, 0)
            self.tape.write(base_pos + 1, 1)
            return 0, 1
        
        if n == 1:
            self.tape.write(base_pos, 1)
            self.tape.write(base_pos + 1, 1)
            return 1, 1
        
        # Recursive case: compute F(k) and F(k+1) where k = n//2
        k = n // 2
        a, b = self.compute_fibonacci_recursive(k, base_pos + 1000)  # Offset to avoid conflicts
        
        # Fast doubling: compute F(2k) and F(2k+1)
        c = a * (2 * b - a)  # F(2k)
        d = a * a + b * b    # F(2k+1)
        
        # Store results on tape
        self.tape.write(base_pos, a)
        self.tape.write(base_pos + 1, b)
        self.tape.write(base_pos + 2, c)
        self.tape.write(base_pos + 3, d)
        
        # If n is even, return F(2k) and F(2k+1)
        # If n is odd, we need F(2k+1) and F(2k+2) = F(2k+1) + F(2k)
        if n % 2 == 0:
            return c, d
        else:
            return d, c + d
    
    def fibonacci(self, n):
        """Compute the nth Fibonacci number."""
        if n < 0:
            raise ValueError("Fibonacci is only defined for non-negative integers")
        
        self.tape = SemiInfiniteTape()
        self.head = 0
        result, _ = self.compute_fibonacci_recursive(n, base_pos=0)
        return result


def main():
    """Demonstrate the Turing Machine computing Fibonacci sequence"""
    print("Turing Machine with Semi-Infinite Tape")
    print("Computing Fibonacci sequence recursively (fast doubling method)\n")
    print("=" * 60)
    
    tm = TuringMachine()
    
    # Compute first 20 Fibonacci numbers
    print("First 20 Fibonacci numbers:")
    for i in range(20):
        fib = tm.fibonacci(i)
        print(f"F({i:2d}) = {fib:>10}")
    
    print("\n" + "=" * 60)
    
    # Compute larger Fibonacci numbers
    print("\nComputing larger Fibonacci numbers:")
    test_values = [30, 40, 50, 100]
    for n in test_values:
        fib = tm.fibonacci(n)
        print(f"F({n:3d}) = {fib}")
    
    print("\n" + "=" * 60)
    print(f"\nTape usage example for F(10):")
    tm.fibonacci(10)
    start, end = tm.tape.get_occupied_range()
    print(f"Occupied tape positions: {start} to {end}")
    print(f"Total occupied cells: {len(tm.tape.tape)}")


if __name__ == "__main__":
    main()


