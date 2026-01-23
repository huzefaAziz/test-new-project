class SemiInfiniteTape:
    def __init__(self):
        self.tape = {}

    def read(self, i):
        return self.tape.get(i, 0)

    def write(self, i, v):
        self.tape[i] = v
def fib_fast(n, tape, base=0):
    if n == 0:
        tape.write(base, 0)
        tape.write(base + 1, 1)
        return 0, 1

    a, b = fib_fast(n // 2, tape, base)

    c = a * (2 * b - a)
    d = a * a + b * b

    tape.write(base + 2, c)
    tape.write(base + 3, d)

    if n % 2 == 0:
        return c, d
    else:
        return d, c + d
def fibonacci(n):
    tape = SemiInfiniteTape()
    return fib_fast(n, tape)[0]
for i in range(1500000):
    print(i, fibonacci(i))
