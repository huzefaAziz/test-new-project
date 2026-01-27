class Fusion:
    def __init__(self, a, b):
        self._a = a
        self._b = b

    def __getattr__(self, name):
        if hasattr(self._a, name):
            return getattr(self._a, name)
        if hasattr(self._b, name):
            return getattr(self._b, name)
        raise AttributeError(name)
class A:
    x = 10
    def f(self): return "A"

class B:
    y = 20
    def f(self): return "B"

f = Fusion(A(), B())
print(f.x, f.y)
print(f.f())
