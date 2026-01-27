import types

def fuse_objects(a, b):
    class Fusion:
        pass

    f = Fusion()

    # merge attributes from a
    for name in dir(a):
        if not name.startswith("__"):
            value = getattr(a, name)
            setattr(f, name, value)

    # merge attributes from b (override if needed)
    for name in dir(b):
        if not name.startswith("__"):
            value = getattr(b, name)
            setattr(f, name, value)

    return f

class A:
    x = 10
    def f(self): return "A"

class B:
    y = 20
    def f(self): return "B"

fa = fuse_objects(A(), B())
print(fa.x, fa.y)
print(fa.f())  # B overrides A
