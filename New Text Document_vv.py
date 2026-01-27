def fuse_types(a, b):
    return type(
        "FusionType",
        (type(a), type(b)),
        {}
    )
class A:
    x = 10
    def f(self): return "A"

class B:
    y = 20
    def f(self): return "B"
FusionClass = fuse_types(A(), B())
obj = FusionClass()
print(obj.f())  # MRO decides
