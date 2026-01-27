def fuse_vars(a, b):
    class Fusion: pass
    f = Fusion()
    
    for k, v in vars(a).items():
        setattr(f, k, v)
    for k, v in vars(b).items():
        setattr(f, k, v)

    return f

class A:
    def __init__(self):
        self.x = 10
        self.y = "cat"

class B:
    def __init__(self):
        self.y = "dog"
        self.z = 42

a = A()
b = B()

c = fuse_vars(a, b)

print(c.x)  # 10
print(c.y)  # dog (B overrides A)
print(c.z)  # 42
