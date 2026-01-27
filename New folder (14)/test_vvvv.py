# Lambda calculus core
I = lambda x: x                        # Identity
K = lambda x: lambda y: x              # Constant
S = lambda f: lambda g: lambda x: f(x)(g(x))  # Substitution

# Function composition (fusion)
compose = lambda f, g: lambda x: f(g(x))
class LambdaMeta(type):
    def __new__(mcls, name, bases, namespace):
        lambdas = {k: v for k, v in namespace.items() if callable(v)}

        def fused(*args, **kwargs):
            result = args[0]
            for f in lambdas.values():
                result = f(result)
            return result

        namespace["__lambda_fusion__"] = fused
        return super().__new__(mcls, name, bases, namespace)
class LambdaSpace(metaclass=LambdaMeta):
    a = lambda x: x + 1
    b = lambda x: x * 2
    c = lambda x: x ** 2
print(LambdaSpace.__lambda_fusion__(3))
class LambdaTerm:
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def fuse(self, other):
        return LambdaTerm(lambda x: self.f(other.f(x)))

    def __repr__(self):
        return f"λ({self.f})"
f = LambdaTerm(lambda x: x + 1)
g = LambdaTerm(lambda x: x * 3)

h = f.fuse(g)   # λx.(f(g(x)))

print(h(5))  # (5*3)+1 = 16
