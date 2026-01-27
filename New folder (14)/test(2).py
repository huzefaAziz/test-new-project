class MetaLambda(type):
    def __call__(cls, x):
        funcs = [v for v in cls.__dict__.values() if callable(v)]
        for f in funcs:
            x = f(x)
        return x


class Ω(metaclass=MetaLambda):
    f = lambda x: x + 10
    g = lambda x: x * 2
    h = lambda x: x - 3
    
print(Ω(5))
