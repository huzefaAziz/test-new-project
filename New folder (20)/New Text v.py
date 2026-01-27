import types
import inspect

def bind_methods(target, source):
    for name in dir(source):
        if name.startswith("__"):
            continue
        
        attr = getattr(source, name)

        # Case 1: bound method (already attached to source instance)
        if isinstance(attr, types.MethodType):
            func = attr.__func__
            bound = types.MethodType(func, target)
            setattr(target, name, bound)

        # Case 2: plain function (defined in class)
        elif isinstance(attr, types.FunctionType):
            bound = types.MethodType(attr, target)
            setattr(target, name, bound)

        # Case 3: properties
        elif isinstance(getattr(type(source), name, None), property):
            setattr(type(target), name, getattr(type(source), name))

        # Case 4: normal attributes (data fusion)
        elif not inspect.isbuiltin(attr):
            setattr(target, name, attr)

    return target
class A:
    def foo(self):
        return "foo from A"

class B:
    def bar(self):
        return "bar from B"

a = A()
b = B()

bind_methods(a, b)

print(a.foo())  # foo from A
print(a.bar())  # bar from B
def fuse_objects(*objs):
    class Fusion: pass
    f = Fusion()
    for obj in objs:
        bind_methods(f, obj)
    return f
x = fuse_objects(A(), B())
print('test')
print(x.foo())
print(x.bar())
