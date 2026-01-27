class MetaFusion(type):
    def __new__(meta, name, bases, attrs):
        fused_attrs = {}

        # Collect attributes from all base classes
        for base in bases:
            for k, v in base.__dict__.items():
                if not k.startswith("__"):
                    fused_attrs[k] = v

        # Override with new attributes
        fused_attrs.update(attrs)

        return super().__new__(meta, name, bases, fused_attrs)


def fuse_classes(name, *classes):
    return MetaFusion(name, classes, {})

class Cat:
    def speak(self):
        return "meow"

    def identity(self):
        return "animal-cat"


class Dog:
    def speak(self):
        return "woof"

    def identity(self):
        return "animal-dog"


class AI:
    def think(self):
        return "I compute meaning"

MetaBeing = fuse_classes("MetaBeing", Cat, Dog, AI)

m = MetaBeing()

print(m.speak())      # conflict resolved by last class (Dog)
print(m.identity())   # Dog identity
print(m.think())      # AI ability

def merge_methods(*methods):
    def fused(self, *args, **kwargs):
        return [m(self, *args, **kwargs) for m in methods]
    return fused


def deep_fuse(name, *classes):
    methods = {}

    for cls in classes:
        for k, v in cls.__dict__.items():
            if callable(v) and not k.startswith("__"):
                methods.setdefault(k, []).append(v)

    fused_attrs = {k: merge_methods(*v) for k, v in methods.items()}
    return type(name, (), fused_attrs)

MetaMind = deep_fuse("MetaMind", Cat, Dog, AI)

x = MetaMind()

print(x.speak())     # ['meow', 'woof']
print(x.identity())  # ['animal-cat', 'animal-dog']
print(x.think())     # ['I compute meaning']

class SelfEvolvingMeta(type):
    def evolve(cls, new_class):
        return fuse_classes("Evolved"+cls.__name__, cls, new_class)


class MetaEntity(metaclass=SelfEvolvingMeta):
    def core(self):
        return "base-existence"


class Quantum:
    def state(self):
        return "superposition"


Evolved = MetaEntity.evolve(Quantum)

e = Evolved()
print(e.core())
print(e.state())
