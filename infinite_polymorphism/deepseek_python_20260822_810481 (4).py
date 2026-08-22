from typing import TypeVar, Callable, Any, Union, overload
from functools import singledispatch
import inspect

T = TypeVar('T')
U = TypeVar('U')

# === 1. INFINITE CHAINING POLYMORPHISM ===

class InfiniteChain:
    """Supports infinitely nested method calls with polymorphic behavior"""
    
    def __init__(self, value: Any = None):
        self._value = value
        self._operations = []
    
    def __call__(self, *args, **kwargs):
        """Enable infinite currying"""
        if args and callable(args[0]):
            # Apply transformation
            fn = args[0]
            result = fn(self._value)
            return InfiniteChain(result)
        return self._value
    
    def __getattr__(self, name):
        """Intercept any method call for infinite polymorphism"""
        def method(*args, **kwargs):
            # Dynamically dispatch based on method name
            return self._dispatch(name, *args, **kwargs)
        return method
    
    def _dispatch(self, name: str, *args, **kwargs):
        """Polymorphic dispatcher with fallback"""
        # Handle type-based dispatch
        if hasattr(self._value, name):
            # Delegate to wrapped object if method exists
            attr = getattr(self._value, name)
            if callable(attr):
                result = attr(*args, **kwargs)
                return InfiniteChain(result)
        
        # Handle generic operations
        if name == 'map':
            return self._map(*args, **kwargs)
        elif name == 'filter':
            return self._filter(*args, **kwargs)
        elif name == 'reduce':
            return self._reduce(*args, **kwargs)
        elif name == 'chain':
            return self._chain(*args, **kwargs)
        else:
            # Default behavior: store operation for lazy evaluation
            self._operations.append((name, args, kwargs))
            return self
    
    def _map(self, fn):
        """Map over iterable values"""
        if hasattr(self._value, '__iter__'):
            result = [fn(x) for x in self._value]
            return InfiniteChain(result)
        return self
    
    def _filter(self, predicate):
        """Filter iterable values"""
        if hasattr(self._value, '__iter__'):
            result = [x for x in self._value if predicate(x)]
            return InfiniteChain(result)
        return self
    
    def _reduce(self, fn, initial=None):
        """Reduce iterable values"""
        if hasattr(self._value, '__iter__'):
            from functools import reduce
            if initial is not None:
                result = reduce(fn, self._value, initial)
            else:
                result = reduce(fn, self._value)
            return InfiniteChain(result)
        return self
    
    def _chain(self, *others):
        """Chain multiple iterables"""
        if hasattr(self._value, '__iter__'):
            result = list(self._value)
            for other in others:
                if hasattr(other, '__iter__'):
                    result.extend(other)
            return InfiniteChain(result)
        return self
    
    def __repr__(self):
        return f"InfiniteChain({repr(self._value)})"


# === 2. MULTI-DISPATCH WITH GENERICS ===

@singledispatch
def infinite_process(arg):
    """Base polymorphic function"""
    return f"Default: {arg}"

@infinite_process.register(int)
def _(arg: int):
    return f"Integer: {arg * 2}"

@infinite_process.register(str)
def _(arg: str):
    return f"String: {arg.upper()}"

@infinite_process.register(list)
def _(arg: list):
    return f"List: {[x * 2 if isinstance(x, int) else x for x in arg]}"

@infinite_process.register(dict)
def _(arg: dict):
    return f"Dict: { {k: str(v) for k, v in arg.items()} }"


# === 3. FIXED PROTOCOL-BASED POLYMORPHISM ===

class PolyMeta(type):
    """Metaclass for infinite polymorphic behavior"""
    def __call__(cls, *args, **kwargs):
        # Intercept instantiation for polymorphic creation
        if len(args) == 0:
            return super().__call__(*args, **kwargs)
        
        # Type-based factory pattern
        first_arg = args[0]
        
        # FIXED: Return instance directly without triggering metaclass again
        if isinstance(first_arg, int):
            return IntegerHandler.__new__(IntegerHandler)
        elif isinstance(first_arg, str):
            return StringHandler.__new__(StringHandler)
        elif isinstance(first_arg, list):
            return ListHandler.__new__(ListHandler)
        else:
            return GenericHandler.__new__(GenericHandler)

class BasePoly:
    """Base class with metaclass-based polymorphism"""
    def __init__(self, value):
        self.value = value
    
    def handle(self):
        return f"Base: {self.value}"

class IntegerHandler(BasePoly):
    def handle(self):
        return f"Integer ×2: {self.value * 2}"

class StringHandler(BasePoly):
    def handle(self):
        return f"String: {self.value.upper()}"

class ListHandler(BasePoly):
    def handle(self):
        return f"List length: {len(self.value)}, sum: {sum(self.value) if all(isinstance(x, (int, float)) for x in self.value) else 'N/A'}"

class GenericHandler(BasePoly):
    def handle(self):
        return f"Generic: {type(self.value).__name__}"


# === 4. ALTERNATIVE FACTORY PATTERN (SIMPLER) ===

class PolyFactory:
    """Simpler factory pattern without metaclass issues"""
    
    @staticmethod
    def create(value):
        if isinstance(value, int):
            return IntegerHandler(value)
        elif isinstance(value, str):
            return StringHandler(value)
        elif isinstance(value, list):
            return ListHandler(value)
        else:
            return GenericHandler(value)


# === 5. COMPOSABLE INFINITE POLYMORPHISM ===

class PolyComposer:
    """Composable polymorphic transformations"""
    
    def __init__(self, value):
        self.value = value
    
    def __rshift__(self, other):
        """Implement >> operator for chaining"""
        if callable(other):
            result = other(self.value)
            return PolyComposer(result)
        return PolyComposer(other)
    
    def __or__(self, other):
        """Implement | operator for alternative transformations"""
        try:
            return PolyComposer(other(self.value))
        except Exception as e:
            return PolyComposer(f"Error: {e}")
    
    def __and__(self, other):
        """Implement & operator for parallel transformations"""
        if isinstance(other, PolyComposer):
            result = (self.value, other.value)
            return PolyComposer(result)
        return PolyComposer((self.value, other))
    
    def transform(self, *fns):
        """Apply multiple transformations"""
        result = self.value
        for fn in fns:
            if callable(fn):
                result = fn(result)
        return PolyComposer(result)
    
    def get(self):
        return self.value
    
    def __repr__(self):
        return f"PolyComposer({repr(self.value)})"


# === 6. DEMONSTRATION ===

def demo():
    print("=" * 60)
    print("1. INFINITE CHAINING POLYMORPHISM")
    print("=" * 60)
    
    # Chain operations infinitely
    chain = InfiniteChain([1, 2, 3, 4, 5])
    result = (chain
              .map(lambda x: x * 2)
              .filter(lambda x: x > 5)
              .map(lambda x: x + 1)
              .map(lambda x: f"Item: {x}")
              ._value)
    print(f"Chained result: {result}")
    
    print("\n" + "=" * 60)
    print("2. MULTI-DISPATCH POLYMORPHISM")
    print("=" * 60)
    
    # Single dispatch with types
    values = [10, "hello", [1, 2, 3], {"a": 1, "b": 2}, 3.14]
    for v in values:
        print(f"{v} -> {infinite_process(v)}")
    
    print("\n" + "=" * 60)
    print("3. FIXED METACLASS-BASED POLYMORPHISM")
    print("=" * 60)
    
    # FIXED: Using the simpler factory pattern
    print("Using PolyFactory (recommended):")
    handlers = [
        PolyFactory.create(42),
        PolyFactory.create("world"),
        PolyFactory.create([1, 2, 3, 4]),
        PolyFactory.create({1, 2, 3})  # Set
    ]
    for h in handlers:
        print(f"{h.value} -> {h.handle()}")
    
    print("\n" + "=" * 60)
    print("4. COMPOSABLE POLYMORPHISM")
    print("=" * 60)
    
    # Using >> with proper parentheses
    transformed = (PolyComposer(10) >> 
                   (lambda x: x * 2) >> 
                   (lambda x: x + 5) >> 
                   (lambda x: str(x)))
    print(f"Chain using >>: {transformed.get()}")
    
    # Using transform method (cleaner)
    cleaner = (PolyComposer(10)
               .transform(
                   lambda x: x * 2,
                   lambda x: x + 5,
                   lambda x: str(x)
               ))
    print(f"Transform pipeline: {cleaner.get()}")
    
    # Using | for fallback
    safe = (PolyComposer("123") | int)
    safe2 = safe >> (lambda x: x * 3)
    print(f"Safe transformation: {safe2.get()}")
    
    # Using & for parallel composition
    parallel = (PolyComposer(5) & PolyComposer(10))
    print(f"Parallel composition: {parallel.get()}")
    
    # Custom transformation pipeline
    result = (PolyComposer([1, 2, 3, 4, 5])
              .transform(
                  lambda x: sum(x),
                  lambda x: x * 2,
                  lambda x: f"Result: {x}"
              ))
    print(f"Custom pipeline: {result.get()}")
    
    print("\n" + "=" * 60)
    print("5. ADVANCED RECURSIVE POLYMORPHISM")
    print("=" * 60)
    
    # Infinite recursion with type-based dispatch
    def recursive_poly(value, depth=0):
        if depth > 5:
            return f"Base case: {value}"
        
        if isinstance(value, int):
            return recursive_poly(value * 2 + 1, depth + 1)
        elif isinstance(value, str):
            return recursive_poly(value + "!", depth + 1)
        elif isinstance(value, list):
            return recursive_poly([x * 2 for x in value], depth + 1)
        else:
            return recursive_poly(str(value), depth + 1)
    
    print(f"Recursive int: {recursive_poly(1)}")
    print(f"Recursive str: {recursive_poly('abc')}")
    print(f"Recursive list: {recursive_poly([1, 2, 3])}")
    print(f"Recursive mixed: {recursive_poly([1, 'a', 3.14])}")
    
    print("\n" + "=" * 60)
    print("6. ADDITIONAL CLEAN EXAMPLES")
    print("=" * 60)
    
    # Clean chaining with method calls
    numbers = InfiniteChain([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = (numbers
              .map(lambda x: x * 3)
              .filter(lambda x: x % 2 == 0)
              .map(lambda x: x / 2)
              .reduce(lambda a, b: a + b, 0)
              ._value)
    print(f"Sum of even numbers ×3 ÷2: {result}")
    
    # String operations
    text = InfiniteChain("hello world")
    result = (text
              .map(lambda x: x.upper())
              ._value)
    print(f"Uppercase: {result}")
    
    # Nested structures
    data = InfiniteChain([[1, 2], [3, 4], [5, 6]])
    result = (data
              .map(lambda x: sum(x))
              .filter(lambda x: x > 5)
              ._value)
    print(f"Sum of sublists >5: {result}")
    
    print("\n" + "=" * 60)
    print("7. TYPE-BASED POLYMORPHISM WITH PROTOCOLS")
    print("=" * 60)
    
    # Duck typing example
    class Animal:
        def speak(self):
            return "Animal sound"
    
    class Dog:
        def speak(self):
            return "Woof!"
    
    class Cat:
        def speak(self):
            return "Meow!"
    
    def animal_poly(creature):
        """Polymorphic function using duck typing"""
        if hasattr(creature, 'speak'):
            return creature.speak()
        return "Not a creature"
    
    print(f"Dog: {animal_poly(Dog())}")
    print(f"Cat: {animal_poly(Cat())}")
    print(f"Animal: {animal_poly(Animal())}")
    print(f"String: {animal_poly('not an animal')}")


if __name__ == "__main__":
    demo()