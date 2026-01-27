class InfiniteMeta(type):
    def __new__(mcls, name, bases, namespace):
        print(f"Creating class: {name}")
        
        # Create a new class dynamically
        cls = super().__new__(mcls, name, bases, namespace)
        
        # Generate a new class using the same metaclass (recursive)
        if name != "Infinity":
            class Infinity(metaclass=InfiniteMeta):
                pass
        
        return cls


class Root(metaclass=InfiniteMeta):
    pass


# Let's explore what happened
print("\n--- Analysis ---")
print(f"Root class: {Root}")
print(f"Root metaclass: {type(Root)}")

# Create more classes to see the pattern
print("\n--- Creating more classes ---")

class Branch(metaclass=InfiniteMeta):
    value = 42

class Leaf(metaclass=InfiniteMeta):
    def speak(self):
        return "Hello from Leaf"


# Demonstrate the recursive nature
print("\n--- Demonstrating recursion control ---")

class ControlledMeta(type):
    _depth = 0
    _max_depth = 3
    
    def __new__(mcls, name, bases, namespace):
        ControlledMeta._depth += 1
        print(f"{'  ' * ControlledMeta._depth}Creating: {name} (depth: {ControlledMeta._depth})")
        
        cls = super().__new__(mcls, name, bases, namespace)
        
        # Controlled recursion with depth limit
        if name != "RecursiveClass" and ControlledMeta._depth < ControlledMeta._max_depth:
            class RecursiveClass(metaclass=ControlledMeta):
                level = ControlledMeta._depth
        
        ControlledMeta._depth -= 1
        return cls


print("\nCreating controlled recursive class:")
ControlledMeta._depth = 0

class StartPoint(metaclass=ControlledMeta):
    pass


# Practical example: Auto-registering classes
print("\n--- Practical Example: Class Registry ---")

class RegistryMeta(type):
    registry = {}
    
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        
        # Auto-register all classes (except base)
        if name != "Registered":
            mcls.registry[name] = cls
            print(f"Registered: {name}")
        
        return cls


class Registered(metaclass=RegistryMeta):
    pass


class Plugin1(Registered):
    feature = "authentication"


class Plugin2(Registered):
    feature = "logging"


class Plugin3(Registered):
    feature = "caching"


print(f"\nRegistry contents: {list(RegistryMeta.registry.keys())}")
print(f"Total registered: {len(RegistryMeta.registry)}")

# Access registered classes
for name, cls in RegistryMeta.registry.items():
    print(f"  {name}: {cls}")