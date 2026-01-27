def combine_objects(obj1, obj2):
    """Dynamically combine attributes and methods of two objects."""
    class Combined:
        pass
    
    combined = Combined()
    
    # Copy all attributes/methods from obj1
    for name in dir(obj1):
        if not name.startswith('__'):
            setattr(combined, name, getattr(obj1, name))
    
    # Copy all attributes/methods from obj2 (overwrites if same name)
    for name in dir(obj2):
        if not name.startswith('__'):
            setattr(combined, name, getattr(obj2, name))
    
    return combined

# Example usage
class A:
    def method_a(self):
        print("Method A")

class B:
    def method_b(self):
        print("Method B")

combined = combine_objects(A(), B())
combined.method_a()  # Output: Method A
combined.method_b()  # Output: Method B
