import random
import string
from types import FunctionType

# Utility to generate random method names
def random_name(length=5):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# Utility to generate a random method
def random_method():
    def method(self):
        return f"My random value: {random.randint(0, 100)}"
    return method

# Function to create a random class
def create_random_class(class_name=None, num_methods=None):
    if class_name is None:
        class_name = random_name().capitalize()
    if num_methods is None:
        num_methods = random.randint(1, 5)
    
    class_dict = {}
    
    # Generate random methods
    for _ in range(num_methods):
        method_name = random_name()
        class_dict[method_name] = random_method()
    
    # Optionally, add a random attribute
    class_dict['random_attribute'] = random.randint(0, 100)
    
    # Create the class dynamically
    NewClass = type(class_name, (object,), class_dict)
    return NewClass

# Example usage
for _ in range(3):
    AIClass = create_random_class()
    obj = AIClass()
    print(f"Class: {AIClass.__name__}, Attribute: {obj.random_attribute}")
    for method_name in dir(AIClass):
      if not method_name.startswith('__'):
        attr = getattr(obj, method_name)
        if callable(attr):  # Only call it if it's a function/method
            print(f"  Method {method_name} output:", attr())
        else:
            print(f"  Attribute {method_name} value:", attr)
