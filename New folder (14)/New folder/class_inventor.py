"""
AI Class Inventor - Automatically creates new classes based on patterns
"""

from typing import Any, Dict, List, Callable
import inspect
from collections import defaultdict


class ClassInventor:
    """
    An AI-like system that analyzes data patterns and automatically 
    invents new classes to represent them.
    """
    
    def __init__(self):
        self.invented_classes = {}
        self.class_registry = {}
        
    def analyze_and_invent(self, data_samples: List[Dict[str, Any]], 
                          base_name: str = "AutoClass") -> type:
        """
        Analyze data samples and invent a new class to represent them.
        
        Args:
            data_samples: List of dictionaries representing data instances
            base_name: Base name for the invented class
            
        Returns:
            Newly created class type
        """
        # Analyze the structure
        field_types = self._infer_types(data_samples)
        common_fields = self._find_common_fields(data_samples)
        
        # Generate class name
        class_name = f"{base_name}_{len(self.invented_classes)}"
        
        # Create the class dynamically
        new_class = self._create_class(class_name, field_types, common_fields)
        
        # Store it
        self.invented_classes[class_name] = new_class
        
        return new_class
    
    def _infer_types(self, samples: List[Dict]) -> Dict[str, type]:
        """Infer types for each field from samples."""
        type_map = defaultdict(set)
        
        for sample in samples:
            for key, value in sample.items():
                type_map[key].add(type(value))
        
        # Choose the most common or general type
        field_types = {}
        for field, types in type_map.items():
            if len(types) == 1:
                field_types[field] = list(types)[0]
            else:
                # If mixed types, use Any
                field_types[field] = Any
                
        return field_types
    
    def _find_common_fields(self, samples: List[Dict]) -> set:
        """Find fields common to all samples."""
        if not samples:
            return set()
        
        common = set(samples[0].keys())
        for sample in samples[1:]:
            common &= set(sample.keys())
            
        return common
    
    def _create_class(self, name: str, field_types: Dict[str, type], 
                     common_fields: set) -> type:
        """Dynamically create a new class."""
        
        def __init__(self, **kwargs):
            """Auto-generated initializer."""
            for field in common_fields:
                if field not in kwargs:
                    raise ValueError(f"Required field '{field}' missing")
                setattr(self, field, kwargs[field])
            
            # Optional fields
            for field in field_types:
                if field not in common_fields and field in kwargs:
                    setattr(self, field, kwargs[field])
        
        def __repr__(self):
            """Auto-generated string representation."""
            attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
            return f"{name}({attrs})"
        
        def to_dict(self):
            """Convert instance to dictionary."""
            return self.__dict__.copy()
        
        def validate(self):
            """Validate instance fields match expected types."""
            for field, expected_type in field_types.items():
                if hasattr(self, field):
                    value = getattr(self, field)
                    if expected_type != Any and not isinstance(value, expected_type):
                        return False, f"Field '{field}' should be {expected_type}"
            return True, "Valid"
        
        # Create class attributes
        class_attrs = {
            '__init__': __init__,
            '__repr__': __repr__,
            'to_dict': to_dict,
            'validate': validate,
            '_field_types': field_types,
            '_common_fields': common_fields,
        }
        
        # Create the class
        new_class = type(name, (object,), class_attrs)
        return new_class


class PatternBasedInventor(ClassInventor):
    """
    Advanced class inventor that recognizes patterns and creates 
    specialized classes with custom methods.
    """
    
    def __init__(self):
        super().__init__()
        self.patterns = {
            'has_coordinates': self._add_location_methods,
            'has_timestamp': self._add_temporal_methods,
            'has_price': self._add_financial_methods,
        }
    
    def analyze_and_invent(self, data_samples: List[Dict[str, Any]], 
                          base_name: str = "SmartClass") -> type:
        """Create a class with pattern-based enhancements."""
        # Get basic class
        new_class = super().analyze_and_invent(data_samples, base_name)
        
        # Detect and apply patterns
        field_types = new_class._field_types
        
        for pattern_name, enhancer in self.patterns.items():
            if self._detect_pattern(pattern_name, field_types):
                new_class = enhancer(new_class, field_types)
        
        return new_class
    
    def _detect_pattern(self, pattern: str, fields: Dict[str, type]) -> bool:
        """Detect if a pattern exists in the fields."""
        if pattern == 'has_coordinates':
            return 'x' in fields and 'y' in fields
        elif pattern == 'has_timestamp':
            return 'timestamp' in fields or 'time' in fields or 'date' in fields
        elif pattern == 'has_price':
            return 'price' in fields or 'cost' in fields or 'amount' in fields
        return False
    
    def _add_location_methods(self, cls: type, fields: Dict) -> type:
        """Add location-based methods to the class."""
        def distance_from_origin(self):
            """Calculate distance from origin."""
            return (self.x ** 2 + self.y ** 2) ** 0.5
        
        def move(self, dx, dy):
            """Move by delta x and y."""
            self.x += dx
            self.y += dy
            return self
        
        cls.distance_from_origin = distance_from_origin
        cls.move = move
        return cls
    
    def _add_temporal_methods(self, cls: type, fields: Dict) -> type:
        """Add time-based methods."""
        def is_recent(self, days=7):
            """Check if timestamp is recent (placeholder)."""
            # This would need proper datetime handling
            return True
        
        cls.is_recent = is_recent
        return cls
    
    def _add_financial_methods(self, cls: type, fields: Dict) -> type:
        """Add financial methods."""
        def apply_discount(self, percentage):
            """Apply discount to price."""
            if hasattr(self, 'price'):
                self.price *= (1 - percentage / 100)
            elif hasattr(self, 'cost'):
                self.cost *= (1 - percentage / 100)
            elif hasattr(self, 'amount'):
                self.amount *= (1 - percentage / 100)
            return self
        
        cls.apply_discount = apply_discount
        return cls


class InheritanceInventor:
    """
    Creates class hierarchies automatically based on shared attributes.
    """
    
    def __init__(self):
        self.class_tree = {}
    
    def create_hierarchy(self, grouped_samples: Dict[str, List[Dict]]) -> Dict[str, type]:
        """
        Create a class hierarchy from grouped samples.
        
        Args:
            grouped_samples: Dict mapping group names to sample lists
            
        Returns:
            Dictionary of created classes
        """
        # Find common attributes across all groups
        all_samples = []
        for samples in grouped_samples.values():
            all_samples.extend(samples)
        
        common_attrs = self._find_common_attributes(all_samples)
        
        # Create base class
        BaseClass = self._create_base_class(common_attrs)
        
        # Create specialized classes
        classes = {'Base': BaseClass}
        
        for group_name, samples in grouped_samples.items():
            SpecializedClass = self._create_specialized_class(
                group_name, BaseClass, samples
            )
            classes[group_name] = SpecializedClass
        
        return classes
    
    def _find_common_attributes(self, samples: List[Dict]) -> set:
        """Find attributes common to all samples."""
        if not samples:
            return set()
        
        common = set(samples[0].keys())
        for sample in samples[1:]:
            common &= set(sample.keys())
        return common
    
    def _create_base_class(self, common_attrs: set) -> type:
        """Create a base class with common attributes."""
        def __init__(self, **kwargs):
            for attr in common_attrs:
                setattr(self, attr, kwargs.get(attr))
        
        def __repr__(self):
            attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
            return f"{self.__class__.__name__}({attrs})"
        
        return type('AutoBase', (object,), {
            '__init__': __init__,
            '__repr__': __repr__,
            '_common_attrs': common_attrs
        })
    
    def _create_specialized_class(self, name: str, base: type, 
                                  samples: List[Dict]) -> type:
        """Create a specialized subclass."""
        # Find unique attributes for this group
        all_attrs = set()
        for sample in samples:
            all_attrs.update(sample.keys())
        
        unique_attrs = all_attrs - base._common_attrs
        
        def __init__(self, **kwargs):
            base.__init__(self, **kwargs)
            for attr in unique_attrs:
                if attr in kwargs:
                    setattr(self, attr, kwargs[attr])
        
        return type(f'Auto{name}', (base,), {
            '__init__': __init__,
            '_unique_attrs': unique_attrs
        })


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("AI Class Inventor Demo")
    print("=" * 60)
    
    # Example 1: Basic class invention
    print("\n1. Basic Class Invention:")
    print("-" * 60)
    
    inventor = ClassInventor()
    
    user_samples = [
        {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'},
        {'name': 'Bob', 'age': 25, 'email': 'bob@example.com'},
        {'name': 'Charlie', 'age': 35, 'email': 'charlie@example.com'},
    ]
    
    UserClass = inventor.analyze_and_invent(user_samples, "User")
    print(f"Created class: {UserClass.__name__}")
    print(f"Common fields: {UserClass._common_fields}")
    print(f"Field types: {UserClass._field_types}")
    
    # Create instances
    user1 = UserClass(name='David', age=28, email='david@example.com')
    print(f"\nInstance: {user1}")
    print(f"Validation: {user1.validate()}")
    
    # Example 2: Pattern-based invention
    print("\n\n2. Pattern-Based Class Invention:")
    print("-" * 60)
    
    smart_inventor = PatternBasedInventor()
    
    location_samples = [
        {'x': 10, 'y': 20, 'name': 'Point A'},
        {'x': 30, 'y': 40, 'name': 'Point B'},
        {'x': 50, 'y': 60, 'name': 'Point C'},
    ]
    
    LocationClass = smart_inventor.analyze_and_invent(location_samples, "Location")
    print(f"Created class: {LocationClass.__name__}")
    
    point = LocationClass(x=3, y=4, name='Origin Point')
    print(f"\nInstance: {point}")
    print(f"Distance from origin: {point.distance_from_origin()}")
    point.move(1, 1)
    print(f"After moving: {point}")
    
    # Example 3: Product with price (gets financial methods)
    product_samples = [
        {'name': 'Widget', 'price': 19.99, 'category': 'Tools'},
        {'name': 'Gadget', 'price': 29.99, 'category': 'Electronics'},
    ]
    
    ProductClass = smart_inventor.analyze_and_invent(product_samples, "Product")
    print(f"\nCreated class: {ProductClass.__name__}")
    
    product = ProductClass(name='SuperWidget', price=100.0, category='Premium')
    print(f"Instance: {product}")
    product.apply_discount(20)
    print(f"After 20% discount: {product}")
    
    # Example 4: Class hierarchy
    print("\n\n3. Automatic Class Hierarchy:")
    print("-" * 60)
    
    hierarchy_inventor = InheritanceInventor()
    
    grouped_data = {
        'Employee': [
            {'name': 'Alice', 'id': 1, 'salary': 50000, 'department': 'Engineering'},
            {'name': 'Bob', 'id': 2, 'salary': 60000, 'department': 'Sales'},
        ],
        'Customer': [
            {'name': 'Charlie', 'id': 3, 'loyalty_points': 100},
            {'name': 'David', 'id': 4, 'loyalty_points': 200},
        ]
    }
    
    classes = hierarchy_inventor.create_hierarchy(grouped_data)
    
    print("Created class hierarchy:")
    for class_name, cls in classes.items():
        print(f"  - {cls.__name__}")
    
    emp = classes['Employee'](name='Eve', id=5, salary=55000, department='HR')
    cust = classes['Customer'](name='Frank', id=6, loyalty_points=150)
    
    print(f"\nEmployee instance: {emp}")
    print(f"Customer instance: {cust}")
    print(f"\nBoth inherit from: {classes['Employee'].__bases__}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
