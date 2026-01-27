"""
Advanced Class Inventor with ML-like Pattern Recognition
"""

import re
from typing import Any, Dict, List, Set
from collections import Counter
from class_inventor import PatternBasedInventor


class MLStyleClassInventor:
    """
    Uses ML-like pattern recognition to invent highly specialized classes.
    """
    
    def __init__(self):
        self.invented_classes = {}
        self.learned_patterns = []
        
    def train(self, training_data: List[Dict[str, List[Dict]]]):
        """
        'Train' the inventor by learning patterns from examples.
        
        Args:
            training_data: List of dicts with 'class_name' and 'samples'
        """
        for data in training_data:
            class_name = data['class_name']
            samples = data['samples']
            
            # Learn patterns
            patterns = self._extract_patterns(samples)
            self.learned_patterns.append({
                'class_name': class_name,
                'patterns': patterns
            })
            
        print(f"Learned {len(self.learned_patterns)} class patterns")
    
    def _extract_patterns(self, samples: List[Dict]) -> Dict:
        """Extract patterns from samples."""
        patterns = {
            'field_names': self._get_field_patterns(samples),
            'naming_conventions': self._detect_naming_conventions(samples),
            'value_ranges': self._detect_value_ranges(samples),
        }
        return patterns
    
    def _get_field_patterns(self, samples: List[Dict]) -> Set[str]:
        """Get common field names."""
        all_fields = set()
        for sample in samples:
            all_fields.update(sample.keys())
        return all_fields
    
    def _detect_naming_conventions(self, samples: List[Dict]) -> Dict:
        """Detect naming conventions (snake_case, camelCase, etc.)."""
        conventions = Counter()
        
        for sample in samples:
            for key in sample.keys():
                if '_' in key:
                    conventions['snake_case'] += 1
                elif any(c.isupper() for c in key[1:]):
                    conventions['camelCase'] += 1
                    
        return dict(conventions)
    
    def _detect_value_ranges(self, samples: List[Dict]) -> Dict:
        """Detect value ranges for numeric fields."""
        ranges = {}
        
        for sample in samples:
            for key, value in sample.items():
                if isinstance(value, (int, float)):
                    if key not in ranges:
                        ranges[key] = {'min': value, 'max': value}
                    else:
                        ranges[key]['min'] = min(ranges[key]['min'], value)
                        ranges[key]['max'] = max(ranges[key]['max'], value)
                        
        return ranges
    
    def invent_class(self, samples: List[Dict], hint: str = None) -> type:
        """
        Invent a class based on samples and learned patterns.
        
        Args:
            samples: Sample data
            hint: Optional hint about what kind of class this should be
        """
        # Match against learned patterns
        best_match = self._find_best_pattern_match(samples)
        
        # Determine class name
        if hint:
            class_name = f"Auto{hint}"
        elif best_match:
            class_name = f"Auto{best_match['class_name']}"
        else:
            class_name = f"AutoDiscovered_{len(self.invented_classes)}"
        
        # Create class with learned behaviors
        new_class = self._build_class(class_name, samples, best_match)
        
        self.invented_classes[class_name] = new_class
        return new_class
    
    def _find_best_pattern_match(self, samples: List[Dict]) -> Dict:
        """Find the best matching learned pattern."""
        sample_fields = set()
        for sample in samples:
            sample_fields.update(sample.keys())
        
        best_score = 0
        best_pattern = None
        
        for learned in self.learned_patterns:
            learned_fields = learned['patterns']['field_names']
            
            # Calculate similarity (Jaccard index)
            intersection = len(sample_fields & learned_fields)
            union = len(sample_fields | learned_fields)
            
            if union > 0:
                score = intersection / union
                
                if score > best_score:
                    best_score = score
                    best_pattern = learned
        
        return best_pattern if best_score > 0.5 else None
    
    def _build_class(self, name: str, samples: List[Dict], 
                    pattern_match: Dict = None) -> type:
        """Build the class with all features."""
        
        # Analyze samples
        field_types = self._infer_types(samples)
        value_ranges = self._detect_value_ranges(samples)
        
        def __init__(self, **kwargs):
            """Smart initializer with validation."""
            for field, value in kwargs.items():
                # Apply range validation if available
                if field in value_ranges and isinstance(value, (int, float)):
                    min_val = value_ranges[field]['min']
                    max_val = value_ranges[field]['max']
                    
                    if not (min_val * 0.8 <= value <= max_val * 1.2):
                        print(f"Warning: {field}={value} is outside typical range [{min_val}, {max_val}]")
                
                setattr(self, field, value)
        
        def __repr__(self):
            attrs = ', '.join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
            return f"{name}({attrs})"
        
        def describe(self):
            """Describe this object's characteristics."""
            print(f"\n{name} Description:")
            print("-" * 40)
            for field, value in sorted(self.__dict__.items()):
                field_type = type(value).__name__
                print(f"  {field}: {value} ({field_type})")
                
                if field in value_ranges:
                    r = value_ranges[field]
                    print(f"    Typical range: [{r['min']}, {r['max']}]")
        
        def to_json_compatible(self):
            """Convert to JSON-compatible dict."""
            result = {}
            for k, v in self.__dict__.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    result[k] = v
                else:
                    result[k] = str(v)
            return result
        
        # Build class attributes
        class_attrs = {
            '__init__': __init__,
            '__repr__': __repr__,
            'describe': describe,
            'to_json_compatible': to_json_compatible,
            '_field_types': field_types,
            '_value_ranges': value_ranges,
            '_pattern_match': pattern_match,
        }
        
        # Add learned methods if pattern matched
        if pattern_match:
            class_attrs['_learned_from'] = pattern_match['class_name']
        
        return type(name, (object,), class_attrs)
    
    def _infer_types(self, samples: List[Dict]) -> Dict[str, type]:
        """Infer field types."""
        type_map = {}
        
        for sample in samples:
            for key, value in sample.items():
                if key not in type_map:
                    type_map[key] = type(value)
                    
        return type_map


def demo_ml_style_inventor():
    """Demonstrate ML-style class invention."""
    
    print("=" * 70)
    print("ML-Style Class Inventor Demo")
    print("=" * 70)
    
    inventor = MLStyleClassInventor()
    
    # Training phase
    print("\n📚 TRAINING PHASE")
    print("-" * 70)
    
    training_data = [
        {
            'class_name': 'Vehicle',
            'samples': [
                {'make': 'Toyota', 'model': 'Camry', 'year': 2020, 'mileage': 15000},
                {'make': 'Honda', 'model': 'Civic', 'year': 2019, 'mileage': 25000},
                {'make': 'Ford', 'model': 'F150', 'year': 2021, 'mileage': 10000},
            ]
        },
        {
            'class_name': 'Book',
            'samples': [
                {'title': '1984', 'author': 'Orwell', 'pages': 328, 'year': 1949},
                {'title': 'Dune', 'author': 'Herbert', 'pages': 412, 'year': 1965},
            ]
        }
    ]
    
    inventor.train(training_data)
    
    # Invention phase
    print("\n🔬 INVENTION PHASE")
    print("-" * 70)
    
    # Test 1: Similar to Vehicle
    print("\nTest 1: Inventing class for vehicle-like data...")
    vehicle_data = [
        {'make': 'Tesla', 'model': 'Model 3', 'year': 2022, 'mileage': 5000},
        {'make': 'BMW', 'model': 'X5', 'year': 2021, 'mileage': 12000},
    ]
    
    VehicleClass = inventor.invent_class(vehicle_data)
    print(f"✓ Created: {VehicleClass.__name__}")
    if hasattr(VehicleClass, '_learned_from'):
        print(f"  Learned from: {VehicleClass._learned_from}")
    
    car = VehicleClass(make='Audi', model='A4', year=2023, mileage=3000)
    print(f"  Instance: {car}")
    car.describe()
    
    # Test 2: Similar to Book
    print("\n\nTest 2: Inventing class for book-like data...")
    book_data = [
        {'title': 'Foundation', 'author': 'Asimov', 'pages': 255, 'year': 1951},
    ]
    
    BookClass = inventor.invent_class(book_data)
    print(f"✓ Created: {BookClass.__name__}")
    if hasattr(BookClass, '_learned_from'):
        print(f"  Learned from: {BookClass._learned_from}")
    
    book = BookClass(title='Neuromancer', author='Gibson', pages=271, year=1984)
    print(f"  Instance: {book}")
    book.describe()
    
    # Test 3: Completely new pattern
    print("\n\nTest 3: Inventing class for new pattern...")
    recipe_data = [
        {'name': 'Pasta', 'prep_time': 15, 'cook_time': 20, 'servings': 4},
        {'name': 'Salad', 'prep_time': 10, 'cook_time': 0, 'servings': 2},
    ]
    
    RecipeClass = inventor.invent_class(recipe_data, hint="Recipe")
    print(f"✓ Created: {RecipeClass.__name__}")
    print(f"  No pattern match - invented from scratch!")
    
    recipe = RecipeClass(name='Soup', prep_time=5, cook_time=30, servings=6)
    print(f"  Instance: {recipe}")
    recipe.describe()
    
    # Test 4: Validation warnings
    print("\n\nTest 4: Testing validation warnings...")
    unusual_car = VehicleClass(make='Mystery', model='X', year=2025, mileage=100000)
    print(f"  Created: {unusual_car}")
    
    print("\n" + "=" * 70)
    print("Demo complete! The AI learned patterns and invented appropriate classes.")
    print("=" * 70)


if __name__ == "__main__":
    demo_ml_style_inventor()
