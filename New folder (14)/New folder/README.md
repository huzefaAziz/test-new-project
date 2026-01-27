# AI Class Inventor - Automatic Class Generation in Python

This project demonstrates AI-like automatic class invention using Python metaprogramming. The system analyzes data patterns and dynamically creates appropriate classes with methods and validation.

## 🎯 Features

### 1. **Basic Class Invention** (`ClassInventor`)
- Analyzes data samples to infer field types
- Identifies common and optional fields
- Creates classes dynamically with appropriate attributes
- Auto-generates `__init__`, `__repr__`, validation methods

### 2. **Pattern-Based Intelligence** (`PatternBasedInventor`)
- Detects patterns in data (coordinates, timestamps, prices)
- Automatically adds relevant methods based on patterns
  - Location data → gets `distance_from_origin()`, `move()` methods
  - Price data → gets `apply_discount()` method
  - Timestamp data → gets temporal methods

### 3. **Class Hierarchies** (`InheritanceInventor`)
- Creates base classes from common attributes
- Generates specialized subclasses automatically
- Maintains proper inheritance relationships

### 4. **ML-Style Pattern Learning** (`MLStyleClassInventor`)
- "Trains" on example data to learn patterns
- Matches new data against learned patterns
- Validates data against typical ranges
- Provides warnings for outlier values

## 🚀 Quick Start

### Basic Usage

```python
from class_inventor import ClassInventor

# Your data samples
user_data = [
    {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'},
    {'name': 'Bob', 'age': 25, 'email': 'bob@example.com'},
]

# Invent a class!
inventor = ClassInventor()
UserClass = inventor.analyze_and_invent(user_data, "User")

# Create instances
user = UserClass(name='Charlie', age=28, email='charlie@example.com')
print(user)  # User_0(name='Charlie', age=28, email='charlie@example.com')
```

### Pattern-Based Class Creation

```python
from class_inventor import PatternBasedInventor

# Location data with x, y coordinates
location_data = [
    {'x': 10, 'y': 20, 'name': 'Point A'},
    {'x': 30, 'y': 40, 'name': 'Point B'},
]

# Creates class with automatic location methods
inventor = PatternBasedInventor()
LocationClass = inventor.analyze_and_invent(location_data, "Location")

# Use the auto-generated methods!
point = LocationClass(x=3, y=4, name='Origin')
print(point.distance_from_origin())  # 5.0
point.move(1, 1)
print(point)  # Location_0(x=4, y=5, name='Origin')
```

### ML-Style Training and Invention

```python
from ml_class_inventor import MLStyleClassInventor

inventor = MLStyleClassInventor()

# Training phase - learn patterns
training_data = [
    {
        'class_name': 'Vehicle',
        'samples': [
            {'make': 'Toyota', 'model': 'Camry', 'year': 2020, 'mileage': 15000},
            {'make': 'Honda', 'model': 'Civic', 'year': 2019, 'mileage': 25000},
        ]
    }
]
inventor.train(training_data)

# Invention phase - apply learned patterns
new_vehicle_data = [
    {'make': 'Tesla', 'model': 'Model 3', 'year': 2022, 'mileage': 5000},
]

VehicleClass = inventor.invent_class(new_vehicle_data)
car = VehicleClass(make='Audi', model='A4', year=2023, mileage=3000)
car.describe()  # Shows detailed information with validation
```

## 📋 How It Works

### Type Inference
```python
# Analyzes samples to determine field types
{'name': str, 'age': int, 'email': str}
```

### Pattern Detection
```python
# Detects patterns and adds appropriate methods
has_coordinates (x, y) → adds distance_from_origin(), move()
has_price → adds apply_discount()
has_timestamp → adds temporal methods
```

### Dynamic Class Creation
```python
# Uses Python's type() to create classes at runtime
new_class = type(
    'ClassName',           # Class name
    (BaseClass,),          # Base classes
    {                      # Class attributes
        '__init__': init_method,
        'custom_method': method,
    }
)
```

### Validation
```python
# Auto-generated validation checks types and ranges
is_valid, message = instance.validate()
```

## 🎨 Use Cases

1. **Data Modeling**: Automatically generate data models from JSON/CSV files
2. **API Responses**: Create classes for API response structures
3. **Configuration**: Generate config classes from example configs
4. **Testing**: Create mock objects from test data
5. **Database ORMs**: Generate model classes from database schemas
6. **Machine Learning**: Create data classes for ML pipeline outputs

## 🔧 Advanced Features

### Automatic Method Generation
The system can generate appropriate methods based on detected patterns:

```python
# For coordinate data
.distance_from_origin()
.move(dx, dy)

# For financial data
.apply_discount(percentage)

# For temporal data
.is_recent(days)
```

### Range Validation
Learns typical value ranges and warns about outliers:

```python
# Warns if value is outside 80-120% of learned range
Warning: mileage=100000 is outside typical range [5000, 25000]
```

### Class Hierarchies
Automatically creates inheritance relationships:

```python
BaseClass
├── EmployeeClass (with salary, department)
└── CustomerClass (with loyalty_points)
```

## 📊 Examples Included

Run the demos to see the system in action:

```bash
python class_inventor.py        # Basic demos
python ml_class_inventor.py     # ML-style learning demo
```

## 🧠 Technical Details

- Uses `type()` for dynamic class creation
- Implements metaprogramming patterns
- Type inference from sample data
- Pattern matching using set operations (Jaccard similarity)
- Automatic method injection
- Runtime class modification

## 🎓 Learning Points

1. **Metaprogramming**: Creating classes at runtime
2. **Type Inference**: Determining types from data
3. **Pattern Recognition**: Detecting structure in data
4. **Dynamic Attributes**: Using `setattr()` and `getattr()`
5. **Inheritance**: Programmatic class hierarchies
6. **Closures**: Methods that reference class creation context

## 🚧 Limitations

- Simple type inference (no complex nested structures)
- Pattern detection is rule-based, not truly ML
- No persistence between sessions
- Limited to basic Python types

## 🔮 Future Enhancements

- [ ] Support for nested/complex types
- [ ] JSON Schema generation
- [ ] Database schema integration
- [ ] True ML-based pattern recognition
- [ ] Class versioning and migration
- [ ] Automatic documentation generation
- [ ] Type hints and static typing support

## 📝 License

Free to use and modify for any purpose.

---

**Created with Python's metaprogramming capabilities** 🐍✨
