"""
Type-based AI Function Approximator for Diabetes Prediction
Uses metaclass for advanced class construction and type-safe operations
"""

from typing import Union, List, Dict, Any, Optional, TypeVar, Generic, Callable
from abc import ABCMeta, abstractmethod
import math


T = TypeVar('T', bound='Numeric')


class Numeric:
    """Base type for numeric operations"""
    
    def __init__(self, value: Union[int, float]):
        self.value = float(value)
    
    def __add__(self, other: 'Numeric') -> 'Numeric':
        return Numeric(self.value + other.value)
    
    def __mul__(self, other: 'Numeric') -> 'Numeric':
        return Numeric(self.value * other.value)
    
    def __repr__(self) -> str:
        return f"Numeric({self.value})"


class AIMetaClass(ABCMeta):
    """
    Metaclass for AI function approximators
    Provides validation and automatic method registration
    """
    
    _registry: Dict[str, type] = {}
    
    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        # Validate required methods
        if name != 'BaseApproximator' and not any(isinstance(base, AIMetaClass) for base in bases):
            required_methods = ['forward', 'predict']
            for method in required_methods:
                if method not in namespace:
                    raise TypeError(f"Class {name} must implement {method} method")
        
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Register the class
        if name != 'BaseApproximator':
            mcs._registry[name] = cls
        
        return cls
    
    @classmethod
    def get_registered_models(mcs) -> Dict[str, type]:
        """Get all registered AI models"""
        return mcs._registry.copy()


class BaseApproximator(metaclass=AIMetaClass):
    """Base class for all AI function approximators"""
    
    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """Forward pass through the approximator"""
        pass
    
    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Make predictions"""
        pass


class DiabetesFeatures:
    """Type-safe diabetes feature container"""
    
    def __init__(
        self,
        glucose: Union[float, Numeric],
        bmi: Union[float, Numeric],
        age: Union[float, Numeric],
        blood_pressure: Union[float, Numeric],
        insulin: Union[float, Numeric],
        pregnancies: Optional[Union[int, Numeric]] = None
    ):
        # Convert to Numeric type for type safety
        self.glucose = glucose if isinstance(glucose, Numeric) else Numeric(glucose)
        self.bmi = bmi if isinstance(bmi, Numeric) else Numeric(bmi)
        self.age = age if isinstance(age, Numeric) else Numeric(age)
        self.blood_pressure = blood_pressure if isinstance(blood_pressure, Numeric) else Numeric(blood_pressure)
        self.insulin = insulin if isinstance(insulin, Numeric) else Numeric(insulin)
        self.pregnancies = pregnancies if isinstance(pregnancies, Numeric) else Numeric(pregnancies if pregnancies else 0)
    
    def to_list(self) -> List[float]:
        """Convert to list when array is needed"""
        return [
            self.glucose.value,
            self.bmi.value,
            self.age.value,
            self.blood_pressure.value,
            self.insulin.value,
            self.pregnancies.value
        ]
    
    def __repr__(self) -> str:
        return (f"DiabetesFeatures(glucose={self.glucose.value:.2f}, "
                f"bmi={self.bmi.value:.2f}, age={self.age.value:.2f})")


class NeuralLayer:
    """Type-based neural network layer without explicit arrays"""
    
    def __init__(self, input_size: int, output_size: int, name: str = "layer"):
        self.name = name
        self.weights: List[List[Numeric]] = []
        self.biases: List[Numeric] = []
        
        # Initialize weights and biases using type-safe approach
        for i in range(output_size):
            weight_row = []
            for j in range(input_size):
                # Small random initialization
                weight_row.append(Numeric(0.1 * math.sin(i + j + 1)))
            self.weights.append(weight_row)
            self.biases.append(Numeric(0.0))
    
    def forward(self, inputs: List[Numeric]) -> List[Numeric]:
        """Forward pass through the layer"""
        outputs = []
        
        for i, weight_row in enumerate(self.weights):
            # Compute weighted sum
            activation = self.biases[i]
            for j, input_val in enumerate(inputs):
                activation = activation + (weight_row[j] * input_val)
            outputs.append(activation)
        
        return outputs
    
    def __repr__(self) -> str:
        return f"NeuralLayer({self.name}, {len(self.weights[0])} -> {len(self.weights)})"


class DiabetesAIApproximator(BaseApproximator):
    """
    AI Function Approximator for Diabetes Prediction
    Uses type-based approach with metaclass validation
    """
    
    def __init__(self, hidden_size: int = 10):
        self.hidden_size = hidden_size
        
        # Create layers using type-based approach
        self.layer1 = NeuralLayer(6, hidden_size, "hidden")
        self.layer2 = NeuralLayer(hidden_size, 1, "output")
        
        self.trained = False
    
    def sigmoid(self, x: Numeric) -> Numeric:
        """Sigmoid activation function"""
        return Numeric(1.0 / (1.0 + math.exp(-x.value)))
    
    def relu(self, x: Numeric) -> Numeric:
        """ReLU activation function"""
        return Numeric(max(0.0, x.value))
    
    def forward(self, inputs: Union[DiabetesFeatures, List[float], List[Numeric]]) -> Numeric:
        """
        Forward pass through the approximator
        Accepts different input types and converts appropriately
        """
        # Convert input to List[Numeric] based on type
        if isinstance(inputs, DiabetesFeatures):
            numeric_inputs = [Numeric(val) for val in inputs.to_list()]
        elif isinstance(inputs, list) and len(inputs) > 0:
            if isinstance(inputs[0], Numeric):
                numeric_inputs = inputs
            else:
                # User provided array, use it
                numeric_inputs = [Numeric(val) for val in inputs]
        else:
            raise ValueError("Input must be DiabetesFeatures or list of numbers")
        
        # Hidden layer with ReLU
        hidden = self.layer1.forward(numeric_inputs)
        hidden_activated = [self.relu(h) for h in hidden]
        
        # Output layer with sigmoid
        output = self.layer2.forward(hidden_activated)
        probability = self.sigmoid(output[0])
        
        return probability
    
    def predict(self, inputs: Union[DiabetesFeatures, List[float]]) -> Dict[str, Any]:
        """
        Make diabetes risk prediction
        Returns type-safe prediction dictionary
        """
        probability = self.forward(inputs)
        
        # Determine risk level
        risk_value = probability.value
        if risk_value < 0.3:
            risk_level = "Low"
        elif risk_value < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        return {
            "probability": risk_value,
            "risk_level": risk_level,
            "has_diabetes": risk_value > 0.5,
            "confidence": abs(risk_value - 0.5) * 2  # Scale to 0-1
        }
    
    def explain(self, inputs: Union[DiabetesFeatures, List[float]]) -> str:
        """Provide explanation of prediction"""
        prediction = self.predict(inputs)
        
        explanation = f"""
Diabetes Risk Assessment:
------------------------
Probability: {prediction['probability']:.2%}
Risk Level: {prediction['risk_level']}
Diagnosis: {'Positive' if prediction['has_diabetes'] else 'Negative'}
Confidence: {prediction['confidence']:.2%}

Model: {self.__class__.__name__}
Architecture: Type-based neural approximator with metaclass validation
        """
        
        return explanation.strip()
    
    def __repr__(self) -> str:
        return f"DiabetesAIApproximator(hidden={self.hidden_size}, trained={self.trained})"


# Demonstration functions
def demonstrate_type_based_approach():
    """Demonstrate the type-based AI approximator"""
    
    print("=" * 60)
    print("Type-Based AI Function Approximator for Diabetes")
    print("Using Metaclass and Type-Safe Operations")
    print("=" * 60)
    print()
    
    # Create approximator
    model = DiabetesAIApproximator(hidden_size=8)
    print(f"Created model: {model}")
    print(f"Layer 1: {model.layer1}")
    print(f"Layer 2: {model.layer2}")
    print()
    
    # Test with type-safe DiabetesFeatures
    print("Test 1: Using DiabetesFeatures (type-based, no arrays)")
    print("-" * 60)
    patient1 = DiabetesFeatures(
        glucose=140.0,
        bmi=28.5,
        age=45.0,
        blood_pressure=85.0,
        insulin=120.0,
        pregnancies=2
    )
    print(f"Patient: {patient1}")
    print(model.explain(patient1))
    print()
    
    # Test with array when user provides it
    print("Test 2: Using array (when user provides array)")
    print("-" * 60)
    patient2_array = [180.0, 35.2, 60.0, 95.0, 200.0, 5.0]
    print(f"Patient data (array): {patient2_array}")
    print(model.explain(patient2_array))
    print()
    
    # Show registered models via metaclass
    print("Registered AI Models (via metaclass):")
    print("-" * 60)
    for name, cls in AIMetaClass.get_registered_models().items():
        print(f"  - {name}: {cls}")
    print()
    
    # Demonstrate type operations
    print("Type-based numeric operations:")
    print("-" * 60)
    a = Numeric(5.0)
    b = Numeric(3.0)
    print(f"{a} + {b} = {a + b}")
    print(f"{a} * {b} = {a * b}")
    print()


if __name__ == "__main__":
    demonstrate_type_based_approach()
