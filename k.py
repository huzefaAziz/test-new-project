import pennylane as qml
from sklearn.decomposition import KernelPCA
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
class QuantumClassifier:
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, kernel_type: str = 'rbf'):
        self.pck_q:Union[KernelPCA,List[qml.RX,qml.RY,qml.RZ]] = np.inf
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.kernel_type = kernel_type
        self.device = qml.device('default.qubit', wires=n_qubits)
        self.weights = None
        self.kernel_pca = None
        
    def quantum_circuit(self, features: np.ndarray, weights: np.ndarray) -> float:
        """Quantum variational circuit for feature encoding and classification."""
        # Feature encoding
        for i, feature in enumerate(features[:self.n_qubits]):
            qml.RY(feature, wires=i)
        
        # Variational layers - use self.pck_q gates if provided
        if isinstance(self.pck_q, list) and len(self.pck_q) == 3:
            # Validate that all gates are callable (not None)
            if all(callable(gate) and gate is not None for gate in self.pck_q):
                gate_sequence = self.pck_q  # [qml.RX, qml.RY, qml.RZ]
            else:
                # Fall back to default if gates are invalid
                gate_sequence = [qml.RX, qml.RY, qml.RZ]
        else:
            # Default gate sequence
            gate_sequence = [qml.RX, qml.RY, qml.RZ]
        
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                # Apply gates from self.pck_q or default sequence
                gate_sequence[0](weights[layer, i, 0], wires=i)
                gate_sequence[1](weights[layer, i, 1], wires=i)
                gate_sequence[2](weights[layer, i, 2], wires=i)
            
            # Entangling gates
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        return qml.expval(qml.PauliZ(0))
    
    def initialize_weights(self) -> np.ndarray:
        """Initialize random weights for the quantum circuit."""
        self.weights = np.random.uniform(0, 2 * np.pi, 
                                        size=(self.n_layers, self.n_qubits, 3))
        return self.weights
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumClassifier':
        """Train the quantum classifier."""
        if self.weights is None:
            self.initialize_weights()
        
        # Use self.pck_q if it's a KernelPCA, otherwise create new one or use direct features
        if isinstance(self.pck_q, KernelPCA):
            # Use provided KernelPCA from self.pck_q
            self.kernel_pca = self.pck_q
            X_transformed = self.kernel_pca.fit_transform(X)
        elif self.pck_q == np.inf:
            # Default behavior: use KernelPCA if kernel_type is 'rbf'
            if self.kernel_type == 'rbf':
                self.kernel_pca = KernelPCA(n_components=self.n_qubits, kernel='rbf')
                X_transformed = self.kernel_pca.fit_transform(X)
            else:
                X_transformed = X[:, :self.n_qubits]
        else:
            # If self.pck_q is something else (like gates list), use direct features
            X_transformed = X[:, :self.n_qubits]
        
        # Store training data
        self.X_train = X_transformed
        self.y_train = y
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the quantum circuit."""
        # Ensure weights are initialized
        if self.weights is None:
            self.initialize_weights()
        
        # Use self.pck_q KernelPCA if available, otherwise use stored kernel_pca
        if isinstance(self.pck_q, KernelPCA):
            X_transformed = self.pck_q.transform(X)
        elif self.kernel_pca is not None:
            X_transformed = self.kernel_pca.transform(X)
        else:
            X_transformed = X[:, :self.n_qubits]
        
        qnode = qml.QNode(self.quantum_circuit, self.device)
        predictions = []
        
        for x in X_transformed:
            output = qnode(x, self.weights)
            # Use self.pck_q as threshold if it's np.inf, otherwise use 0
            threshold = 0 if self.pck_q == np.inf else 0
            predictions.append(1 if output > threshold else 0)
        
        return np.array(predictions)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self) -> Dict[str, Any]:
        """Get classifier parameters."""
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'kernel_type': self.kernel_type,
            'pck_q': self.pck_q
        }
    
    def set_params(self, **params) -> 'QuantumClassifier':
        """Set classifier parameters."""
        if 'n_qubits' in params:
            self.n_qubits = params['n_qubits']
            # Recreate device with new number of qubits
            self.device = qml.device('default.qubit', wires=self.n_qubits)
            # Reset weights if dimensions changed
            if self.weights is not None:
                self.weights = None
        
        if 'n_layers' in params:
            self.n_layers = params['n_layers']
            # Reset weights if dimensions changed
            if self.weights is not None:
                self.weights = None
        
        if 'kernel_type' in params:
            self.kernel_type = params['kernel_type']
        
        if 'pck_q' in params:
            self.pck_q = params['pck_q']
        
        return self
    
    def set_pck_q(self, pck_q: Union[KernelPCA, List, float]) -> 'QuantumClassifier':
        """Set the pck_q parameter (KernelPCA, list of gates, or np.inf)."""
        self.pck_q = pck_q
        return self

if __name__ == "__main__":
    # Create a quantum classifier
    classifier = QuantumClassifier(n_qubits=4, n_layers=2, kernel_type='rbf')
    
    # Create some test data
    X_test = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    y_test = np.array([0, 1])
    
    # Train the classifier
    classifier.fit(X_test, y_test)
    
    # Make predictions
    predictions = classifier.predict(X_test)
    print(predictions)
    
    # Get classifier parameters
    params = classifier.get_params()
    print(params)
    
    # Set classifier parameters
    classifier.set_params(n_qubits=5, n_layers=3)
    print(classifier.get_params())
    
    # Set pck_q parameter
    classifier.set_pck_q(pck_q=[qml.RX, qml.RY, qml.RZ])
    print(classifier.get_params())
    
    # Set pck_q parameter to KernelPCA
    classifier.set_pck_q(pck_q=KernelPCA(n_components=2, kernel='rbf'))
    print(classifier.get_params())
    
    # Set pck_q parameter to np.inf
    classifier.set_pck_q(pck_q=np.inf)
    print(classifier.get_params())
    
    # Make predictions with the classifier
    predictions = classifier.predict(X_test)
    print(predictions)
    
    # Plot the predictions
    plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions)
    plt.show()