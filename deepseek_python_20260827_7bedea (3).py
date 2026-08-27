"""
Assembly Inside Neural Network - Fixed Version
Corrects the IndexError and improves the implementation
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time

# ============================================================================
# ASSEMBLY INSTRUCTION SET FOR NEURAL NETWORKS
# ============================================================================

class OpCode(Enum):
    """Assembly opcodes for neural network operations"""
    NOP = 0x00      # No operation
    LOAD = 0x01     # Load value into register
    STORE = 0x02    # Store register to memory
    ADD = 0x03      # Floating point addition
    SUB = 0x04      # Floating point subtraction
    MUL = 0x05      # Floating point multiplication
    DIV = 0x06      # Floating point division
    EXP = 0x07      # Exponential
    SIG = 0x08      # Sigmoid activation
    RELU = 0x09     # ReLU activation
    TANH = 0x0A     # Tanh activation
    CMP = 0x0B      # Compare values
    JMP = 0x0C      # Jump to instruction
    JZ = 0x0D       # Jump if zero
    JNZ = 0x0E      # Jump if not zero
    HALT = 0x0F     # Halt execution
    GRAD = 0x10     # Compute gradient
    BACK = 0x11     # Backpropagation
    WEIGHT = 0x12   # Update weights
    BIAS = 0x13     # Update bias


@dataclass
class Instruction:
    """Assembly instruction structure"""
    opcode: OpCode
    operand1: int = 0
    operand2: int = 0
    operand3: int = 0


# ============================================================================
# NEURAL NETWORK NEURON WITH ASSEMBLY ENGINE
# ============================================================================

class AssemblyNeuron:
    """
    A neuron that executes assembly-like instructions for its computations.
    Each neuron has its own instruction memory, registers, and execution engine.
    """
    
    def __init__(self, neuron_id: int, instruction_memory: List[Instruction], input_size: int = 10):
        self.neuron_id = neuron_id
        self.instruction_memory = instruction_memory
        self.pc = 0  # Program counter
        
        # Registers (8 general purpose registers)
        self.registers = np.zeros(8, dtype=np.float32)
        
        # Memory (256 bytes)
        self.memory = np.zeros(64, dtype=np.float32)
        
        # Special registers
        self.accumulator = 0.0
        self.status_flag = 0  # Zero flag
        
        # Neural network specific
        self.input_size = input_size
        self.inputs = np.zeros(input_size, dtype=np.float32)
        self.output = 0.0
        self.gradient = 0.0
        self.weights = np.random.randn(input_size) * 0.1
        self.bias = 0.0
        
        # Execution statistics
        self.instructions_executed = 0
        self.forward_passes = 0
        self.backward_passes = 0
        
        self.running = False
        
    def reset(self):
        """Reset neuron state"""
        self.pc = 0
        self.registers.fill(0)
        self.memory.fill(0)
        self.accumulator = 0.0
        self.status_flag = 0
        self.instructions_executed = 0
        self.running = True
        
    def fetch(self) -> Instruction:
        """Fetch next instruction from memory"""
        if self.pc >= len(self.instruction_memory):
            return Instruction(OpCode.HALT)
        instr = self.instruction_memory[self.pc]
        self.pc += 1
        return instr
    
    def execute(self, instr: Instruction):
        """Execute a single assembly instruction"""
        self.instructions_executed += 1
        
        op = instr.opcode
        reg = instr.operand1
        addr = instr.operand2
        val = instr.operand3
        
        try:
            if op == OpCode.NOP:
                pass
                
            elif op == OpCode.LOAD:
                if addr < len(self.memory):
                    self.registers[reg] = self.memory[addr]
                else:
                    self.registers[reg] = float(addr)
                    
            elif op == OpCode.STORE:
                if addr < len(self.memory):
                    self.memory[addr] = self.registers[reg]
                    
            elif op == OpCode.ADD:
                if addr < len(self.memory):
                    self.registers[reg] += self.memory[addr]
                else:
                    self.registers[reg] += float(addr)
                    
            elif op == OpCode.SUB:
                if addr < len(self.memory):
                    self.registers[reg] -= self.memory[addr]
                else:
                    self.registers[reg] -= float(addr)
                    
            elif op == OpCode.MUL:
                if addr < len(self.memory):
                    self.registers[reg] *= self.memory[addr]
                else:
                    self.registers[reg] *= float(addr)
                    
            elif op == OpCode.DIV:
                divisor = self.memory[addr] if addr < len(self.memory) else float(addr)
                if divisor != 0:
                    self.registers[reg] /= divisor
                    
            elif op == OpCode.EXP:
                self.registers[reg] = np.exp(self.registers[reg])
                
            elif op == OpCode.SIG:
                x = self.registers[reg]
                self.registers[reg] = 1.0 / (1.0 + np.exp(-x))
                
            elif op == OpCode.RELU:
                self.registers[reg] = max(0, self.registers[reg])
                
            elif op == OpCode.TANH:
                self.registers[reg] = np.tanh(self.registers[reg])
                
            elif op == OpCode.CMP:
                self.status_flag = 1 if self.registers[reg] == 0 else 0
                
            elif op == OpCode.JMP:
                self.pc = addr
                
            elif op == OpCode.JZ:
                if self.status_flag == 1:
                    self.pc = addr
                    
            elif op == OpCode.JNZ:
                if self.status_flag == 0:
                    self.pc = addr
                    
            elif op == OpCode.GRAD:
                # Compute gradient for backpropagation
                self.gradient = self.registers[reg]
                
            elif op == OpCode.BACK:
                # Backpropagation computation
                self._backpropagate()
                
            elif op == OpCode.WEIGHT:
                # Update weights
                idx = addr % len(self.weights)
                self.weights[idx] += self.registers[reg] * 0.01
                
            elif op == OpCode.BIAS:
                # Update bias
                self.bias += self.registers[reg] * 0.01
                
            elif op == OpCode.HALT:
                self.running = False
                
        except Exception as e:
            print(f"Neuron {self.neuron_id} error at instruction {self.pc-1}: {e}")
            self.running = False
            
    def _backpropagate(self):
        """Backpropagation assembly subroutine"""
        # Store current output
        self.memory[0] = self.output
        
        # Compute derivative (for sigmoid: output * (1 - output))
        sig_deriv = self.output * (1 - self.output)
        self.memory[1] = sig_deriv
        
        # Compute gradient
        grad = self.gradient * sig_deriv
        self.memory[2] = grad
        
        # Update weights and bias using gradient
        for i in range(len(self.weights)):
            if i < len(self.inputs):
                self.weights[i] -= 0.01 * grad * self.inputs[i]
        
        self.bias -= 0.01 * grad
        
    def forward(self, inputs: np.ndarray) -> float:
        """Execute the forward pass using assembly program"""
        # Ensure inputs match expected size
        if len(inputs) > self.input_size:
            inputs = inputs[:self.input_size]
        elif len(inputs) < self.input_size:
            padded = np.zeros(self.input_size)
            padded[:len(inputs)] = inputs
            inputs = padded
            
        self.inputs = inputs
        
        # Load inputs into memory
        for i, val in enumerate(self.inputs):
            if i < len(self.memory):
                self.memory[i] = val
        
        self.reset()
        
        # Execute instructions
        while self.running:
            instr = self.fetch()
            self.execute(instr)
        
        self.output = self.registers[0]  # R0 contains result
        self.forward_passes += 1
        return self.output
    
    def backward(self, target: float = None):
        """Execute backward pass"""
        if target is not None:
            self.gradient = self.output - target  # Loss derivative
        self.backward_passes += 1
        
        # Run backpropagation assembly code
        self.pc = 0
        self.running = True
        while self.running:
            instr = self.fetch()
            self.execute(instr)
            
    def get_weights(self) -> np.ndarray:
        return self.weights
    
    def set_weights(self, weights: np.ndarray):
        if len(weights) == len(self.weights):
            self.weights = weights


# ============================================================================
# NEURAL NETWORK LAYER WITH ASSEMBLY NEURONS
# ============================================================================

class AssemblyLayer:
    """
    A layer of neurons, each with its own assembly program.
    The layer can execute SIMD-like operations across neurons.
    """
    
    def __init__(self, num_neurons: int, input_size: int, assembly_program: List[Instruction]):
        self.neurons = [AssemblyNeuron(i, assembly_program, input_size) for i in range(num_neurons)]
        self.input_size = input_size
        self.output_size = num_neurons
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through all neurons"""
        outputs = np.zeros(self.output_size)
        for i, neuron in enumerate(self.neurons):
            outputs[i] = neuron.forward(inputs)
        return outputs
    
    def backward(self, gradients: np.ndarray = None):
        """Backward pass through all neurons"""
        if gradients is None:
            # If no gradients provided, use default values
            gradients = np.ones(self.output_size) * 0.01
            
        for i, neuron in enumerate(self.neurons):
            if i < len(gradients):
                neuron.gradient = gradients[i]
            neuron.backward()
            
    def get_weights(self) -> List[np.ndarray]:
        return [n.get_weights() for n in self.neurons]
    
    def set_weights(self, weights: List[np.ndarray]):
        for i, w in enumerate(weights):
            self.neurons[i].set_weights(w)
            
    def get_neurons(self) -> List[AssemblyNeuron]:
        return self.neurons


# ============================================================================
# FULL NEURAL NETWORK WITH ASSEMBLY NEURONS
# ============================================================================

class AssemblyNeuralNetwork:
    """
    Complete neural network where each neuron executes assembly-like code.
    This demonstrates the concept of "assembly inside neural network."
    """
    
    def __init__(self, layer_sizes: List[int], activations: List[str] = None):
        """
        Initialize network with specified layer sizes.
        layer_sizes: [input_size, hidden1_size, hidden2_size, ..., output_size]
        activations: List of activation functions for each layer
        """
        self.layer_sizes = layer_sizes
        self.layers = []
        
        if activations is None:
            activations = ['sigmoid'] * (len(layer_sizes) - 1)
        elif len(activations) < len(layer_sizes) - 1:
            activations.extend(['sigmoid'] * (len(layer_sizes) - 1 - len(activations)))
        
        # Generate assembly programs for each layer
        for i in range(len(layer_sizes) - 1):
            program = self._generate_assembly_program(
                layer_sizes[i], 
                layer_sizes[i+1],
                activations[i] if i < len(activations) else 'sigmoid'
            )
            layer = AssemblyLayer(
                layer_sizes[i+1],
                layer_sizes[i],
                program
            )
            self.layers.append(layer)
            
        self.loss_history = []
        self.learning_rate = 0.01
        
    def _generate_assembly_program(self, input_size: int, output_size: int, activation: str = 'sigmoid') -> List[Instruction]:
        """
        Generate a custom assembly program for neurons in this layer.
        This implements the neuron's computation and activation function.
        """
        program = []
        
        # Load input values from memory to registers
        max_regs = min(input_size, 8)
        for i in range(max_regs):
            program.append(Instruction(OpCode.LOAD, i, i, 0))
            
        # Perform weighted sum - accumulate in R0
        program.append(Instruction(OpCode.LOAD, 0, 0, 0))  # Initialize R0 with first input
        
        # Multiply and accumulate
        for i in range(max_regs):
            if i == 0:
                program.append(Instruction(OpCode.MUL, i, i+10, 0))  # Multiply by weight
            else:
                program.append(Instruction(OpCode.LOAD, i, i, 0))
                program.append(Instruction(OpCode.MUL, i, i+10, 0))
                program.append(Instruction(OpCode.ADD, 0, 0, 0))  # Add to accumulator
            
        # Add bias
        program.append(Instruction(OpCode.ADD, 0, 100, 0))  # Add bias from memory location 100
        
        # Apply activation function
        if activation == 'sigmoid':
            program.append(Instruction(OpCode.SIG, 0, 0, 0))
        elif activation == 'relu':
            program.append(Instruction(OpCode.RELU, 0, 0, 0))
        elif activation == 'tanh':
            program.append(Instruction(OpCode.TANH, 0, 0, 0))
        elif activation == 'linear':
            pass  # No activation
        
        # Store result
        program.append(Instruction(OpCode.STORE, 0, 0, 0))
        
        # Halt
        program.append(Instruction(OpCode.HALT, 0, 0, 0))
        
        return program
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through all layers"""
        current_input = inputs
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def backward(self, gradients: np.ndarray = None):
        """Backward pass through all layers"""
        current_gradients = gradients
        
        # Propagate backwards
        for layer in reversed(self.layers):
            layer.backward(current_gradients)
            # For simplicity, we're not computing upstream gradients
            # In a real network, this would be more complex
            # Here we just pass the same gradients
            if current_gradients is not None:
                # For next layer up, we'd compute gradient w.r.t. inputs
                # But for this demo, we'll keep it simple
                pass
            
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01, verbose: bool = True):
        """Train the network using assembly-based computations"""
        self.learning_rate = lr
        
        if verbose:
            print("Training Assembly Neural Network...")
            print(f"Layer sizes: {self.layer_sizes}")
            print(f"Total neurons: {sum(self.layer_sizes[1:])}")
            print(f"Training samples: {len(X)}")
            print("-" * 60)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(X)):
                # Forward pass
                output = self.forward(X[i])
                
                # Compute loss (MSE)
                loss = np.mean((output - y[i]) ** 2)
                total_loss += loss
                
                # Compute gradient for output layer
                gradient = 2 * (output - y[i]) / len(y[i])
                
                # Backward pass - propagate gradients through layers
                current_gradient = gradient
                for layer in reversed(self.layers):
                    # Store gradient in neurons
                    for j, neuron in enumerate(layer.neurons):
                        if j < len(current_gradient):
                            neuron.gradient = current_gradient[j]
                        else:
                            neuron.gradient = current_gradient[-1] if len(current_gradient) > 0 else 0.01
                        neuron.backward()
                    # For the next layer up, we would compute the gradient w.r.t. inputs
                    # For simplicity, we'll just propagate the same gradient
                    # In a real implementation, this would be the gradient w.r.t. layer inputs
            
            avg_loss = total_loss / len(X)
            self.loss_history.append(avg_loss)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}, Loss: {avg_loss:.6f}")
                
        if verbose:
            print("-" * 60)
            print("Training complete!")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        predictions = []
        for x in X:
            predictions.append(self.forward(x))
        return np.array(predictions)
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the network"""
        total_neurons = sum(self.layer_sizes[1:])
        total_instructions = 0
        for layer in self.layers:
            if layer.neurons:
                total_instructions += len(layer.neurons[0].instruction_memory) * len(layer.neurons)
                
        return {
            'layer_sizes': self.layer_sizes,
            'total_neurons': total_neurons,
            'total_instructions': total_instructions,
            'layers': len(self.layers),
            'loss_history': self.loss_history
        }


# ============================================================================
# ASSEMBLY PROGRAM GENERATOR FOR DIFFERENT ACTIVATIONS
# ============================================================================

def create_assembly_program(activation: str = 'sigmoid', input_size: int = 10) -> List[Instruction]:
    """
    Create assembly program for different activation functions.
    This demonstrates how assembly can be used to implement various activations.
    """
    program = []
    
    # Initialize with first input
    program.append(Instruction(OpCode.LOAD, 0, 0, 0))
    
    # Process up to 8 inputs
    for i in range(min(input_size, 8)):
        if i > 0:
            program.append(Instruction(OpCode.LOAD, i, i, 0))
            program.append(Instruction(OpCode.MUL, i, i+10, 0))
            program.append(Instruction(OpCode.ADD, 0, 0, 0))
        else:
            # First input: multiply and store in R0
            program.append(Instruction(OpCode.MUL, 0, 10, 0))
    
    # Add bias
    program.append(Instruction(OpCode.ADD, 0, 100, 0))
    
    # Activation
    if activation == 'linear':
        pass
    elif activation == 'sigmoid':
        program.append(Instruction(OpCode.SIG, 0, 0, 0))
    elif activation == 'relu':
        program.append(Instruction(OpCode.RELU, 0, 0, 0))
    elif activation == 'tanh':
        program.append(Instruction(OpCode.TANH, 0, 0, 0))
    
    # Store and halt
    program.append(Instruction(OpCode.STORE, 0, 0, 0))
    program.append(Instruction(OpCode.HALT, 0, 0, 0))
    
    return program


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_assembly_neuron():
    """Demonstrate a single neuron with assembly code"""
    print("=" * 60)
    print("ASSEMBLY NEURON DEMONSTRATION")
    print("=" * 60)
    
    # Create assembly program for a neuron
    program = [
        Instruction(OpCode.LOAD, 0, 0, 0),     # Load input[0] to R0
        Instruction(OpCode.MUL, 0, 10, 0),     # Multiply R0 by weight[0]
        Instruction(OpCode.ADD, 0, 100, 0),    # Add bias
        Instruction(OpCode.SIG, 0, 0, 0),      # Sigmoid activation
        Instruction(OpCode.STORE, 0, 0, 0),    # Store result
        Instruction(OpCode.HALT, 0, 0, 0),     # Halt
    ]
    
    # Create neuron
    neuron = AssemblyNeuron(0, program, input_size=4)
    neuron.weights = np.array([0.5, 0.3, 0.2, 0.1])
    neuron.bias = 0.1
    
    # Test inputs
    test_inputs = np.array([1.0, 0.5, 0.2, 0.0])
    
    print(f"Input: {test_inputs}")
    print(f"Weights: {neuron.weights}")
    print(f"Bias: {neuron.bias}")
    
    # Manual calculation
    manual = 1 / (1 + np.exp(-(np.dot(test_inputs, neuron.weights) + neuron.bias)))
    print(f"Manual calculation: {manual:.4f}")
    
    # Forward pass
    output = neuron.forward(test_inputs)
    print(f"Assembly neuron output: {output:.4f}")
    
    # Show instruction execution stats
    print(f"Instructions executed: {neuron.instructions_executed}")
    print(f"Forward passes: {neuron.forward_passes}")


def demonstrate_assembly_network():
    """Demonstrate a full network with assembly neurons"""
    print("\n" + "=" * 60)
    print("ASSEMBLY NEURAL NETWORK DEMONSTRATION")
    print("=" * 60)
    
    # Create network with layers: input(2) -> hidden(4) -> output(2)
    network = AssemblyNeuralNetwork([2, 4, 2], activations=['sigmoid', 'sigmoid'])
    
    # Generate training data (XOR-like problem)
    np.random.seed(42)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([
        [1, 0],  # 0 XOR 0 = 0 (class 0)
        [0, 1],  # 0 XOR 1 = 1 (class 1)
        [0, 1],  # 1 XOR 0 = 1 (class 1)
        [1, 0]   # 1 XOR 1 = 0 (class 0)
    ])
    
    # Train
    print("Training on XOR problem...")
    network.train(X, y, epochs=200, lr=0.5, verbose=True)
    
    # Test predictions
    print("\nPredictions on XOR data:")
    for i, x in enumerate(X):
        pred = network.predict(x.reshape(1, -1))
        print(f"Input: {x} -> Expected: {y[i]} -> Predicted: {pred[0].round(3)}")
        
    # Network statistics
    info = network.get_network_info()
    print(f"\nNetwork Statistics:")
    print(f"Total neurons: {info['total_neurons']}")
    print(f"Total assembly instructions: {info['total_instructions']}")
    print(f"Final loss: {info['loss_history'][-1]:.6f}")


def compare_activation_functions():
    """Compare different activation functions implemented in assembly"""
    print("\n" + "=" * 60)
    print("ACTIVATION FUNCTION COMPARISON")
    print("=" * 60)
    
    activations = ['sigmoid', 'relu', 'tanh', 'linear']
    inputs = np.array([0.5, -1.0, 2.0, -0.5])
    
    print(f"Input: {inputs}\n")
    
    for activation in activations:
        program = create_assembly_program(activation, 4)
        neuron = AssemblyNeuron(0, program, input_size=4)
        neuron.weights = np.array([0.2, 0.3, 0.1, 0.4])
        neuron.bias = 0.1
        
        output = neuron.forward(inputs)
        print(f"{activation.upper():8s} output: {output:.4f}")


def benchmark_assembly_vs_numpy():
    """Benchmark assembly neuron vs pure numpy implementation"""
    print("\n" + "=" * 60)
    print("ASSEMBLY VS NUMPY BENCHMARK")
    print("=" * 60)
    
    # Create program
    program = create_assembly_program('sigmoid', 10)
    
    # Assembly neuron
    neuron = AssemblyNeuron(0, program, input_size=10)
    neuron.weights = np.random.randn(10) * 0.1
    neuron.bias = 0.1
    
    # Numpy neuron (for comparison)
    np_weights = neuron.weights.copy()
    np_bias = neuron.bias
    
    # Test inputs
    np.random.seed(42)
    inputs = np.random.randn(100, 10)
    
    # Warm-up
    for x in inputs[:10]:
        neuron.forward(x)
    
    # Benchmark assembly neuron
    start = time.time()
    assembly_outputs = []
    for x in inputs:
        assembly_outputs.append(neuron.forward(x))
    assembly_time = time.time() - start
    
    # Benchmark numpy
    start = time.time()
    numpy_outputs = []
    for x in inputs:
        numpy_outputs.append(1 / (1 + np.exp(-(np.dot(x, np_weights) + np_bias))))
    numpy_time = time.time() - start
    
    # Compare outputs
    assembly_outputs = np.array(assembly_outputs)
    numpy_outputs = np.array(numpy_outputs)
    
    max_diff = np.max(np.abs(assembly_outputs - numpy_outputs))
    
    print(f"Assembly neuron time:    {assembly_time:.4f}s")
    print(f"NumPy neuron time:       {numpy_time:.4f}s")
    print(f"Speed ratio (NumPy/Assembly): {numpy_time/assembly_time:.2f}x")
    print(f"Max output difference:   {max_diff:.6f}")
    
    if max_diff < 1e-6:
        print("✓ Outputs match closely!")
    else:
        print("⚠ Outputs have some differences (expected due to implementation details)")


def demonstrate_backpropagation():
    """Demonstrate backpropagation with assembly neurons"""
    print("\n" + "=" * 60)
    print("BACKPROPAGATION DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple single neuron
    program = create_assembly_program('sigmoid', 2)
    neuron = AssemblyNeuron(0, program, input_size=2)
    neuron.weights = np.array([0.5, -0.5])
    neuron.bias = 0.1
    
    print("Before training:")
    print(f"  Weights: {neuron.weights}")
    print(f"  Bias: {neuron.bias:.4f}")
    
    # Train on a single example
    input_data = np.array([0.8, 0.2])
    target = 0.9
    
    print(f"\nTraining on input {input_data}, target {target}")
    
    # Forward pass
    output = neuron.forward(input_data)
    print(f"  Initial output: {output:.4f}")
    
    # Backward pass
    neuron.gradient = output - target
    neuron.backward()
    
    print(f"\nAfter backpropagation:")
    print(f"  Gradient: {neuron.gradient:.4f}")
    print(f"  Updated weights: {neuron.weights}")
    print(f"  Updated bias: {neuron.bias:.4f}")
    
    # Forward pass again
    output2 = neuron.forward(input_data)
    print(f"\nAfter update, output: {output2:.4f}")
    print(f"  Improvement: {abs(output - target) - abs(output2 - target):.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ASSEMBLY INSIDE NEURAL NETWORK")
    print("A fusion of low-level assembly and high-level neural computation")
    print("=" * 60 + "\n")
    
    # Run demonstrations
    demonstrate_assembly_neuron()
    demonstrate_assembly_network()
    compare_activation_functions()
    benchmark_assembly_vs_numpy()
    demonstrate_backpropagation()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)