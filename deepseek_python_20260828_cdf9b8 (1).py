"""
Circuit Equivalence Benchmark Generator - Fully Fixed Version
Based on: "Challenging Benchmarks for Diagrammatic Equivalence of Circuits in TPTP and SMT-LIB"
arXiv:2608.27087v1
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict, Any
from enum import Enum
import random
import itertools
from abc import ABC, abstractmethod


class CircuitType(Enum):
    """Types of circuits in the benchmark"""
    GENERAL = "general"
    PERMUTATION = "permutation"
    RESTRICTED_PERMUTATION = "restricted_permutation"


@dataclass
class Generator:
    """A generator (box) in a circuit"""
    name: str
    domain: int
    codomain: int
    
    def __repr__(self):
        return f"{self.name}_{self.domain}_{self.codomain}"


class Circuit(ABC):
    """Abstract base class for circuits"""
    
    @abstractmethod
    def domain(self) -> int:
        pass
    
    @abstractmethod
    def codomain(self) -> int:
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def to_tptp(self, prefix: str = "") -> str:
        """Convert to TPTP format"""
        pass
    
    @abstractmethod
    def to_smtlib(self, prefix: str = "") -> str:
        """Convert to SMT-LIB format"""
        pass
    
    @abstractmethod
    def is_permutation(self) -> bool:
        """Check if circuit represents a permutation"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get the size/complexity of the circuit"""
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        """Check structural equality"""
        pass


class Identity(Circuit):
    """Identity circuit id_n"""
    
    def __init__(self, n: int):
        self.n = n
    
    def domain(self) -> int:
        return self.n
    
    def codomain(self) -> int:
        return self.n
    
    def is_permutation(self) -> bool:
        return True
    
    def size(self) -> int:
        return 1
    
    def __str__(self) -> str:
        if self.n == 0:
            return "id0"
        return f"id_{self.n}"
    
    def to_tptp(self, prefix: str = "") -> str:
        if self.n == 0:
            return "id_0"
        return f"id_{self.n}"
    
    def to_smtlib(self, prefix: str = "") -> str:
        if self.n == 0:
            return "id0"
        return f"id{self.n}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Identity):
            return False
        return self.n == other.n


class Swap(Circuit):
    """Swap circuit sigma_{n,m}"""
    
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
    
    def domain(self) -> int:
        return self.n + self.m
    
    def codomain(self) -> int:
        return self.n + self.m
    
    def is_permutation(self) -> bool:
        return True
    
    def size(self) -> int:
        return 1
    
    def __str__(self) -> str:
        return f"sigma_{self.n}_{self.m}"
    
    def to_tptp(self, prefix: str = "") -> str:
        return f"sigma_{self.n}_{self.m}"
    
    def to_smtlib(self, prefix: str = "") -> str:
        return f"sigma_{self.n}_{self.m}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Swap):
            return False
        return self.n == other.n and self.m == other.m


class GeneratorCircuit(Circuit):
    """A circuit consisting of a single generator"""
    
    def __init__(self, generator: Generator):
        self.generator = generator
    
    def domain(self) -> int:
        return self.generator.domain
    
    def codomain(self) -> int:
        return self.generator.codomain
    
    def is_permutation(self) -> bool:
        return False
    
    def size(self) -> int:
        return 1
    
    def __str__(self) -> str:
        return f"gen_{self.generator.name}_{self.generator.domain}_{self.generator.codomain}"
    
    def to_tptp(self, prefix: str = "") -> str:
        return f"{prefix}gen_{self.generator.name}_{self.generator.domain}_{self.generator.codomain}"
    
    def to_smtlib(self, prefix: str = "") -> str:
        return f"{prefix}gen_{self.generator.name}_{self.generator.domain}_{self.generator.codomain}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, GeneratorCircuit):
            return False
        return self.generator == other.generator


class SequentialComposition(Circuit):
    """Sequential composition C1 ⨟ C2"""
    
    def __init__(self, c1: Circuit, c2: Circuit):
        # Fix mismatched domains by adding identities
        self.c1 = c1
        self.c2 = c2
        
        # Ensure compatibility for sequential composition
        if c1.codomain() != c2.domain():
            if c1.codomain() < c2.domain():
                # Add identity to match sizes
                diff = c2.domain() - c1.codomain()
                self.c1 = ParallelComposition(c1, Identity(diff))
            else:
                # Add identity to match sizes
                diff = c1.codomain() - c2.domain()
                self.c2 = SequentialComposition(Identity(diff), c2)
    
    def domain(self) -> int:
        return self.c1.domain()
    
    def codomain(self) -> int:
        return self.c2.codomain()
    
    def is_permutation(self) -> bool:
        return self.c1.is_permutation() and self.c2.is_permutation()
    
    def size(self) -> int:
        return self.c1.size() + self.c2.size()
    
    def __str__(self) -> str:
        return f"seq({self.c1}, {self.c2})"
    
    def to_tptp(self, prefix: str = "") -> str:
        return f"seq({self.c1.to_tptp(prefix)}, {self.c2.to_tptp(prefix)})"
    
    def to_smtlib(self, prefix: str = "") -> str:
        return f"(seq {self.c1.to_smtlib(prefix)} {self.c2.to_smtlib(prefix)})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, SequentialComposition):
            return False
        return self.c1 == other.c1 and self.c2 == other.c2


class ParallelComposition(Circuit):
    """Parallel composition C1 ⊗ C2"""
    
    def __init__(self, c1: Circuit, c2: Circuit):
        self.c1 = c1
        self.c2 = c2
    
    def domain(self) -> int:
        return self.c1.domain() + self.c2.domain()
    
    def codomain(self) -> int:
        return self.c1.codomain() + self.c2.codomain()
    
    def is_permutation(self) -> bool:
        return self.c1.is_permutation() and self.c2.is_permutation()
    
    def size(self) -> int:
        return self.c1.size() + self.c2.size()
    
    def __str__(self) -> str:
        return f"par({self.c1}, {self.c2})"
    
    def to_tptp(self, prefix: str = "") -> str:
        return f"par({self.c1.to_tptp(prefix)}, {self.c2.to_tptp(prefix)})"
    
    def to_smtlib(self, prefix: str = "") -> str:
        return f"(par {self.c1.to_smtlib(prefix)} {self.c2.to_smtlib(prefix)})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, ParallelComposition):
            return False
        return self.c1 == other.c1 and self.c2 == other.c2


class CircuitEquivalenceProblem:
    """Represents a diagrammatic equivalence problem"""
    
    def __init__(self, c1: Circuit, c2: Circuit, problem_type: CircuitType, is_equivalent: bool, problem_id: int = 0):
        self.c1 = c1
        self.c2 = c2
        self.problem_type = problem_type
        self.is_equivalent = is_equivalent
        self.problem_id = problem_id
        self.name = self._generate_name()
    
    def _generate_name(self) -> str:
        """Generate a unique name for this problem"""
        return f"equiv_{self.problem_type.value}_{self.problem_id}"
    
    def to_tptp(self, prefix: str = "") -> str:
        """Convert to TPTP format"""
        if self.is_equivalent:
            return f"fof({self.name}, conjecture, {self.c1.to_tptp(prefix)} = {self.c2.to_tptp(prefix)})."
        else:
            return f"fof({self.name}, conjecture, {self.c1.to_tptp(prefix)} != {self.c2.to_tptp(prefix)})."
    
    def to_smtlib(self, prefix: str = "") -> str:
        """Convert to SMT-LIB format"""
        if self.is_equivalent:
            return f"(assert (= {self.c1.to_smtlib(prefix)} {self.c2.to_smtlib(prefix)}))"
        else:
            return f"(assert (not (= {self.c1.to_smtlib(prefix)} {self.c2.to_smtlib(prefix)})))"
    
    def __str__(self) -> str:
        return f"{self.name}: {self.c1} {'==' if self.is_equivalent else '!='} {self.c2}"


class CircuitGenerator:
    """Generator for circuit benchmarks - Fully Fixed Version"""
    
    def __init__(self, max_size: int = 8, seed: Optional[int] = None):
        self.max_size = max_size
        if seed is not None:
            random.seed(seed)
        self.counter = 0
        self.generators: List[Generator] = []
        self._setup_generators()
    
    def _setup_generators(self):
        """Setup standard generators for circuits"""
        self.generators.append(Generator("A", 1, 1))
        self.generators.append(Generator("B", 1, 1))
        self.generators.append(Generator("C", 1, 2))
        self.generators.append(Generator("D", 2, 1))
        self.generators.append(Generator("E", 1, 1))
        self.generators.append(Generator("F", 2, 2))
    
    def _ensure_circuit_size(self, circuit: Circuit) -> Circuit:
        """Ensure circuit has valid domain/codomain sizes"""
        if circuit.domain() <= 0 and circuit.codomain() <= 0:
            return Identity(1)
        if circuit.domain() <= 0:
            return Identity(circuit.codomain())
        if circuit.codomain() <= 0:
            return Identity(circuit.domain())
        return circuit
    
    def random_circuit(self, max_domain: int = None) -> Circuit:
        """Generate a random circuit with controlled size"""
        if max_domain is None:
            max_domain = random.randint(1, min(4, self.max_size))
        
        # Base case: simple circuit
        if max_domain <= 1:
            return self._random_simple_circuit()
        
        # Randomly choose construction method
        choice = random.random()
        
        if choice < 0.3:
            # Simple circuit
            return self._random_simple_circuit()
        elif choice < 0.6:
            # Sequential composition
            split = random.randint(1, max_domain - 1)
            c1 = self.random_circuit(split)
            c2 = self.random_circuit(max_domain - split)
            if c1.codomain() == c2.domain():
                return SequentialComposition(c1, c2)
            else:
                # Try parallel composition instead
                return self._random_simple_circuit()
        else:
            # Parallel composition
            n1 = random.randint(1, max_domain - 1)
            n2 = max_domain - n1
            c1 = self.random_circuit(n1)
            c2 = self.random_circuit(n2)
            return ParallelComposition(c1, c2)
    
    def _random_simple_circuit(self) -> Circuit:
        """Generate a simple circuit"""
        choices = []
        
        # Add generators
        for gen in self.generators:
            if gen.domain <= 4 and gen.codomain <= 4:
                choices.append(("generator", gen))
        
        # Add identity
        for n in range(1, min(5, self.max_size) + 1):
            choices.append(("identity", n))
        
        # Add swaps
        for n in range(1, min(4, self.max_size) + 1):
            for m in range(1, min(4, self.max_size - n) + 1):
                choices.append(("swap", (n, m)))
        
        if not choices:
            return Identity(1)
        
        # Choose a circuit type randomly
        choice_type, value = random.choice(choices)
        
        if choice_type == "generator":
            return GeneratorCircuit(value)
        elif choice_type == "identity":
            return Identity(value)
        elif choice_type == "swap":
            n, m = value
            return Swap(n, m)
        
        return Identity(1)
    
    def generate_permutation_circuit(self, max_domain: int = None) -> Circuit:
        """Generate a circuit that represents a permutation"""
        if max_domain is None:
            max_domain = random.randint(1, min(5, self.max_size))
        
        if max_domain <= 1:
            return Identity(max_domain)
        
        # Use identities and swaps
        if max_domain <= 3:
            n = random.randint(1, max_domain - 1)
            m = max_domain - n
            if n > 0 and m > 0:
                return Swap(n, m)
            else:
                return Identity(max_domain)
        
        # Split into parallel composition
        n1 = random.randint(1, max_domain // 2)
        n2 = max_domain - n1
        c1 = self.generate_permutation_circuit(n1)
        c2 = self.generate_permutation_circuit(n2)
        return ParallelComposition(c1, c2)
    
    def generate_equivalent_pair(self, problem_type: CircuitType, complexity: int = 3) -> Tuple[Circuit, Circuit]:
        """Generate a pair of equivalent circuits"""
        if problem_type == CircuitType.GENERAL:
            return self._generate_equivalent_general(complexity)
        elif problem_type == CircuitType.PERMUTATION:
            return self._generate_equivalent_permutation(complexity)
        else:
            return self._generate_equivalent_restricted(complexity)
    
    def _generate_equivalent_general(self, complexity: int) -> Tuple[Circuit, Circuit]:
        """Generate equivalent general circuits"""
        # Start with a simple circuit
        c1 = self.random_circuit(complexity)
        c1 = self._ensure_circuit_size(c1)
        
        # Apply equivalence transformations to get c2
        c2 = self._apply_equivalence_transformations(c1)
        
        # If transformation failed or produced same circuit, create an alternative
        if c2 is None or c1 == c2:
            c2 = self._create_alternative_equivalent(c1)
        
        return c1, c2
    
    def _apply_equivalence_transformations(self, circuit: Circuit, depth: int = 0) -> Optional[Circuit]:
        """Apply equivalence transformations (coherence equations)"""
        if depth > 2:
            return circuit
        
        transformations = [
            self._apply_associativity,
            self._apply_swap_identity,
            self._apply_swap_inverse,
            self._apply_parallel_associativity
        ]
        
        # Apply 1-2 random transformations
        result = circuit
        for _ in range(random.randint(1, 2)):
            transformation = random.choice(transformations)
            result = transformation(result, depth + 1)
            if result is None:
                return circuit
        
        return result
    
    def _create_alternative_equivalent(self, circuit: Circuit) -> Circuit:
        """Create a different but equivalent circuit"""
        if isinstance(circuit, Identity):
            # id_n = id_{n-1} ⊗ id_1 for n > 1
            n = circuit.n
            if n > 1:
                return ParallelComposition(Identity(n - 1), Identity(1))
            return circuit
        
        if isinstance(circuit, Swap):
            # sigma_{n,m} = sigma_{m,n} ⨟ sigma_{n,m} ⨟ sigma_{m,n}
            return SequentialComposition(
                SequentialComposition(
                    Swap(circuit.m, circuit.n),
                    Swap(circuit.n, circuit.m)
                ),
                Swap(circuit.m, circuit.n)
            )
        
        if isinstance(circuit, SequentialComposition):
            # Try to add identity
            if circuit.c1.domain() > 0:
                return SequentialComposition(
                    SequentialComposition(circuit.c1, Identity(1)),
                    circuit.c2
                )
        
        if isinstance(circuit, ParallelComposition):
            # Swap the order in parallel
            return ParallelComposition(circuit.c2, circuit.c1)
        
        # Default: add identity composition
        return SequentialComposition(circuit, Identity(1))
    
    def _apply_associativity(self, circuit: Circuit, depth: int) -> Optional[Circuit]:
        """Apply associativity: (A ⨟ B) ⨟ C = A ⨟ (B ⨟ C)"""
        if isinstance(circuit, SequentialComposition):
            c1 = circuit.c1
            c2 = circuit.c2
            if isinstance(c1, SequentialComposition):
                return SequentialComposition(c1.c1, SequentialComposition(c1.c2, c2))
            elif isinstance(c2, SequentialComposition):
                return SequentialComposition(SequentialComposition(c1, c2.c1), c2.c2)
        return circuit
    
    def _apply_swap_identity(self, circuit: Circuit, depth: int) -> Optional[Circuit]:
        """Apply swap with identity: id_n ⊗ id_m = id_{n+m}"""
        if isinstance(circuit, ParallelComposition):
            if (isinstance(circuit.c1, Identity) and 
                isinstance(circuit.c2, Identity)):
                return Identity(circuit.c1.n + circuit.c2.n)
        return circuit
    
    def _apply_swap_inverse(self, circuit: Circuit, depth: int) -> Optional[Circuit]:
        """Apply swap inverse: σ_{m,n} ⨟ σ_{n,m} = id_{n+m}"""
        if isinstance(circuit, SequentialComposition):
            if (isinstance(circuit.c1, Swap) and 
                isinstance(circuit.c2, Swap)):
                s1, s2 = circuit.c1, circuit.c2
                if s1.n == s2.m and s1.m == s2.n:
                    return Identity(s1.n + s1.m)
        return circuit
    
    def _apply_parallel_associativity(self, circuit: Circuit, depth: int) -> Optional[Circuit]:
        """Apply parallel associativity: (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)"""
        if isinstance(circuit, ParallelComposition):
            c1 = circuit.c1
            c2 = circuit.c2
            if isinstance(c1, ParallelComposition):
                return ParallelComposition(c1.c1, ParallelComposition(c1.c2, c2))
            elif isinstance(c2, ParallelComposition):
                return ParallelComposition(ParallelComposition(c1, c2.c1), c2.c2)
        return circuit
    
    def _generate_equivalent_permutation(self, complexity: int) -> Tuple[Circuit, Circuit]:
        """Generate equivalent permutation circuits"""
        c1 = self.generate_permutation_circuit(complexity)
        c1 = self._ensure_circuit_size(c1)
        c2 = self._apply_permutation_equivalences(c1)
        if c2 is None or c1 == c2:
            c2 = self._create_alternative_equivalent(c1)
        return c1, c2
    
    def _apply_permutation_equivalences(self, circuit: Circuit) -> Optional[Circuit]:
        """Apply permutation-specific equivalences"""
        transformations = [
            self._apply_associativity,
            self._apply_swap_identity,
            self._apply_swap_inverse
        ]
        
        result = circuit
        for _ in range(random.randint(1, 2)):
            transformation = random.choice(transformations)
            result = transformation(result, 0)
            if result is None:
                return circuit
        
        return result
    
    def _generate_equivalent_restricted(self, complexity: int) -> Tuple[Circuit, Circuit]:
        """Generate equivalent restricted permutation circuits"""
        # Simple swap sequence
        c1 = self._generate_simple_swap_sequence(complexity)
        c1 = self._ensure_circuit_size(c1)
        c2 = self._apply_restricted_equivalences(c1)
        if c2 is None or c1 == c2:
            c2 = self._create_alternative_equivalent(c1)
        return c1, c2
    
    def _generate_simple_swap_sequence(self, complexity: int) -> Circuit:
        """Generate a simple sequence of swaps"""
        circuits = []
        total_size = 0
        
        for _ in range(min(complexity, 3)):
            if total_size >= 4:
                break
            n = random.randint(1, 2)
            m = random.randint(1, 2)
            circuits.append(Swap(n, m))
            total_size += n + m
        
        if not circuits:
            return Identity(1)
        
        # Compose sequentially
        result = circuits[0]
        for c in circuits[1:]:
            if result.codomain() == c.domain():
                result = SequentialComposition(result, c)
            else:
                # Adjust with identity
                diff = result.codomain() - c.domain()
                if diff > 0:
                    result = SequentialComposition(result, Identity(diff))
                    result = SequentialComposition(result, c)
                else:
                    result = SequentialComposition(Identity(-diff), c)
        
        return result
    
    def _apply_restricted_equivalences(self, circuit: Circuit) -> Optional[Circuit]:
        """Apply restricted equivalences"""
        result = self._apply_swap_identity(circuit, 0)
        if result is not None:
            result = self._apply_swap_inverse(result, 0)
            if result is not None:
                return result
        return circuit
    
    def generate_problem(self, problem_type: CircuitType, 
                         complexity: int = 3, 
                         is_equivalent: bool = True,
                         problem_id: int = 0) -> CircuitEquivalenceProblem:
        """Generate a complete equivalence problem"""
        if is_equivalent:
            c1, c2 = self.generate_equivalent_pair(problem_type, complexity)
        else:
            # Generate non-equivalent pair
            if problem_type == CircuitType.GENERAL:
                c1 = self.random_circuit(complexity)
                c2 = self.random_circuit(complexity)
                attempts = 0
                while c1 == c2 and attempts < 10:
                    c2 = self.random_circuit(complexity)
                    attempts += 1
            else:
                c1 = self.generate_permutation_circuit(complexity)
                c2 = self.generate_permutation_circuit(complexity)
                attempts = 0
                while c1 == c2 and attempts < 10:
                    c2 = self.generate_permutation_circuit(complexity)
                    attempts += 1
        
        return CircuitEquivalenceProblem(c1, c2, problem_type, is_equivalent, problem_id)
    
    def generate_benchmark_suite(self, num_problems: int = 8, 
                                 min_complexity: int = 2,
                                 max_complexity: int = 6,
                                 include_non_equivalent: bool = True) -> List[CircuitEquivalenceProblem]:
        """Generate a benchmark suite of problems"""
        problems = []
        problem_id = 0
        
        for problem_type in CircuitType:
            for complexity in range(min_complexity, max_complexity + 1, 2):
                # Equivalent problems
                for _ in range(num_problems // 3):
                    problem = self.generate_problem(problem_type, complexity, True, problem_id)
                    problems.append(problem)
                    problem_id += 1
                
                if include_non_equivalent:
                    # Non-equivalent problems
                    for _ in range(num_problems // 6):
                        problem = self.generate_problem(problem_type, complexity, False, problem_id)
                        problems.append(problem)
                        problem_id += 1
        
        return problems


class BenchmarkExporter:
    """Exports benchmarks to various formats"""
    
    @staticmethod
    def export_tptp(problems: List[CircuitEquivalenceProblem], filename: str):
        """Export problems to TPTP format"""
        with open(filename, 'w') as f:
            f.write("%% TPTP Benchmark for Circuit Equivalence\n")
            f.write("%% Generated by CircuitEquivalenceBenchmark\n\n")
            
            # Write axioms for circuit theory
            f.write("%% Axioms for circuit theory\n")
            f.write("fof(associativity, axiom, ! [A,B,C] : seq(seq(A,B),C) = seq(A,seq(B,C))).\n")
            f.write("fof(parallel_associativity, axiom, ! [A,B,C] : par(par(A,B),C) = par(A,par(B,C))).\n")
            f.write("fof(swap_identity, axiom, ! [N,M] : seq(sigma(N,M), sigma(M,N)) = id(N+M)).\n")
            f.write("fof(swap_commutation, axiom, ! [N,M,K] : seq(sigma(N+M,K), par(sigma(N,M), id(K))) = seq(par(id(N), sigma(M,K)), sigma(N, M+K))).\n\n")
            
            # Write each problem
            for i, problem in enumerate(problems):
                f.write(f"%% Problem {i+1}: {problem.name}\n")
                f.write(problem.to_tptp() + "\n\n")
    
    @staticmethod
    def export_smtlib(problems: List[CircuitEquivalenceProblem], filename: str):
        """Export problems to SMT-LIB format"""
        with open(filename, 'w') as f:
            f.write("; SMT-LIB Benchmark for Circuit Equivalence\n")
            f.write("; Generated by CircuitEquivalenceBenchmark\n\n")
            
            f.write("(set-logic QF_UFLIA)\n")
            f.write("(set-option :produce-models true)\n\n")
            
            # Declare sorts and functions
            f.write("; Circuit sorts and functions\n")
            f.write("(declare-sort Circuit 0)\n")
            f.write("(declare-fun id (Int) Circuit)\n")
            f.write("(declare-fun seq (Circuit Circuit) Circuit)\n")
            f.write("(declare-fun par (Circuit Circuit) Circuit)\n")
            f.write("(declare-fun sigma (Int Int) Circuit)\n\n")
            
            # Write axioms
            f.write("; Circuit axioms\n")
            f.write("(assert (forall ((A Circuit) (B Circuit) (C Circuit))\n")
            f.write("    (= (seq (seq A B) C) (seq A (seq B C)))))\n")
            f.write("(assert (forall ((A Circuit) (B Circuit) (C Circuit))\n")
            f.write("    (= (par (par A B) C) (par A (par B C)))))\n")
            f.write("(assert (forall ((N Int) (M Int))\n")
            f.write("    (= (seq (sigma N M) (sigma M N)) (id (+ N M)))))\n\n")
            
            # Write each problem
            for i, problem in enumerate(problems):
                f.write(f"; Problem {i+1}: {problem.name}\n")
                f.write(f"(assert (not (= {problem.c1.to_smtlib()} {problem.c2.to_smtlib()})))\n")
                f.write("(check-sat)\n\n")
    
    @staticmethod
    def export_python_code(problems: List[CircuitEquivalenceProblem], filename: str):
        """Export problems as Python code for verification"""
        with open(filename, 'w') as f:
            f.write("# Python verification code for circuit equivalence benchmarks\n")
            f.write("# Generated by CircuitEquivalenceBenchmark\n\n")
            
            f.write("class CircuitVerifier:\n")
            f.write("    def __init__(self):\n")
            f.write("        self.equivalence_rules = []\n\n")
            
            f.write("    def verify_equivalence(self, c1, c2) -> bool:\n")
            f.write('        """Verify if two circuits are equivalent"""\n')
            f.write("        # Placeholder for verification logic\n")
            f.write("        return True\n\n")
            
            # Write test cases
            for i, problem in enumerate(problems):
                f.write(f"def test_problem_{i}(verifier):\n")
                f.write(f'    """Test {problem.name}"""\n')
                f.write(f"    # Expected result: {problem.is_equivalent}\n")
                f.write(f"    # Circuit 1: {problem.c1}\n")
                f.write(f"    # Circuit 2: {problem.c2}\n")
                f.write(f"    return verifier.verify_equivalence({problem.c1}, {problem.c2})\n\n")


def main():
    """Main function to demonstrate benchmark generation"""
    print("Circuit Equivalence Benchmark Generator - Fully Fixed Version")
    print("=" * 50)
    
    # Create generator with smaller max size for safety
    generator = CircuitGenerator(max_size=5, seed=42)
    
    # Generate benchmark suite
    print("Generating benchmark problems...")
    try:
        problems = generator.generate_benchmark_suite(
            num_problems=6,
            min_complexity=2,
            max_complexity=4,  # Reduced for stability
            include_non_equivalent=True
        )
        
        print(f"Generated {len(problems)} problems")
        
        # Count problem types
        type_counts = {}
        for p in problems:
            type_counts[p.problem_type] = type_counts.get(p.problem_type, 0) + 1
        
        print("\nProblem distribution:")
        for pt, count in type_counts.items():
            equiv_count = sum(1 for p in problems if p.problem_type == pt and p.is_equivalent)
            non_equiv_count = count - equiv_count
            print(f"  {pt.value}: {count} total ({equiv_count} equivalent, {non_equiv_count} non-equivalent)")
        
        # Export to various formats
        print("\nExporting benchmarks...")
        
        exporter = BenchmarkExporter()
        exporter.export_tptp(problems, "circuit_benchmarks_fixed.tptp")
        exporter.export_smtlib(problems, "circuit_benchmarks_fixed.smt2")
        exporter.export_python_code(problems, "circuit_verification_fixed.py")
        
        print("✓ Exported TPTP benchmark to circuit_benchmarks_fixed.tptp")
        print("✓ Exported SMT-LIB benchmark to circuit_benchmarks_fixed.smt2")
        print("✓ Exported Python verification code to circuit_verification_fixed.py")
        
        # Print sample problems
        if problems:
            print("\nSample problems:")
            for i, problem in enumerate(problems[:3]):
                print(f"  {i+1}. {problem}")
                
    except Exception as e:
        print(f"Error generating benchmarks: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()