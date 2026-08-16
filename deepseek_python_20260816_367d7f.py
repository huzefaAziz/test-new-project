import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# PART 1: EL++ Ontology Representation
# =============================================================================

class Concept:
    """Represents an EL++ concept."""
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return self.name
    
    def __eq__(self, other):
        return isinstance(other, Concept) and self.name == other.name
    
    def __hash__(self):
        return hash(self.name)

class Role:
    """Represents an EL++ role."""
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return self.name
    
    def __eq__(self, other):
        return isinstance(other, Role) and self.name == other.name
    
    def __hash__(self):
        return hash(self.name)

class ExistentialRestriction:
    """Represents ∃R.C existential restriction."""
    def __init__(self, role: Role, concept: Concept):
        self.role = role
        self.concept = concept
    
    def __repr__(self):
        return f"∃{self.role}.{self.concept}"
    
    def __eq__(self, other):
        return (isinstance(other, ExistentialRestriction) and 
                self.role == other.role and self.concept == other.concept)
    
    def __hash__(self):
        return hash((self.role, self.concept))

@dataclass
class GCI:
    """General Concept Inclusion: C ⊑ D."""
    lhs: Any
    rhs: Any
    
    def __repr__(self):
        return f"{self.lhs} ⊑ {self.rhs}"

@dataclass
class RoleChain:
    """Role chain: R1 ∘ R2 ⊑ S."""
    roles: List[Role]
    super_role: Role

class Ontology:
    """EL++ ontology."""
    def __init__(self):
        self.gcis: List[GCI] = []
        self.role_chains: List[RoleChain] = []
        self.concepts: Set[Concept] = set()
        self.roles: Set[Role] = set()
    
    def add_gci(self, lhs, rhs):
        self.gcis.append(GCI(lhs, rhs))
        self._collect_concepts(lhs)
        self._collect_concepts(rhs)
    
    def add_role_chain(self, roles: List[Role], super_role: Role):
        self.role_chains.append(RoleChain(roles, super_role))
        for r in roles:
            self.roles.add(r)
        self.roles.add(super_role)
    
    def _collect_concepts(self, expr):
        if isinstance(expr, Concept):
            self.concepts.add(expr)
        elif isinstance(expr, ExistentialRestriction):
            self.concepts.add(expr.concept)
            self.roles.add(expr.role)

# =============================================================================
# PART 2: ELK-style Saturation
# =============================================================================

class ELKSaturator:
    """ELK-style saturation for EL++ ontologies."""
    
    def __init__(self):
        self.subsumptions: Set[Tuple] = set()
        self.role_links: Set[Tuple] = set()
        
    def saturate(self, ontology: Ontology):
        """Run ELK completion rules to saturation."""
        # Initialize with ontology axioms
        for gci in ontology.gcis:
            self._add_subsumption(gci.lhs, gci.rhs)
        
        # Apply completion rules until fixpoint
        changed = True
        while changed:
            changed = False
            
            # Reflexivity
            for c in ontology.concepts:
                if self._add_subsumption(c, c):
                    changed = True
            
            # Transitivity
            for c, d in list(self.subsumptions):
                for e, d2 in list(self.subsumptions):
                    if d == d2:
                        if self._add_subsumption(c, e):
                            changed = True
            
            # Role links from existential restrictions
            for c, d in list(self.subsumptions):
                for role in ontology.roles:
                    if self._add_role_link(c, role, d):
                        changed = True
            
            # Existential decomposition
            for c, role, d in list(self.role_links):
                for d2, e in list(self.subsumptions):
                    if d == d2:
                        if self._add_role_link(c, role, e):
                            changed = True
        
        return self.subsumptions, self.role_links
    
    def _add_subsumption(self, lhs, rhs) -> bool:
        key = (self._normalize(lhs), self._normalize(rhs))
        if key not in self.subsumptions:
            self.subsumptions.add(key)
            return True
        return False
    
    def _add_role_link(self, lhs, role, rhs) -> bool:
        key = (self._normalize(lhs), role, self._normalize(rhs))
        if key not in self.role_links:
            self.role_links.add(key)
            return True
        return False
    
    def _normalize(self, expr):
        return expr

# =============================================================================
# PART 3: Simplified SDD
# =============================================================================

class SDDNodeType(Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    DECISION = "DECISION"

@dataclass
class SDDNode:
    node_type: SDDNodeType
    var: Optional[int] = None
    primes: List['SDDNode'] = field(default_factory=list)
    subs: List['SDDNode'] = field(default_factory=list)
    value: Optional[bool] = None

class SDDCompiler:
    """Simplified SDD compiler."""
    
    def compile_to_sdd(self, clauses: List[List[Tuple[int, bool]]]) -> SDDNode:
        """Compile CNF to SDD using Shannon expansion."""
        if not clauses:
            return SDDNode(node_type=SDDNodeType.TRUE, value=True)
        
        # Get all variables
        vars_set = set()
        for clause in clauses:
            for lit, _ in clause:
                vars_set.add(lit)
        
        if not vars_set:
            return self._evaluate_clauses(clauses)
        
        var_order = sorted(vars_set)
        memo = {}
        return self._build_sdd(clauses, var_order, 0, memo)
    
    def _build_sdd(self, clauses, var_order, idx, memo):
        if idx >= len(var_order):
            return self._evaluate_clauses(clauses)
        
        # Simple memoization
        key = tuple(sorted([lit for clause in clauses for lit, _ in clause]))
        if key in memo:
            return memo[key]
        
        var = var_order[idx]
        
        # Split clauses
        clauses_true = []
        clauses_false = []
        
        for clause in clauses:
            clause_true = []
            clause_false = []
            rest = []
            
            for lit, sign in clause:
                if lit == var:
                    if sign:
                        clause_true.append((lit, sign))
                    else:
                        clause_false.append((lit, sign))
                else:
                    rest.append((lit, sign))
            
            if not rest and not clause_true and not clause_false:
                # Empty clause
                clauses_true.append([])
                clauses_false.append([])
            elif rest:
                clauses_true.append(rest)
                clauses_false.append(rest)
            else:
                if clause_true:
                    clauses_true.append(clause_true)
                if clause_false:
                    clauses_false.append(clause_false)
        
        # Ensure non-empty
        if not clauses_true:
            clauses_true = [[]]
        if not clauses_false:
            clauses_false = [[]]
        
        prime = self._build_sdd(clauses_true, var_order, idx + 1, memo)
        sub = self._build_sdd(clauses_false, var_order, idx + 1, memo)
        
        node = SDDNode(
            node_type=SDDNodeType.DECISION,
            var=var,
            primes=[prime],
            subs=[sub]
        )
        
        memo[key] = node
        return node
    
    def _evaluate_clauses(self, clauses):
        for clause in clauses:
            if not clause:
                return SDDNode(node_type=SDDNodeType.FALSE, value=False)
        return SDDNode(node_type=SDDNodeType.TRUE, value=True)

# =============================================================================
# PART 4: Weighted Model Counter
# =============================================================================

class WeightedModelCounter:
    def __init__(self):
        self.cache = {}
    
    def wmc(self, node: SDDNode, weights: Dict[int, float]) -> float:
        node_id = id(node)
        if node_id in self.cache:
            return self.cache[node_id]
        
        if node.node_type == SDDNodeType.TRUE:
            result = 1.0
        elif node.node_type == SDDNodeType.FALSE:
            result = 0.0
        elif node.node_type == SDDNodeType.DECISION:
            total = 0.0
            for prime, sub in zip(node.primes, node.subs):
                total += self.wmc(prime, weights) * self.wmc(sub, weights)
            result = total
        else:
            result = 0.0
        
        self.cache[node_id] = result
        return result
    
    def clear_cache(self):
        self.cache.clear()

# =============================================================================
# PART 5: Moose Framework (Fixed)
# =============================================================================

class Moose:
    def __init__(self, ontology: Ontology, observable_concepts: Set[Concept], 
                 latent_concepts: Set[Concept]):
        self.ontology = ontology
        self.observable_concepts = observable_concepts
        self.latent_concepts = latent_concepts
        
        # ELK saturation
        self.saturator = ELKSaturator()
        self.subsumptions, self.role_links = self.saturator.saturate(ontology)
        
        # SDD compiler
        self.sdd_compiler = SDDCompiler()
        self.sdd = None
        
        # Weighted model counter
        self.wmc_counter = WeightedModelCounter()
        
        # Variable mapping
        self.ground_atoms: Dict[Any, int] = {}
        self.atom_vars: Dict[int, Any] = {}
        self.var_counter = 0
    
    def ground_ontology(self, domain: List[str]):
        """Ground the ontology to the ABox domain."""
        for a in domain:
            # Concept atoms
            for c in self.ontology.concepts:
                atom = (c, a)
                var_id = self.var_counter
                self.var_counter += 1
                self.ground_atoms[atom] = var_id
                self.atom_vars[var_id] = atom
            
            # Role atoms
            for r in self.ontology.roles:
                for b in domain:
                    atom = (r, a, b)
                    var_id = self.var_counter
                    self.var_counter += 1
                    self.ground_atoms[atom] = var_id
                    self.atom_vars[var_id] = atom
    
    def compile_to_sdd(self, domain: List[str], add_closure: bool = True):
        """Compile the grounded ontology to an SDD."""
        self.ground_ontology(domain)
        
        # Extract clauses
        clauses = self._extract_clauses(domain)
        
        # Add closure axioms
        if add_closure:
            clauses.extend(self._add_closure_axioms(domain))
        
        # Compile to SDD
        self.sdd = self.sdd_compiler.compile_to_sdd(clauses)
        return self.sdd
    
    def _extract_clauses(self, domain: List[str]) -> List[List[Tuple[int, bool]]]:
        """Extract propositional clauses from saturation."""
        clauses = []
        
        # Subsumption clauses
        for c, d in self.subsumptions:
            if isinstance(c, Concept) and isinstance(d, Concept):
                for a in domain:
                    c_var = self.ground_atoms.get((c, a))
                    d_var = self.ground_atoms.get((d, a))
                    if c_var is not None and d_var is not None:
                        clauses.append([(c_var, False), (d_var, True)])
        
        # Role link clauses
        for c, role, d in self.role_links:
            if isinstance(c, Concept) and isinstance(d, Concept):
                for a in domain:
                    c_var = self.ground_atoms.get((c, a))
                    for b in domain:
                        r_var = self.ground_atoms.get((role, a, b))
                        d_var = self.ground_atoms.get((d, b))
                        if c_var is not None and r_var is not None and d_var is not None:
                            clauses.append([(c_var, False), (r_var, False), (d_var, True)])
        
        return clauses
    
    def _add_closure_axioms(self, domain: List[str]) -> List[List[Tuple[int, bool]]]:
        """Add closure axioms for exhaustive families."""
        clauses = []
        
        # Find digit concepts
        digit_concepts = [c for c in self.ontology.concepts 
                         if c.name.startswith("D") and len(c.name) <= 2]
        
        if digit_concepts:
            for a in domain:
                # Pairwise disjointness
                for i, di in enumerate(digit_concepts):
                    for dj in digit_concepts[i+1:]:
                        di_var = self.ground_atoms.get((di, a))
                        dj_var = self.ground_atoms.get((dj, a))
                        if di_var is not None and dj_var is not None:
                            clauses.append([(di_var, False), (dj_var, False)])
                
                # Covering
                digit_vars = []
                for d in digit_concepts:
                    var = self.ground_atoms.get((d, a))
                    if var is not None:
                        digit_vars.append(var)
                
                if digit_vars:
                    clauses.append([(var, True) for var in digit_vars])
        
        return clauses
    
    def compute_loss(self, perception_outputs: Dict[int, float], 
                     evidence: Dict[int, bool]) -> float:
        """Compute the negative log-likelihood loss."""
        if self.sdd is None:
            raise ValueError("SDD not compiled. Call compile_to_sdd() first.")
        
        # Build weights
        weights = {}
        for var_id, prob in perception_outputs.items():
            weights[var_id] = max(0.0, min(1.0, prob))
        
        # Clamp evidence
        for var_id, value in evidence.items():
            weights[var_id] = 1.0 if value else 0.0
        
        # Compute WMC
        self.wmc_counter.clear_cache()
        wmc_value = self.wmc_counter.wmc(self.sdd, weights)
        wmc_value = max(wmc_value, 1e-10)
        
        return -np.log(wmc_value)
    
    def get_var_id(self, atom) -> Optional[int]:
        """Safe method to get variable ID for an atom."""
        return self.ground_atoms.get(atom)

# =============================================================================
# PART 6: Simple CNN
# =============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_concepts: int):
        super().__init__()
        self.num_concepts = num_concepts
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_concepts),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return torch.sigmoid(x)

# =============================================================================
# PART 7: Main Example - Fixed
# =============================================================================

def create_mnist_ontology() -> Ontology:
    """Create the MNIST digit ontology."""
    ontology = Ontology()
    
    # Digit concepts
    digits = [Concept(f"D{i}") for i in range(10)]
    for d in digits:
        ontology.concepts.add(d)
    
    # Property concepts
    even = Concept("Even")
    odd = Concept("Odd")
    prime = Concept("Prime")
    composite = Concept("Composite")
    
    # Roles
    succ = Role("succ")
    plus_two = Role("plus_two")
    
    # Add roles
    ontology.roles.add(succ)
    ontology.roles.add(plus_two)
    
    # Digit to property GCIs
    digit_properties = {
        "D0": [even, composite],
        "D1": [odd],
        "D2": [even, prime],
        "D3": [odd, prime],
        "D4": [even, composite],
        "D5": [odd, prime],
        "D6": [even, composite],
        "D7": [odd, prime],
        "D8": [even, composite],
        "D9": [odd, composite],
    }
    
    for digit_name, props in digit_properties.items():
        digit = Concept(digit_name)
        for prop in props:
            ontology.add_gci(digit, prop)
    
    # Role chain
    ontology.add_role_chain([succ, succ], plus_two)
    
    # Digit existential restrictions
    for i in range(10):
        digit_i = Concept(f"D{i}")
        digit_next = Concept(f"D{(i+1) % 10}")
        ontology.add_gci(digit_i, ExistentialRestriction(succ, digit_next))
    
    return ontology

def main():
    print("Creating MNIST ontology...")
    ontology = create_mnist_ontology()
    
    # Define observable and latent signatures
    observable_concepts = {Concept("Even"), Concept("Odd"), Concept("Prime"), Concept("Composite")}
    latent_concepts = {Concept(f"D{i}") for i in range(10)}
    
    print(f"Ontology: {len(ontology.gcis)} GCIs, {len(ontology.role_chains)} role chains")
    print(f"Concepts: {len(ontology.concepts)}, Roles: {len(ontology.roles)}")
    
    # Create Moose instance
    moose = Moose(ontology, observable_concepts, latent_concepts)
    
    # Define domain
    domain = ["a"]
    
    # Compile to SDD
    print("\nCompiling to SDD...")
    moose.compile_to_sdd(domain, add_closure=True)
    print(f"SDD compiled successfully")
    print(f"Number of ground atoms: {len(moose.atom_vars)}")
    print(f"Number of clauses: {len(moose._extract_clauses(domain))}")
    
    # Create evidence generator with proper error handling
    def evidence_generator(labels, domain, moose_instance):
        """Generate evidence from digit labels with safe lookups."""
        evidence = {}
        digit = labels[0].item() if isinstance(labels, torch.Tensor) else labels
        
        # Map digit to properties
        props = {
            0: [("Even", True), ("Composite", True)],
            1: [("Odd", True)],
            2: [("Even", True), ("Prime", True)],
            3: [("Odd", True), ("Prime", True)],
            4: [("Even", True), ("Composite", True)],
            5: [("Odd", True), ("Prime", True)],
            6: [("Even", True), ("Composite", True)],
            7: [("Odd", True), ("Prime", True)],
            8: [("Even", True), ("Composite", True)],
            9: [("Odd", True), ("Composite", True)],
        }
        
        for prop_name, value in props.get(digit, []):
            prop_concept = Concept(prop_name)
            var_id = moose_instance.get_var_id((prop_concept, domain[0]))
            if var_id is not None:
                evidence[var_id] = value
            else:
                print(f"Warning: Could not find variable for {prop_name} in domain")
        
        return evidence
    
    # Test evidence generation
    print("\nTesting evidence generation:")
    for digit in [0, 3, 5, 7]:
        evidence = evidence_generator(digit, domain, moose)
        print(f"  Digit {digit}: {len(evidence)} evidence atoms")
        for var_id, value in evidence.items():
            if var_id in moose.atom_vars:
                atom = moose.atom_vars[var_id]
                print(f"    {atom} = {value}")
    
    # Create perception model
    num_atoms = len(moose.atom_vars)
    print(f"\nCreating perception model for {num_atoms} atoms...")
    perception_model = SimpleCNN(num_concepts=num_atoms)
    
    print("\nMoose ready for training!")
    print(f"  - {len(moose.atom_vars)} ground atoms")
    print(f"  - {len(latent_concepts)} latent concepts")
    print(f"  - {len(observable_concepts)} observable concepts")
    print(f"  - {len(moose.subsumptions)} subsumptions")
    print(f"  - {len(moose.role_links)} role links")

if __name__ == "__main__":
    main()