import itertools
from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Tuple, Any  # Added Any here

# === Core Data Structures based on the paper ===

@dataclass
class Atom:
    """Represents a predicate with arguments (variables/constants)."""
    predicate: str
    args: List[str]

    def __hash__(self) -> int:
        return hash((self.predicate, tuple(self.args)))

@dataclass
class Rule:
    """Represents a rule: body+ and body- -> head with existential vars."""
    id: str
    body_positive: List[Atom]
    body_negative: List[Atom]
    head: List[Atom]
    existential_vars: Set[str]  # Variables existentially quantified in head

    def is_datalog(self) -> bool:
        return not self.body_negative and not self.existential_vars

    def __hash__(self) -> int:
        return hash(self.id)

# Reliance relations from the paper
# Represented as sets of tuples (rho1, rho2)
PositiveReliance = Set[Tuple[Rule, Rule]]
NegativeReliance = Set[Tuple[Rule, Rule]]
RestraintReliance = Set[Tuple[Rule, Rule]]  # The "square" (☐) relation

@dataclass
class Chain:
    """Represents a decoupled chain of rule instances."""
    instances: List['Instance']  # Sequence of rule instances
    chain_rule: Optional[Rule] = None # The derived chain rule (rho_c)
    labels: Optional[Set[Any]] = None # For language-based cycle detection

    def __hash__(self) -> int:
        return hash(tuple(id(i) for i in self.instances))

# === The Core Chain Stratification Checker ===

class ChainStratificationChecker:
    """Implements the core logic for determining chain stratification."""
    
    def __init__(self, ruleset: List[Rule]):
        self.ruleset = ruleset
        self.positive_reliances: PositiveReliance = set()
        self.negative_reliances: NegativeReliance = set()
        self.restraint_reliances: RestraintReliance = set()
        
        # Store chains for the algorithm
        self.chains: Dict[Tuple[str, str], Set[Chain]] = {}
        self.seen_labels: Set[Any] = set() # For cycle detection in algorithm
        
        # Cache for computed reliances
        self._reliance_cache: Dict[Tuple[Rule, Rule], bool] = {}

    def compute_positive_reliance(self, rho1: Rule, rho2: Rule) -> bool:
        """Check if rho1 positively relies on rho2 (Definition 3)."""
        # This would require a chase simulation / homomorphism test.
        # For this conceptual implementation, we will simulate by checking
        # if head of rho1 unifies with a positive body atom of rho2.
        # This is a simplification for demonstration.
        for head_atom in rho1.head:
            for body_atom in rho2.body_positive:
                if head_atom.predicate == body_atom.predicate:
                    # A complete check would require mapping variables
                    return True
        return False

    def compute_negative_reliance(self, rho1: Rule, rho2: Rule) -> bool:
        """Check if rho1 negatively relies on rho2 (Definition 4)."""
        for head_atom in rho1.head:
            for neg_body_atom in rho2.body_negative:
                if head_atom.predicate == neg_body_atom.predicate:
                    return True
        return False

    def compute_restraint_reliance(self, rho1: Rule, rho2: Rule) -> bool:
        """Check if rho1 restrains rho2 (Definition 5)."""
        # rho1 creates an alternative match for rho2
        # Simplified check: rho1's head contains all existential variables
        # needed by rho2's head.
        if rho1.is_datalog(): # Datalog rules cannot restrain (by definition)
            return False
        # Check if rho1's head can "cover" a piece of rho2's head
        for head_atom1 in rho1.head:
            for head_atom2 in rho2.head:
                if head_atom1.predicate == head_atom2.predicate:
                    return True
        return False

    def build_chain_rule(self, chain: Chain) -> Optional[Rule]:
        """Build the chain rule rho_c (Equation 2 in the paper)."""
        instances = chain.instances
        if not instances:
            return None

        body_positive = []
        body_negative = []
        head = []
        existential_vars = set()

        # Add body+ from all instances
        for inst in instances:
            body_positive.extend(inst.rule.body_positive)
            body_negative.extend(inst.rule.body_negative)
            # All existential vars except from the last are "absorbed"
            if inst == instances[-1]:
                existential_vars = inst.rule.existential_vars
            else:
                # Head of all but last instance becomes part of body+
                body_positive.extend(inst.rule.head)

        # Head is the head of the last instance
        head = instances[-1].rule.head

        # Create a new rule with a unique ID
        rule_id = f"chain_{hash(tuple(id(i) for i in instances))}"
        return Rule(
            id=rule_id,
            body_positive=list(dict.fromkeys(body_positive)), # Remove duplicates
            body_negative=body_negative,
            head=head,
            existential_vars=existential_vars
        )

    def can_extend_chain(self, chain: Chain, rho: Rule) -> bool:
        """Check if chain can be extended by an instance of rule rho (Definition 9)."""
        # Create the chain rule
        chain_rule = self.build_chain_rule(chain)
        if not chain_rule:
            return False
        # Check if chain_rule positively relies on rho
        # Need to find an instance (a match) that makes this true
        # For this demo, we use a simple check
        # This corresponds to the condition: rho_c <+ rho (Definition 9)
        return self.compute_positive_reliance(chain_rule, rho)

    def is_chain(self, instance_sequence: List['Instance']) -> bool:
        """Check if a sequence is a decoupled chain (Definition 10)."""
        if len(instance_sequence) <= 1:
            return True  # Single instance is a chain
        
        # Check decoupling (Definition 11)
        stale_vars = set()
        for i in range(1, len(instance_sequence)):
            current_vars = set(instance_sequence[i].rule.body_positive[0].args)  # Simplified
            prev_head_vars = set()
            if i > 0:
                prev_head_vars = set(instance_sequence[i-1].rule.head[0].args)  # Simplified
            # Check if any variable in current instance's body is stale
            # A variable is stale if it appears before but not in the previous head
            for j in range(i):
                stale_vars.update(
                    set(instance_sequence[i].rule.body_positive[0].args) - prev_head_vars
                )
        
        # If stale vars exist, the sequence is not decoupled (but may be decoupled after renaming)
        # For this demonstration, we will not enforce decoupling strictly.
        
        # Recursively check if prefix is a chain and can be extended
        prefix = instance_sequence[:-1]
        if not self.is_chain(prefix):
            return False
        return self.can_extend_chain(Chain(prefix), instance_sequence[-1].rule)

    def compute_chain_stratification(self) -> bool:
        """
        Main algorithm for checking chain stratification (Algorithm 1 from the paper).
        Returns True if the ruleset is chain-stratified.
        """
        print("Computing reliances...")
        # Initialize reliances (would be computed more thoroughly)
        for rho1 in self.ruleset:
            for rho2 in self.ruleset:
                if self.compute_positive_reliance(rho1, rho2):
                    self.positive_reliances.add((rho1, rho2))
                if self.compute_negative_reliance(rho1, rho2):
                    self.negative_reliances.add((rho1, rho2))
                if self.compute_restraint_reliance(rho1, rho2):
                    self.restraint_reliances.add((rho1, rho2))

        print("Initializing chains...")
        # Initialize chains with all rules (single rule instances)
        # and their reliances (these form the initial chain rules)
        for rho in self.ruleset:
            chain = Chain(instances=[Instance(rho, {})]) # Dummy instance
            self.chains.setdefault((rho.id, rho.id), set()).add(chain)
        
        print("Iteratively extending chains...")
        # Main loop (Algorithm 1)
        changed = True
        iteration = 0
        while changed:
            iteration += 1
            changed = False
            print(f"Iteration {iteration}, checking {len(self.chains)} chains...")
            for (start_id, end_id), chain_set in list(self.chains.items()):
                for chain in list(chain_set):
                    # Compute the chain rule (rho_c)
                    chain_rule = self.build_chain_rule(chain)
                    if not chain_rule:
                        continue
                    
                    # Check extension by all rules in the ruleset
                    for rho in self.ruleset:
                        if self.can_extend_chain(chain, rho):
                            # Build the extended chain and its label
                            new_chain = Chain(chain.instances + [Instance(rho, {})])
                            new_chain.chain_rule = self.build_chain_rule(new_chain)
                            
                            # Check for cycles (Definition 15)
                            # Negative/restraint reliances from the chain rule
                            if self.compute_negative_reliance(chain_rule, rho) or \
                               self.compute_restraint_reliance(chain_rule, rho):
                                print(f"Cycle detected involving chain from {start_id} to {rho.id}")
                                # In a real implementation, this would check for cycles in
                                # the precedence relation defined by <_c^- and <_c^☐
                                # For this demo, we treat any such reliance as a potential cycle
                                return False
                            
                            # Add new chain to the set
                            if new_chain not in chain_set:
                                self.chains.setdefault((start_id, rho.id), set()).add(new_chain)
                                changed = True
        
        print("No cycles found. Ruleset is chain-stratified.")
        return True

# === Helper classes ===

class Instance:
    """Represents a rule instance (with a specific mapping of variables to constants/terms)."""
    def __init__(self, rule: Rule, mapping: Dict[str, str]):
        self.rule = rule
        self.mapping = mapping

    def __hash__(self) -> int:
        return hash((id(self.rule), tuple(sorted(self.mapping.items()))))

# === Example from the paper: Problem 2 (L8-L11) ===

def create_problem2_rules():
    """Creates the rules from Section 1, Problem 2."""
    # Predicates: :type, :father, :Man, :eq, :mef (simplified to strings)
    # We'll use atomic predicates for clarity
    
    # L8: {?x rdf:type :Human} => {?x :father _:l . _:l rdf:type :Man}
    r8 = Rule(
        id="L8",
        body_positive=[Atom("type", ["?x", "Human"])],
        body_negative=[],
        head=[Atom("father", ["?x", "_:l"]), Atom("type", ["_:l", "Man"])],
        existential_vars={"_:l"}
    )
    
    # L9: {?x :father ?y} => {?y rdf:type :Man}
    r9 = Rule(
        id="L9",
        body_positive=[Atom("father", ["?x", "?y"])],
        body_negative=[],
        head=[Atom("type", ["?y", "Man"])],
        existential_vars=set()
    )
    
    # L10: {?x :father ?y} => {?y :eq ?y}
    r10 = Rule(
        id="L10",
        body_positive=[Atom("father", ["?x", "?y"])],
        body_negative=[],
        head=[Atom("eq", ["?y", "?y"])],
        existential_vars=set()
    )
    
    # L11: {?x :father ?y1 . ?x :father ?y2 . not {?y1 :eq ?y2}} => {?y1 :mef ?y2}
    r11 = Rule(
        id="L11",
        body_positive=[
            Atom("father", ["?x", "?y1"]),
            Atom("father", ["?x", "?y2"])
        ],
        body_negative=[
            Atom("eq", ["?y1", "?y2"])
        ],
        head=[Atom("mef", ["?y1", "?y2"])],
        existential_vars=set()
    )
    
    return [r8, r9, r10, r11]

# === Main execution ===

if __name__ == "__main__":
    print("=== Chain Stratification Example from Paper ===")
    rules = create_problem2_rules()
    checker = ChainStratificationChecker(rules)
    
    # This would compute the reliances and check for chain stratification
    # In the paper, these rules are classified as NOT chain-stratified.
    result = checker.compute_chain_stratification()
    print(f"Ruleset is chain-stratified: {result}")
    print("Note: This is a simplified conceptual implementation. The actual algorithm")
    print("would require proper chase simulation, unification, and homomorphism checks.")