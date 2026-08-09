"""
Relational Frame AI v1.0
No gradients. No math. Pure relationship networks.
Based on Relational Frame Theory (RFT) — humans learn through relations.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Set, Tuple, Optional, List
import json
from enum import Enum

class RelationType(Enum):
    """Core relational frames — how we connect everything."""
    SAME = "same_as"
    OPPOSITE = "opposite_of"
    CAUSE = "causes"
    EFFECT = "effect_of"
    PART = "part_of"
    WHOLE = "contains"
    BEFORE = "before"
    AFTER = "after"
    HAS_PROPERTY = "has_property"
    INSTANCE = "instance_of"
    CATEGORY = "category_of"
    DEPENDS_ON = "depends_on"
    CONFLICTS_WITH = "conflicts_with"

@dataclass
class RelationalFrame:
    """A single relationship between two concepts."""
    source: str
    target: str
    relation: RelationType
    strength: float = 1.0  # Manual confidence (0-1)
    context: str = "default"  # When this relation applies
    evidence: List[str] = field(default_factory=list)  # Why we believe this

@dataclass
class Concept:
    """A node in the relational network."""
    name: str
    properties: Set[str] = field(default_factory=set)
    instances: Set[str] = field(default_factory=set)
    category: Optional[str] = None

class RelationalFrameAI:
    def __init__(self):
        # The entire knowledge graph
        self.frames: List[RelationalFrame] = []
        
        # Quick lookup: concept -> list of frames
        self.concept_index: defaultdict = defaultdict(list)
        
        # Concepts as objects (with properties)
        self.concepts: dict = {}
        
        # Curiosity — gaps in our relational network
        self.gaps: List[str] = []
        
        # Built-in relational logic
        self.inference_rules = {
            RelationType.SAME: self._infer_same,
            RelationType.CAUSE: self._infer_cause,
            RelationType.PART: self._infer_part,
            RelationType.BEFORE: self._infer_before,
        }
    
    def add_concept(self, name: str, category: str = None, properties: Set[str] = None):
        """Directly create a concept (no learning, just definition)."""
        if name not in self.concepts:
            self.concepts[name] = Concept(
                name=name,
                category=category,
                properties=properties or set()
            )
        return self.concepts[name]
    
    def relate(self, source: str, target: str, relation: RelationType, 
               strength: float = 1.0, context: str = "default", evidence: List[str] = None):
        """
        Create a relational frame between two concepts.
        You directly define how concepts relate.
        """
        # Ensure concepts exist
        self.add_concept(source)
        self.add_concept(target)
        
        # Create the frame
        frame = RelationalFrame(
            source=source,
            target=target,
            relation=relation,
            strength=strength,
            context=context,
            evidence=evidence or [f"Direct definition"]
        )
        
        self.frames.append(frame)
        self.concept_index[source].append(frame)
        self.concept_index[target].append(frame)
        
        # Also store inverse relation for symmetry
        if relation == RelationType.SAME:
            # Same is symmetric
            inverse = RelationalFrame(
                source=target,
                target=source,
                relation=RelationType.SAME,
                strength=strength,
                context=context,
                evidence=["Symmetry of SAME relation"]
            )
            self.frames.append(inverse)
            self.concept_index[target].append(inverse)
        
        elif relation == RelationType.OPPOSITE:
            # Opposite is symmetric too
            inverse = RelationalFrame(
                source=target,
                target=source,
                relation=RelationType.OPPOSITE,
                strength=strength,
                context=context,
                evidence=["Symmetry of OPPOSITE relation"]
            )
            self.frames.append(inverse)
            self.concept_index[target].append(inverse)
        
        elif relation == RelationType.CAUSE:
            # Effects are inverse of causes
            effect_frame = RelationalFrame(
                source=target,
                target=source,
                relation=RelationType.EFFECT,
                strength=strength * 0.8,  # Slightly weaker inverse
                context=context,
                evidence=["Inverse of cause"]
            )
            self.frames.append(effect_frame)
            self.concept_index[target].append(effect_frame)
        
        return frame
    
    def _infer_same(self, source: str, target: str) -> bool:
        """If A same as B, and B same as C, then A same as C (transitivity)."""
        # Find all SAME frames from target
        for frame in self.concept_index.get(target, []):
            if frame.relation == RelationType.SAME:
                if frame.target != source:  # Avoid infinite loop
                    # Create transitive relation
                    self.relate(source, frame.target, RelationType.SAME, 
                               strength=min(frame.strength, 0.9),
                               evidence=["Transitivity inference"])
                    return True
        return False
    
    def _infer_cause(self, source: str, target: str) -> bool:
        """If A causes B, and B causes C, then A causes C (transitive)."""
        for frame in self.concept_index.get(target, []):
            if frame.relation == RelationType.CAUSE:
                self.relate(source, frame.target, RelationType.CAUSE,
                           strength=min(frame.strength, 0.9),
                           evidence=["Transitive cause inference"])
                return True
        return False
    
    def _infer_part(self, source: str, target: str) -> bool:
        """If A is part of B, and B is part of C, then A is part of C."""
        for frame in self.concept_index.get(target, []):
            if frame.relation == RelationType.PART:
                self.relate(source, frame.target, RelationType.PART,
                           strength=min(frame.strength, 0.9),
                           evidence=["Transitive part inference"])
                return True
        return False
    
    def _infer_before(self, source: str, target: str) -> bool:
        """If A before B, and B before C, then A before C."""
        for frame in self.concept_index.get(target, []):
            if frame.relation == RelationType.BEFORE:
                self.relate(source, frame.target, RelationType.BEFORE,
                           strength=min(frame.strength, 0.9),
                           evidence=["Transitive temporal inference"])
                return True
        return False
    
    def reason(self, source: str, target: str, relation: RelationType) -> Tuple[bool, float, List[str]]:
        """
        Answer: Does source relate to target in this way?
        Returns: (True/False, confidence, evidence chain)
        """
        # Direct match
        for frame in self.concept_index.get(source, []):
            if frame.target == target and frame.relation == relation:
                return True, frame.strength, frame.evidence
        
        # Try inference
        if relation in self.inference_rules:
            if self.inference_rules[relation](source, target):
                # Check again after inference
                for frame in self.concept_index.get(source, []):
                    if frame.target == target and frame.relation == relation:
                        return True, frame.strength, frame.evidence + ["Inferred"]
        
        # Not found — mark as curiosity
        self.gaps.append(f"{source} -> {target} ({relation.value})")
        return False, 0.0, ["No relation found"]
    
    def query(self, source: str, relation: RelationType = None) -> List[str]:
        """Find all concepts related to source by a specific relation."""
        results = []
        for frame in self.concept_index.get(source, []):
            if relation is None or frame.relation == relation:
                results.append({
                    'target': frame.target,
                    'relation': frame.relation.value,
                    'strength': frame.strength,
                    'context': frame.context
                })
        return results
    
    def derive(self, concept: str) -> dict:
        """Show all known relations for a concept (your AI's understanding)."""
        frames = self.concept_index.get(concept, [])
        return {
            'relations': [
                {
                    'source': f.source,
                    'target': f.target,
                    'relation': f.relation.value,
                    'strength': f.strength
                } for f in frames
            ],
            'concept_info': self.concepts.get(concept, {})
        }
    
    def teach_chain(self, chain: List[Tuple[str, str, RelationType]]):
        """Teach a chain of relations at once."""
        for source, target, relation in chain:
            self.relate(source, target, relation)

# ===== DEMO =====

if __name__ == "__main__":
    ai = RelationalFrameAI()
    
    print("=== Building Relational Knowledge ===\n")
    
    # Create concepts directly
    ai.add_concept("Fire", properties={"hot", "bright"})
    ai.add_concept("Smoke", properties={"gray", "rises"})
    ai.add_concept("Heat", category="energy")
    ai.add_concept("Burn", category="action")
    ai.add_concept("Wood", properties={"brown", "dry"})
    ai.add_concept("Oxygen", properties={"invisible", "necessary"})
    ai.add_concept("Water", properties={"wet", "cold"})
    
    # Define relationships (YOU create these directly)
    ai.relate("Fire", "Smoke", RelationType.CAUSE, strength=0.95, 
              evidence=["Observation: Fire always produces smoke"])
    
    ai.relate("Fire", "Heat", RelationType.CAUSE, strength=1.0,
              evidence=["Definition: Fire generates heat"])
    
    ai.relate("Fire", "Wood", RelationType.DEPENDS_ON, strength=0.9,
              evidence=["Fire needs fuel"])
    
    ai.relate("Fire", "Oxygen", RelationType.DEPENDS_ON, strength=1.0,
              evidence=["Fire needs oxygen to burn"])
    
    ai.relate("Water", "Fire", RelationType.OPPOSITE, strength=0.95,
              evidence=["Water extinguishes fire"])
    
    ai.relate("Burn", "Fire", RelationType.CAUSE, strength=1.0,
              evidence=["Fire causes burning"])
    
    ai.relate("Wood", "Burn", RelationType.CAN_CAUSE, strength=0.7,
              evidence=["Wood can burn"])
    
    print("Knowledge built. Now reasoning:\n")
    
    # Query: What causes smoke?
    print("Q: What causes Smoke?")
    for frame in ai.concept_index.get("Smoke", []):
        if frame.relation == RelationType.EFFECT:
            print(f"  A: {frame.source} (confidence: {frame.strength})")
    
    print("\nQ: Is Fire opposite of Water?")
    result, conf, evidence = ai.reason("Fire", "Water", RelationType.OPPOSITE)
    print(f"  A: {result} (confidence: {conf})")
    print(f"  Evidence: {', '.join(evidence)}")
    
    print("\nQ: Does Fire depend on Oxygen?")
    result, conf, evidence = ai.reason("Fire", "Oxygen", RelationType.DEPENDS_ON)
    print(f"  A: {result} (confidence: {conf})")
    print(f"  Evidence: {', '.join(evidence)}")
    
    print("\nQ: Does Wood cause Fire? (Indirect)")
    result, conf, evidence = ai.reason("Wood", "Fire", RelationType.CAN_CAUSE)
    print(f"  A: {result} (confidence: {conf})")
    
    print("\n=== Chain Reasoning ===")
    # Teach a causal chain
    ai.teach_chain([
        ("Rain", "Clouds", RelationType.CAUSE),
        ("Clouds", "Water", RelationType.CONTAINS),
        ("Water", "Wet", RelationType.CAUSE)
    ])
    
    # Now reason across the chain
    result, conf, evidence = ai.reason("Rain", "Wet", RelationType.CAUSE)
    print(f"Rain causes Wet? {result} (confidence: {conf})")
    print(f"Reasoning: {evidence}")
    
    print("\n=== Full Knowledge Graph for Fire ===")
    print(json.dumps(ai.derive("Fire"), indent=2, default=str))
    
    print(f"\nCuriosities (gaps in knowledge): {ai.gaps[:3]}")