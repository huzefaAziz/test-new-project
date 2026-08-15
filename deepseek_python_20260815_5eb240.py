"""
A Triumvirate of AI Driven Theoretical Discovery
Complete Implementation based on arXiv:2405.19973v1
Author: Yang-Hui He (adapted implementation)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# For meta-mathematics / NLP components
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# For automated theorem proving (bottom-up)
from itertools import combinations, product
import sympy as sp
from sympy import symbols, Eq, solve, simplify, expand, factor
from sympy.logic.boolalg import to_cnf, to_dnf

import random
import json
import os
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("="*80)
print("AI DRIVEN THEORETICAL DISCOVERY - COMPLETE IMPLEMENTATION")
print("Based on arXiv:2405.19973v1 by Yang-Hui He")
print("="*80)

#=============================================================================
# PART 1: BOTTOM-UP MATHEMATICS (Automated Theorem Proving)
#=============================================================================

class AutomatedTheoremProver:
    """
    Implements bottom-up mathematics using symbolic computation
    and automated reasoning
    """
    
    def __init__(self):
        self.theorems = []
        self.axioms = []
        self.known_facts = set()
        self.proof_history = []
        
    def add_axiom(self, axiom):
        """Add a foundational axiom"""
        self.axioms.append(axiom)
        self.known_facts.add(str(axiom))
        print(f"Added axiom: {axiom}")
        
    def define_theorem(self, name, proposition):
        """Define a theorem to be proven"""
        theorem = {
            'name': name,
            'proposition': proposition,
            'proven': False,
            'proof': None
        }
        self.theorems.append(theorem)
        print(f"Defined theorem: {name} = {proposition}")
        return theorem
    
    def prove_by_symbolic_manipulation(self, theorem_name, target_expression):
        """Attempt to prove a theorem using symbolic manipulation"""
        print(f"\nAttempting to prove: {theorem_name}")
        print(f"Target: {target_expression}")
        
        # Convert to sympy expression
        try:
            expr = sp.sympify(target_expression)
            print(f"Parsed expression: {expr}")
            
            # Try various proof strategies
            strategies = [
                ('simplify', lambda x: sp.simplify(x)),
                ('expand', lambda x: sp.expand(x)),
                ('factor', lambda x: sp.factor(x)),
                ('collect', lambda x: sp.collect(x, sp.symbols('x'))),
            ]
            
            proof_steps = []
            current_expr = expr
            
            for name, strategy in strategies:
                try:
                    result = strategy(current_expr)
                    if result != current_expr:
                        proof_steps.append({
                            'strategy': name,
                            'before': str(current_expr),
                            'after': str(result)
                        })
                        current_expr = result
                except:
                    continue
            
            # Check if we've simplified to zero or a known constant
            if current_expr == 0 or current_expr == 1 or current_expr == True:
                proof_steps.append({
                    'strategy': 'identity',
                    'before': str(current_expr),
                    'after': 'proved'
                })
                
                # Record proof
                for theorem in self.theorems:
                    if theorem['name'] == theorem_name:
                        theorem['proven'] = True
                        theorem['proof'] = proof_steps
                        self.proof_history.append({
                            'theorem': theorem_name,
                            'proof': proof_steps,
                            'timestamp': datetime.now()
                        })
                        print(f"✓ Theorem {theorem_name} proven!")
                        return True
            
            print(f"Could not prove {theorem_name} automatically")
            return False
            
        except Exception as e:
            print(f"Error during proof attempt: {e}")
            return False
    
    def generate_corollaries(self, theorem_name):
        """Generate corollaries from proven theorems"""
        print(f"\nGenerating corollaries from {theorem_name}")
        
        # Find the theorem
        theorem = None
        for t in self.theorems:
            if t['name'] == theorem_name and t['proven']:
                theorem = t
                break
        
        if not theorem:
            print(f"Theorem {theorem_name} not found or not proven")
            return []
        
        corollaries = []
        # Simple corollary generation
        if '=' in str(theorem['proposition']):
            # Try to derive variations
            base = str(theorem['proposition'])
            # Add trivial variations
            variations = [
                f"({base}) → True",
                f"not ({base}) → False",
                f"({base}) and True",
                f"({base}) or False",
            ]
            
            for i, var in enumerate(variations[:2]):  # Limit to avoid explosion
                corollary_name = f"{theorem_name}_corollary_{i+1}"
                corollary = self.define_theorem(corollary_name, var)
                corollaries.append(corollary_name)
                print(f"Generated corollary: {corollary_name} = {var}")
        
        return corollaries
    
    def knowledge_based_reasoning(self, question):
        """Answer questions using the knowledge base"""
        print(f"\nKnowledge-based reasoning: {question}")
        
        # Simple pattern matching
        if 'prime' in question.lower():
            return "Prime numbers are integers greater than 1 that have no positive divisors other than 1 and themselves."
        elif 'theorem' in question.lower():
            proven = [t['name'] for t in self.theorems if t['proven']]
            if proven:
                return f"Proven theorems: {', '.join(proven)}"
            return "No theorems proven yet."
        elif 'axiom' in question.lower():
            return f"Axioms: {[str(a) for a in self.axioms]}"
        else:
            return "I need more information to answer that question."

#=============================================================================
# PART 2: META-MATHEMATICS (Mathematics as Language)
#=============================================================================

class MetaMathematics:
    """
    Implements meta-mathematics approach using NLP and LLM-like techniques
    """
    
    def __init__(self):
        self.corpus = []
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.word_embeddings = None
        self.pca_model = None
        self.vocabulary = set()
        
    def load_mathematical_corpus(self, texts):
        """Load mathematical texts for analysis"""
        self.corpus = texts
        self._build_vocabulary()
        print(f"Loaded corpus with {len(texts)} documents")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        
    def _build_vocabulary(self):
        """Build vocabulary from corpus"""
        all_words = []
        for text in self.corpus:
            words = re.findall(r'\w+', text.lower())
            all_words.extend(words)
        self.vocabulary = set(all_words)
        
    def generate_word_embeddings(self):
        """Generate word embeddings using TF-IDF and SVD"""
        print("\nGenerating word embeddings...")
        
        # Create document-term matrix
        tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        self.word_embeddings = tfidf_matrix.toarray()
        
        print(f"Embeddings shape: {self.word_embeddings.shape}")
        return self.word_embeddings
    
    def find_mathematical_patterns(self):
        """Find patterns in mathematical texts"""
        print("\nFinding mathematical patterns...")
        
        patterns = {}
        
        # Look for common mathematical phrases
        phrase_patterns = [
            (r'theorem\s+\d+', 'theorem_patterns'),
            (r'proof\s+of\s+([a-zA-Z\s]+)', 'proof_subjects'),
            (r'if\s+([a-zA-Z\s]+)\s+then\s+([a-zA-Z\s]+)', 'conditional_statements'),
            (r'let\s+([a-zA-Z\s]+)\s+be\s+([a-zA-Z\s]+)', 'definitions'),
        ]
        
        for text in self.corpus:
            for pattern, key in phrase_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if key not in patterns:
                        patterns[key] = []
                    patterns[key].extend(matches)
        
        return patterns
    
    def analyze_arxiv_titles(self):
        """Analyze arXiv titles - similar to the paper's experiment"""
        print("\nAnalyzing mathematical paper titles...")
        
        # Sample titles - representative from paper
        titles = [
            "String model building, reinforcement learning and genetic algorithms",
            "Ising machines for Diophantine problems in physics",
            "Machine learning meets number theory: The data science of Birch-Swinnerton-Dyer",
            "Estimating Calabi-Yau hypersurface and triangulation counts with equation learners",
            "Machine learning Calabi-Yau metrics",
            "Hilbert series, machine learning, and applications to physics",
            "Polytopes and machine learning",
            "Quiver mutations, Seiberg duality and machine learning",
            "Machine learning CICY threefolds",
            "Machine learning in the string landscape",
        ]
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(titles)
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features.toarray())
        
        print("Title analysis complete")
        return {
            'titles': titles,
            'features_2d': features_2d,
            'vocabulary': vectorizer.get_feature_names_out()
        }
    
    def generate_mathematical_text(self, seed_text, length=20):
        """Generate mathematical text using Markov chain"""
        print(f"\nGenerating mathematical text from seed: '{seed_text}'")
        
        if not self.corpus:
            print("No corpus loaded for generation")
            return None
        
        # Simple Markov chain for demonstration
        words = ' '.join(self.corpus).split()
        if len(words) < 10:
            print("Corpus too small for meaningful generation")
            return None
        
        # Build transition matrix
        transitions = {}
        for i in range(len(words) - 1):
            current = words[i]
            next_word = words[i + 1]
            if current not in transitions:
                transitions[current] = []
            transitions[current].append(next_word)
        
        # Generate text
        seed_words = seed_text.split()
        if not seed_words or seed_words[-1] not in transitions:
            # Start with a random word from vocabulary
            current = random.choice(list(transitions.keys()))
        else:
            current = seed_words[-1]
        
        generated = list(seed_words)
        for _ in range(length):
            if current in transitions and transitions[current]:
                next_word = random.choice(transitions[current])
                generated.append(next_word)
                current = next_word
            else:
                break
        
        return ' '.join(generated)

#=============================================================================
# PART 3: TOP-DOWN MATHEMATICS (Pattern Recognition & Conjecture Formulation)
#=============================================================================

class TopDownMathematics:
    """
    Implements top-down mathematics using pattern recognition
    and machine learning for conjecture formulation
    """
    
    def __init__(self):
        self.models = {}
        self.datasets = {}
        self.conjectures = []
        self.patterns_found = []
        
    def generate_binary_sequence_problem(self, sequence_type='divisibility'):
        """
        Generate the binary sequence problem from the paper
        Examples: divisibility by 3, primality, Möbius function
        """
        n_samples = 1000
        sequence = []
        
        if sequence_type == 'divisibility_by_3':
            # Case (i): whether n divisible by 3
            for n in range(1, n_samples + 1):
                sequence.append(1 if n % 3 == 0 else 0)
            description = "Divisibility by 3 sequence"
            
        elif sequence_type == 'primality':
            # Case (ii): whether n is prime
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(np.sqrt(n)) + 1):
                    if n % i == 0:
                        return False
                return True
            
            for n in range(1, n_samples + 1):
                sequence.append(1 if is_prime(n) else 0)
            description = "Primality sequence"
            
        elif sequence_type == 'mobius':
            # Case (iii): parity of number of prime factors
            def mobius_parity(n):
                # Simplified: 1 if odd number of prime factors, 0 if even
                factors = []
                temp = n
                p = 2
                while p * p <= temp:
                    while temp % p == 0:
                        factors.append(p)
                        temp //= p
                    p += 1 if p == 2 else 2
                if temp > 1:
                    factors.append(temp)
                return len(factors) % 2
            
            for n in range(1, n_samples + 1):
                sequence.append(mobius_parity(n))
            description = "Möbius parity sequence"
            
        else:
            raise ValueError(f"Unknown sequence type: {sequence_type}")
        
        return sequence, description
    
    def create_sliding_windows(self, sequence, window_size=20, stride=1):
        """
        Create sliding windows for machine learning
        As described in Eq (2) of the paper
        """
        X = []
        y = []
        
        for i in range(0, len(sequence) - window_size - 1, stride):
            window = sequence[i:i + window_size]
            next_val = sequence[i + window_size]
            X.append(window)
            y.append(next_val)
        
        return np.array(X), np.array(y)
    
    def train_sequence_predictor(self, sequence, window_size=20):
        """
        Train ML models to predict sequences
        Implements the experiment from Section 4.1
        """
        print(f"\nTraining sequence predictor with window size {window_size}")
        
        # Create dataset
        X, y = self.create_sliding_windows(sequence, window_size)
        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train multiple models
        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(50, 25),
                max_iter=500,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
        
        return results, X_test, y_test
    
    def visualize_sequence_patterns(self, sequence, title="Sequence Patterns"):
        """Visualize sequence patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original sequence
        axes[0, 0].plot(sequence[:500])
        axes[0, 0].set_title(f"{title} (First 500 values)")
        axes[0, 0].set_xlabel("Index")
        axes[0, 0].set_ylabel("Value")
        
        # Autocorrelation
        autocorr = np.correlate(sequence[:500], sequence[:500], mode='full')
        autocorr = autocorr / autocorr.max()
        axes[0, 1].plot(autocorr[len(autocorr)//2:][:100])
        axes[0, 1].set_title("Autocorrelation")
        axes[0, 1].set_xlabel("Lag")
        
        # Distribution of 0s and 1s
        ones = sum(sequence)
        zeros = len(sequence) - ones
        axes[1, 0].bar(['0', '1'], [zeros, ones])
        axes[1, 0].set_title(f"Distribution (zeros: {zeros}, ones: {ones})")
        
        # Running average
        window = 50
        running_avg = np.convolve(sequence, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(running_avg)
        axes[1, 1].set_title(f"Running Average (window={window})")
        axes[1, 1].set_xlabel("Position")
        axes[1, 1].set_ylabel("Average")
        
        plt.tight_layout()
        plt.show()
        
    def formulate_conjecture(self, sequence, model_results):
        """
        Formulate a conjecture based on pattern recognition
        This is the "Conjecture Formulation" step from Section 4.2
        """
        print("\n" + "="*60)
        print("FORMULATING CONJECTURE")
        print("="*60)
        
        conjecture = {
            'title': 'Pattern-Based Conjecture',
            'description': '',
            'evidence': [],
            'confidence': 0.0
        }
        
        # Analyze model performance
        for name, result in model_results.items():
            accuracy = result['accuracy']
            if accuracy > 0.85:
                evidence = f"{name} achieved {accuracy:.2%} accuracy"
                conjecture['evidence'].append(evidence)
                print(f"✓ Strong evidence: {evidence}")
        
        # Analyze sequence properties
        ones = sum(sequence)
        zeros = len(sequence) - ones
        ratio = ones / len(sequence) if len(sequence) > 0 else 0
        
        if 0.4 < ratio < 0.6:
            print(f"✓ Sequence is balanced (ones: {ones}, zeros: {zeros})")
            conjecture['evidence'].append(f"Sequence is well-balanced with {ratio:.1%} ones")
        
        # Look for periodic patterns
        # Check first period
        for period in range(2, 20):
            matches = 0
            for i in range(period, min(len(sequence), period * 10)):
                if sequence[i] == sequence[i - period]:
                    matches += 1
            if matches > 0.8 * min(len(sequence), period * 10):
                print(f"✓ Possible periodicity of length {period} detected")
                conjecture['evidence'].append(f"Period {period} detected with {matches/(period*10):.1%} match")
                break
        
        # Formulate conjecture
        if len(conjecture['evidence']) > 1:
            conjecture['description'] = (
                "This sequence exhibits predictable patterns that can be captured "
                "by machine learning models. Based on the high accuracy of multiple "
                "classifiers, we conjecture that there exists an underlying mathematical "
                "structure that governs this binary sequence."
            )
            conjecture['confidence'] = min(0.9, 0.5 + 0.1 * len(conjecture['evidence']))
        else:
            conjecture['description'] = (
                "Limited evidence suggests weak patterns in this sequence. "
                "The inability of standard ML models to achieve high accuracy "
                "indicates complexity that may require deeper mathematical insight."
            )
            conjecture['confidence'] = 0.3
        
        self.conjectures.append(conjecture)
        
        print(f"\nConjecture Formulated:")
        print(f"  Title: {conjecture['title']}")
        print(f"  Confidence: {conjecture['confidence']:.2%}")
        print(f"  Description: {conjecture['description']}")
        print(f"  Evidence: {len(conjecture['evidence'])} pieces")
        
        return conjecture

#=============================================================================
# PART 4: BIRCH TEST EVALUATION
#=============================================================================

class BirchTest:
    """
    Implements the Birch Test criteria for AI-driven theoretical discovery
    (A) Automaticity, (I) Interpretability, (N) Non-Triviality
    """
    
    def __init__(self):
        self.criteria = {
            'Automaticity': {
                'passed': False,
                'score': 0,
                'description': 'Discovery must be completely made by AI'
            },
            'Interpretability': {
                'passed': False,
                'score': 0,
                'description': 'Results must be precise and understandable to human mathematicians'
            },
            'Non-Triviality': {
                'passed': False,
                'score': 0,
                'description': 'Discovery must be significant enough for experts to work on it'
            }
        }
        self.overall_pass = False
        
    def evaluate_discovery(self, discovery, human_intervention=True, 
                          interpretable=True, significant=True):
        """
        Evaluate a discovery against Birch Test criteria
        """
        print("\n" + "="*60)
        print("BIRCH TEST EVALUATION")
        print("="*60)
        
        # Evaluate Automaticity
        if human_intervention:
            print("✗ Automaticity: FAILED (human intervention detected)")
            self.criteria['Automaticity']['passed'] = False
            self.criteria['Automaticity']['score'] = 0.3
        else:
            print("✓ Automaticity: PASSED (fully AI-driven)")
            self.criteria['Automaticity']['passed'] = True
            self.criteria['Automaticity']['score'] = 1.0
            
        # Evaluate Interpretability
        if interpretable:
            print("✓ Interpretability: PASSED (results are precise and understandable)")
            self.criteria['Interpretability']['passed'] = True
            self.criteria['Interpretability']['score'] = 1.0
        else:
            print("✗ Interpretability: FAILED (results are black-box)")
            self.criteria['Interpretability']['passed'] = False
            self.criteria['Interpretability']['score'] = 0.2
            
        # Evaluate Non-Triviality
        if significant:
            print("✓ Non-Triviality: PASSED (discovery is significant)")
            self.criteria['Non-Triviality']['passed'] = True
            self.criteria['Non-Triviality']['score'] = 1.0
        else:
            print("✗ Non-Triviality: FAILED (discovery is not significant enough)")
            self.criteria['Non-Triviality']['passed'] = False
            self.criteria['Non-Triviality']['score'] = 0.3
            
        # Overall evaluation
        scores = [c['score'] for c in self.criteria.values()]
        avg_score = np.mean(scores)
        self.overall_pass = all(c['passed'] for c in self.criteria.values())
        
        print(f"\nOverall Birch Test Score: {avg_score:.2f}")
        if self.overall_pass:
            print("✓ ALL THREE CRITERIA PASSED! This is a genuine AI-driven discovery!")
        else:
            print("✗ Discovery does not pass full Birch Test")
            print("  (See the paper: Section 4.2 - The Birch Test)")
            
        return {
            'criteria_scores': self.criteria,
            'average_score': avg_score,
            'overall_pass': self.overall_pass
        }

#=============================================================================
# PART 5: FULL DEMONSTRATION
#=============================================================================

def run_full_demonstration():
    """
    Run the complete demonstration of all three approaches
    """
    print("\n" + "="*80)
    print("FULL DEMONSTRATION OF AI-DRIVEN THEORETICAL DISCOVERY")
    print("="*80)
    
    #----------------------------------------------------------------------
    # Part 1: Bottom-Up Mathematics
    #----------------------------------------------------------------------
    print("\n" + "="*60)
    print("PART 1: BOTTOM-UP MATHEMATICS")
    print("="*60)
    
    prover = AutomatedTheoremProver()
    
    # Add axioms
    prover.add_axiom(Eq(sp.Symbol('x'), sp.Symbol('x')))
    prover.add_axiom(Eq(sp.Symbol('x') + 0, sp.Symbol('x')))
    prover.add_axiom(Eq(sp.Symbol('x') * 1, sp.Symbol('x')))
    
    # Define and prove a simple theorem
    x = sp.Symbol('x')
    theorem_expr = (x**2 - 1) / (x - 1)
    prover.define_theorem("algebraic_simplification", Eq(theorem_expr, x + 1))
    prover.prove_by_symbolic_manipulation("algebraic_simplification", str(theorem_expr))
    
    # Try a harder theorem
    theorem_expr2 = (x**3 - 1) / (x - 1)
    prover.define_theorem("cubic_simplification", Eq(theorem_expr2, x**2 + x + 1))
    prover.prove_by_symbolic_manipulation("cubic_simplification", str(theorem_expr2))
    
    # Generate corollaries
    prover.generate_corollaries("algebraic_simplification")
    
    # Knowledge-based reasoning
    response = prover.knowledge_based_reasoning("What are the proven theorems?")
    print(f"Q: What are the proven theorems?\nA: {response}")
    
    #----------------------------------------------------------------------
    # Part 2: Meta-Mathematics
    #----------------------------------------------------------------------
    print("\n" + "="*60)
    print("PART 2: META-MATHEMATICS")
    print("="*60)
    
    meta = MetaMathematics()
    
    # Load mathematical corpus
    corpus = [
        "The theorem states that for any prime number p, there exists no integer solution.",
        "The proof of Fermat's Last Theorem was completed by Andrew Wiles in 1994.",
        "Riemann hypothesis concerns the distribution of prime numbers.",
        "The Birch and Swinnerton-Dyer conjecture relates the rank of elliptic curves to L-functions.",
        "Machine learning can help discover patterns in mathematical data.",
        "The Prime Number Theorem states that π(x) ~ x/ln(x).",
        "Gödel's incompleteness theorems show that mathematics cannot be fully axiomatized.",
        "The classification of finite simple groups is a monumental achievement in mathematics."
    ]
    
    meta.load_mathematical_corpus(corpus)
    embeddings = meta.generate_word_embeddings()
    
    # Find mathematical patterns
    patterns = meta.find_mathematical_patterns()
    print(f"\nFound patterns: {list(patterns.keys())}")
    for pattern_type, matches in patterns.items():
        if len(matches) > 0:
            print(f"  {pattern_type}: {matches[:3]}...")
    
    # Analyze titles
    title_analysis = meta.analyze_arxiv_titles()
    print(f"\nAnalyzed {len(title_analysis['titles'])} paper titles")
    print(f"Features: {title_analysis['features_2d'].shape}")
    print(f"Vocabulary: {title_analysis['vocabulary'][:10]}...")
    
    # Generate mathematical text
    generated = meta.generate_mathematical_text("The theorem", length=15)
    if generated:
        print(f"\nGenerated text: {generated}")
    
    #----------------------------------------------------------------------
    # Part 3: Top-Down Mathematics
    #----------------------------------------------------------------------
    print("\n" + "="*60)
    print("PART 3: TOP-DOWN MATHEMATICS")
    print("="*60)
    
    topdown = TopDownMathematics()
    
    # Test with different sequences
    sequence_types = ['divisibility_by_3', 'primality', 'mobius']
    all_results = {}
    
    for seq_type in sequence_types:
        print(f"\n--- Analyzing {seq_type.upper()} ---")
        sequence, description = topdown.generate_binary_sequence_problem(seq_type)
        
        # Visualize the sequence
        topdown.visualize_sequence_patterns(sequence, description)
        
        # Train ML models
        results, X_test, y_test = topdown.train_sequence_predictor(sequence, window_size=15)
        all_results[seq_type] = results
        
        # Check if any model performs well
        best_accuracy = max(r['accuracy'] for r in results.values())
        print(f"\nBest accuracy for {seq_type}: {best_accuracy:.4f}")
        
        # Determine which case this represents
        if seq_type == 'divisibility_by_3':
            print("  → Case (i): Trivial problem - high accuracy expected")
        elif seq_type == 'primality':
            print("  → Case (ii): Moderate difficulty - ~80% accuracy expected")
        else:  # mobius
            print("  → Case (iii): Hard problem - ~50% accuracy expected")
        
        # Try to formulate a conjecture
        if best_accuracy > 0.75:
            conjecture = topdown.formulate_conjecture(sequence, results)
    
    #----------------------------------------------------------------------
    # Part 4: Birch Test Evaluation
    #----------------------------------------------------------------------
    print("\n" + "="*60)
    print("PART 4: BIRCH TEST EVALUATION")
    print("="*60)
    
    birch = BirchTest()
    
    # Evaluate the discoveries made
    print("\nEvaluating the top-down discoveries against Birch Test:")
    results = birch.evaluate_discovery(
        "Pattern-based conjecture from sequence analysis",
        human_intervention=True,  # We provided guidance
        interpretable=False,       # Models are black-box
        significant=False          # Not significant enough for expert attention
    )
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("  - Bottom-up mathematics: Automated theorem proving demonstrated")
    print("  - Meta-mathematics: NLP-based mathematical analysis shown")
    print("  - Top-down mathematics: Pattern recognition and conjecture formulation")
    print("  - Birch Test: Evaluated discovery against all three criteria")
    print("\nKey insights from the paper:")
    print("  ✓ AI can assist in all three approaches to mathematical discovery")
    print("  ✓ The Birch Test provides a high bar for AI-driven discoveries")
    print("  ✓ Human experts remain essential for interpretability and significance")
    print("  ✓ The future lies in human-AI collaboration")
    
    return all_results

#=============================================================================
# ADDITIONAL: Extended Examples
#=============================================================================

def extended_examples():
    """
    Additional examples and visualizations
    """
    print("\n" + "="*80)
    print("EXTENDED EXAMPLES AND ANALYSIS")
    print("="*80)
    
    # Create the famous sequence from Section 4.1
    print("\n1. The Three Sequences from Section 4.1:")
    print("   (i)  Divisibility by 3: 0,0,1,0,0,1,0,0,1,...")
    print("   (ii) Primality: 0,1,1,0,1,0,1,0,0,0,1,0,1,...")
    print("   (iii) Möbius parity: 1,1,1,0,1,1,1,0,0,1,1,0,...")
    
    # Demonstrate the sequence visualization
    topdown = TopDownMathematics()
    
    # Generate and analyze each sequence
    sequences = []
    for seq_type in ['divisibility_by_3', 'primality', 'mobius']:
        seq, desc = topdown.generate_binary_sequence_problem(seq_type)
        sequences.append((seq_type, seq, desc))
        
        # Print first 20 values
        print(f"\n{seq_type} (first 20): {seq[:20]}")
    
    # Show the pixelated image concept
    print("\n2. Pixelated Image Representation:")
    print("   (As mentioned in the paper: wrapping sequences into matrices)")
    
    # Take the primality sequence and reshape into 10x10
    seq_primality = sequences[1][1]
    matrix_10x10 = np.array(seq_primality[:100]).reshape(10, 10)
    
    print("\n   Primality sequence as 10x10 image:")
    print("   (1 = black, 0 = white)")
    for row in matrix_10x10:
        print('   ', ' '.join('█' if val else '░' for val in row))
    
    # Show the analogy
    print("\n3. Analogy from the paper:")
    print("   Bottom-up (and meta-) mathematics is language processing")
    print("   Top-down mathematics is image processing")
    print("\n   This is why wrapping mathematical objects as")
    print("   pixelated images can help uncover patterns!")
    
    return sequences

#=============================================================================
# MAIN EXECUTION
#=============================================================================

if __name__ == "__main__":
    # Run the full demonstration
    results = run_full_demonstration()
    
    # Run extended examples
    sequences = extended_examples()
    
    print("\n" + "="*80)
    print("COMPLETE IMPLEMENTATION FINISHED")
    print("="*80)
    print("\nThis implementation covers:")
    print("  ✓ Bottom-up mathematics (Automated Theorem Proving)")
    print("  ✓ Meta-mathematics (NLP-based mathematical analysis)")
    print("  ✓ Top-down mathematics (Pattern recognition & conjectures)")
    print("  ✓ Birch Test evaluation")
    print("  ✓ Sequence analysis and visualization")
    print("  ✓ The three binary sequences from Section 4.1")
    print("  ✓ Pixelated image representation concept")
    print("\nThe code demonstrates how AI can assist in theoretical")
    print("discovery across all three approaches described in the paper.")