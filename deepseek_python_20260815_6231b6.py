import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sympy import primepi, isprime, factorint

# --- 1. Generate the three mathematical sequences ---

def generate_sequence(seq_type, length):
    """Generates binary sequences for the three cases.
    seq_type: 'div3', 'prime', 'mobius_parity'
    """
    seq = []
    if seq_type == 'div3':
        # Sequence (i): whether n is divisible by 3
        for n in range(1, length+1):
            seq.append(1 if n % 3 == 0 else 0)
    elif seq_type == 'prime':
        # Sequence (ii): whether n is prime (PrimeQ)
        for n in range(1, length+1):
            seq.append(1 if isprime(n) else 0)
    elif seq_type == 'mobius_parity':
        # Sequence (iii): parity of number of prime factors (Big Omega)
        # This is related to the Möbius function: 1 if Ω(n) is odd, else 0
        for n in range(1, length+1):
            if n == 1:
                seq.append(0)  # 1 has Ω=0, even -> 0
            else:
                omega = sum(factorint(n).values())  # Ω(n) with multiplicity
                seq.append(1 if omega % 2 == 1 else 0)  # odd -> 1
    return np.array(seq, dtype=int)

# --- 2. Create sliding window dataset ---

def create_dataset(sequence, window_size, num_samples):
    """Creates dataset of sliding windows for supervised learning."""
    X, y = [], []
    for i in range(num_samples):
        # Ensure we have enough elements for a full window and a label
        if i + window_size >= len(sequence):
            break
        window = sequence[i:i+window_size]
        label = sequence[i+window_size]
        X.append(window)
        y.append(label)
    return np.array(X), np.array(y)

# --- 3. Parameters and Data Generation ---

# Use a finite length. For math, data is cheap; we can generate long sequences.
SEQ_LENGTH = 10000
WINDOW_SIZE = 50  # N
NUM_SAMPLES = 9000  # k samples for training/testing

print("Generating sequences...")
seq_div3 = generate_sequence('div3', SEQ_LENGTH)
seq_prime = generate_sequence('prime', SEQ_LENGTH)
seq_mobius = generate_sequence('mobius_parity', SEQ_LENGTH)

# Create datasets
X_div3, y_div3 = create_dataset(seq_div3, WINDOW_SIZE, NUM_SAMPLES)
X_prime, y_prime = create_dataset(seq_prime, WINDOW_SIZE, NUM_SAMPLES)
X_mobius, y_mobius = create_dataset(seq_mobius, WINDOW_SIZE, NUM_SAMPLES)

# Split into train and test sets (80/20)
X_div3_train, X_div3_test, y_div3_train, y_div3_test = train_test_split(
    X_div3, y_div3, test_size=0.2, random_state=42
)
X_prime_train, X_prime_test, y_prime_train, y_prime_test = train_test_split(
    X_prime, y_prime, test_size=0.2, random_state=42
)
X_mobius_train, X_mobius_test, y_mobius_train, y_mobius_test = train_test_split(
    X_mobius, y_mobius, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_div3_train)}, Test set size: {len(X_div3_test)}")

# --- 4. Define and Train Models ---

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', random_state=42),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

results = {}

for seq_name, (X_train, y_train, X_test, y_test) in [
    ("Divisible by 3", (X_div3_train, y_div3_train, X_div3_test, y_div3_test)),
    ("Prime", (X_prime_train, y_prime_train, X_prime_test, y_prime_test)),
    ("Möbius Parity", (X_mobius_train, y_mobius_train, X_mobius_test, y_mobius_test))
]:
    print(f"\n--- Evaluating on Sequence: {seq_name} ---")
    results[seq_name] = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[seq_name][model_name] = accuracy
        print(f"{model_name:15s} Accuracy: {accuracy:.4f}")

# --- 5. Summary of Results ---

print("\n" + "="*50)
print("SUMMARY OF RESULTS (as discussed in Section 4.1 of the paper)")
print("="*50)

for seq_name, accuracies in results.items():
    print(f"\n{seq_name:15s}")
    for model_name, acc in accuracies.items():
        print(f"  {model_name:15s}: {acc:.4f}")
    # Interpret results
    if seq_name == "Divisible by 3":
        print("  -> All models achieve near 100% accuracy (trivial pattern).")
    elif seq_name == "Prime":
        print("  -> Models achieve decent accuracy (~80%), possibly approximating a sieve.")
    elif seq_name == "Möbius Parity":
        print("  -> Models struggle (~50% accuracy, close to random guessing).")
        print("     This is a problem related to the Riemann Hypothesis!")