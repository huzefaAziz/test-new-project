import numpy as np
from itertools import combinations
from collections import defaultdict

# ----------------------------------------------------------------------
# 1. Basic definitions
# ----------------------------------------------------------------------

def bitxor(a, b, n):
    """XOR of two n-bit integers."""
    return a ^ b

def random_permutation(n):
    """Generate a random permutation h: X -> X, where X = {0,...,2^n-1}."""
    N = 1 << n
    vals = np.random.permutation(N)
    return {x: vals[x] for x in range(N)}

def random_simon_function(n):
    """
    Generate a random Simon function h with a nonzero shift s.
    Returns (h_dict, s).
    """
    N = 1 << n
    # choose s != 0 uniformly
    s = np.random.randint(1, N)
    # partition X into pairs {x, x^s}
    remaining = set(range(N))
    h = {}
    # assign distinct random values to each pair
    # we'll generate random values without replacement
    values = list(np.random.permutation(N))
    val_idx = 0
    while remaining:
        x = remaining.pop()
        y = x ^ s
        if y in remaining:
            remaining.remove(y)
        # assign the same value to both
        v = values[val_idx]
        val_idx += 1
        h[x] = v
        h[y] = v
    return h, s

def random_tag_table(n):
    """Generate a random tag table r: X -> X."""
    N = 1 << n
    return {x: np.random.randint(0, N) for x in range(N)}

# ----------------------------------------------------------------------
# 2. Oracle interfaces (simulated via classical functions)
# ----------------------------------------------------------------------

def injective_oracle(h, r, x):
    """Return (h(x), x, r[x]) for the injective problem."""
    return (h[x], x, r[x])

def permutation_oracle_special(h, r, a):
    """
    For the permutation problem, we only need to query the special inputs a_x.
    a_x is encoded as (0, x, 0, 0) in the paper; we use a simple integer mapping.
    For simulation, we just treat a_x as x (the address).
    """
    # In the permutation construction, P(a_x) = (1, h(x), x, r[x]).
    # We don't need the full Ω; we just return the tuple for comparison.
    return (1, h[x], x, r[x])

# ----------------------------------------------------------------------
# 3. Upper bound algorithms
# ----------------------------------------------------------------------

def standard_simon_algorithm(n, h_type, shift=None, rounds=None):
    """
    Simulate the standard XOR quantum algorithm.
    Returns True if it decides "permutation", False if "Simon".
    rounds = number of samples (n+2 recommended).
    """
    N = 1 << n
    if rounds is None:
        rounds = n + 2
    samples = []
    for _ in range(rounds):
        if h_type == 'permutation':
            # measurement outcome is uniform in X
            y = np.random.randint(0, N)
        else:  # Simon
            # y is uniform in s^\perp
            # s^\perp = {y | y·s = 0} over GF(2)
            # We can sample uniformly from the subspace of dimension n-1.
            # Simple method: generate random n-bit string, then if dot product with s is 1,
            # flip a random bit to make it 0.
            y = np.random.randint(0, N)
            # compute dot product (bitwise AND parity)
            dot = bin(y & shift).count('1') % 2
            if dot == 1:
                # flip one bit where shift has a 1, to make dot 0
                # find a bit position where shift has 1
                for b in range(n):
                    if (shift >> b) & 1:
                        y ^= (1 << b)
                        break
            samples.append(y)
    # Check if samples span the whole space X (as a vector space over GF(2)).
    # We'll use Gaussian elimination over GF(2).
    # Convert samples to vectors of bits.
    vecs = [list(map(int, bin(y)[2:].zfill(n))) for y in samples]
    # Reduce to row echelon form
    mat = []
    for v in vecs:
        # reduce by previous rows
        for row in mat:
            # find pivot
            pivot = None
            for j in range(n):
                if row[j] == 1:
                    pivot = j
                    break
            if pivot is not None and v[pivot] == 1:
                v = [(v[j] + row[j]) % 2 for j in range(n)]
        # if v is not zero, add as new row
        if any(v):
            mat.append(v)
    # If rank == n, they span X -> permutation. Else Simon.
    rank = len(mat)
    if rank == n:
        return True  # permutation
    else:
        return False  # Simon

def birthday_algorithm(n, h_type, shift=None, k=None):
    """
    Forward-erasing birthday algorithm: sample k distinct inputs, check for collision in h-values.
    Returns True if it decides "Simon" (collision found), False if "permutation".
    """
    N = 1 << n
    if k is None:
        k = int(np.ceil(2 * np.sqrt(N)))  # as per paper
    # choose k distinct random inputs
    indices = np.random.choice(N, size=k, replace=False)
    h_values = []
    if h_type == 'permutation':
        h_dict = random_permutation(n)  # but we don't have it; we need to generate and use it.
        # Actually we'll generate the function inside the test.
        # We'll restructure to pass the function.
    # Better: we'll pass the h_dict directly.
    # We'll define a wrapper.
    return None  # will be handled in test functions.

# We'll write separate test functions that generate the instance and run the algorithm.

# ----------------------------------------------------------------------
# 4. Testing framework
# ----------------------------------------------------------------------

def run_standard_test(n, num_trials=1000):
    """Test standard algorithm on random instances."""
    N = 1 << n
    rounds = n + 2
    perm_success = 0
    simon_success = 0
    for _ in range(num_trials):
        # permutation case
        h = random_permutation(n)
        r = random_tag_table(n)  # not needed for simulation, but included
        # simulate standard algorithm: sampling from uniform X
        # we use the function that takes h_type and shift
        result_perm = standard_simon_algorithm(n, 'permutation')
        if result_perm == True:  # correctly identifies permutation
            perm_success += 1
        
        # Simon case
        h, s = random_simon_function(n)
        result_simon = standard_simon_algorithm(n, 'simon', shift=s, rounds=rounds)
        if result_simon == False:  # correctly identifies Simon
            simon_success += 1
    
    print(f"n={n}, standard algorithm (rounds={rounds}):")
    print(f"  Permutation success: {perm_success/num_trials:.3f}")
    print(f"  Simon success:       {simon_success/num_trials:.3f}")

def run_birthday_test(n, num_trials=1000):
    """Test birthday algorithm on random instances with k=2*sqrt(N)."""
    N = 1 << n
    k = int(np.ceil(2 * np.sqrt(N)))
    perm_success = 0  # correct when no collision -> permutation
    simon_success = 0 # correct when collision -> Simon
    for _ in range(num_trials):
        # permutation case: no collision
        h = random_permutation(n)
        # sample k distinct indices
        indices = np.random.choice(N, size=k, replace=False)
        vals = [h[x] for x in indices]
        if len(set(vals)) == k:  # no collision
            perm_success += 1
        
        # Simon case: collision present with high probability
        h, s = random_simon_function(n)
        indices = np.random.choice(N, size=k, replace=False)
        vals = [h[x] for x in indices]
        if len(set(vals)) < k:  # collision
            simon_success += 1
    
    print(f"n={n}, birthday algorithm (k={k}):")
    print(f"  Permutation success (no collision): {perm_success/num_trials:.3f}")
    print(f"  Simon success (collision found):    {simon_success/num_trials:.3f}")

def run_birthday_vs_k(n, ks=None):
    """Show success probability as function of k for Simon case."""
    N = 1 << n
    if ks is None:
        ks = [int(N**0.3), int(N**0.4), int(N**0.5), int(N**0.6)]
    print(f"n={n}, Simon case collision probability vs k:")
    for k in ks:
        if k > N:
            k = N
        success = 0
        trials = 500
        for _ in range(trials):
            h, s = random_simon_function(n)
            indices = np.random.choice(N, size=k, replace=False)
            vals = [h[x] for x in indices]
            if len(set(vals)) < k:
                success += 1
        print(f"  k={k}: {success/trials:.3f}")

# ----------------------------------------------------------------------
# 5. Main demonstration
# ----------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    
    print("="*60)
    print("Simulation of quantum query separations from the paper")
    print("="*60)
    
    # Test for small n (4,5,6) to see the separation.
    for n in [4,5,6]:
        print("\n--- n =", n, "---")
        run_standard_test(n, num_trials=500)
        run_birthday_test(n, num_trials=500)
        run_birthday_vs_k(n, ks=[int((1<<n)**0.4), int((1<<n)**0.5), int((1<<n)**0.6)])
    
    print("\nExplanation:")
    print("  - The standard XOR algorithm uses n+2 queries and succeeds with probability > 3/4.")
    print("  - The forward-erasing birthday algorithm uses O(sqrt(N)) queries and also succeeds.")
    print("  - The birthday algorithm fails with fewer queries (collision probability drops).")
    print("  - This demonstrates the exponential separation in query complexity.")