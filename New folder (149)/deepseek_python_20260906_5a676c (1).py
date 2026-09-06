import numpy as np
import math
from collections import defaultdict

# -------------------------------------------------------------------
# 1.  Single‑Layer Attention Transformer (exact as defined in the paper)
# -------------------------------------------------------------------

class SingleLayerAttention:
    """
    One‑layer attention‑only transformer with k heads.
    Reads a Boolean input x = (x1,...,xn) followed by a query token '?'.
    Outputs 1 iff logit_1 > logit_0 (ties resolve to 0).
    """
    def __init__(self, k, d, d_out=2):
        self.k = k
        self.d = d
        self.d_out = d_out

        # embedding for each token: we store vectors for (bit, position) and for '?'
        # bit ∈ {0,1}, position ∈ {0,...,n-1} (0‑based)
        self.emb = {}          # key: (bit, pos) -> np.array(d)
        self.emb_query = None  # np.array(d)

        # per‑head parameters
        self.A = []   # list of d x d matrices
        self.V = []   # list of d x d_out matrices
        self.WO = None  # d_out x 2 matrix (two logits)

    def set_embedding(self, emb_query, emb_dict):
        """
        emb_query : numpy array of shape (d,)
        emb_dict  : dict mapping (bit, pos) -> numpy array (d,)
        """
        self.emb_query = np.asarray(emb_query, dtype=float)
        self.emb = {(b, p): np.asarray(v, dtype=float) for (b, p), v in emb_dict.items()}

    def add_head(self, A, V):
        """Add one attention head with matrices A (d x d) and V (d x d_out)."""
        self.A.append(np.asarray(A, dtype=float))
        self.V.append(np.asarray(V, dtype=float))

    def set_output(self, WO):
        """Set the output matrix WO (d_out x 2)."""
        self.WO = np.asarray(WO, dtype=float)

    def forward(self, x):
        """
        x : list of ints (0/1) of length n.
        Returns predicted output bit (0 or 1).
        """
        n = len(x)
        # Build the input sequence: tokens for each position then query.
        tokens = []
        for i, bit in enumerate(x):
            tokens.append(self.emb[(bit, i)])
        tokens.append(self.emb_query)   # query token at the end
        T = np.array(tokens)            # shape (n+1) x d

        # Compute weighted sum over heads
        MHA = np.zeros(self.d_out)
        for j in range(self.k):
            A_j = self.A[j]
            V_j = self.V[j]
            # query vector for this head
            q = self.emb_query @ A_j   # shape (d,)
            # scores for all tokens (including query itself)
            scores = T @ q             # shape (n+1,)
            # softmax
            exp_scores = np.exp(scores - np.max(scores))  # numerical stability
            soft = exp_scores / np.sum(exp_scores)
            # weighted sum of values
            values = T @ V_j           # shape (n+1) x d_out
            weighted = soft @ values   # shape (d_out,)
            MHA += weighted

        # logits
        logits = MHA @ self.WO         # shape (2,)
        # output 1 if logit_1 > logit_0 else 0
        return 1 if logits[1] > logits[0] else 0

    def scalar_gap(self, x):
        """
        Compute the decision gap Δ(x) = Z0 - Z1 as a sum of rational terms.
        Returns Δ(x) (float).  Output = 1 iff Δ(x) < 0.
        """
        n = len(x)
        gap = 0.0
        for j in range(self.k):
            A_j = self.A[j]
            V_j = self.V[j]
            q = self.emb_query @ A_j
            # numerator and denominator for this head
            denom = 0.0
            num = 0.0
            # self‑attention (query attends to itself)
            r_q = math.exp(q @ self.emb_query)
            # For the scalar normal form, we need u_q = emb_query · (V_j (w0-w1)^T)
            # but we can compute directly from the folded matrices.
            # However, for the parity construction we use the scalar form,
            # so we can also compute the gap as sum_j s_j/d_j by using the
            # explicit formulas from the paper.  We'll implement a helper below.
            # This is a generic version that uses the actual forward pass but
            # computes the difference of logits directly.
            # We already have MHA from forward, so we can compute logit difference.
        # Better: use the generic forward to compute logits and return difference.
        # But we need the gap for verification; we'll implement a separate method
        # that computes the scalar normal form from the parameters directly
        # (for the parity construction).
        return None  # placeholder, see below

# -------------------------------------------------------------------
# 2.  Parity construction (Theorem 2 and Lemma 4)
# -------------------------------------------------------------------

def polynomial_from_roots(roots):
    """Return coefficients of ∏ (t - r) as list [c0, c1, ..., cdeg]."""
    poly = [1.0]
    for r in roots:
        # multiply by (t - r)
        new = [0.0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new[i] += c * (-r)
            new[i+1] += c
        poly = new
    return poly

def eval_poly(poly, t):
    """Evaluate polynomial given by coefficients [c0, c1, ...] at t."""
    val = 0.0
    for i, c in enumerate(poly):
        val += c * (t ** i)
    return val

def get_alpha_beta(k):
    """
    Compute coefficients α_j, β_j (j=1..k) such that
        P(t)/D(t) = Σ_{j=1}^k (α_j + β_j t) / (t + j)
    where D(t)=∏_{j=1}^k (t+j), P(t)=(-1)^k ∏_{m=0}^{k-1} (t - m - 1/2).
    Returns lists alpha[1..k], beta[1..k] (1‑based indexing).
    """
    # D(t) = ∏_{j=1}^k (t + j)  => roots are -1, -2, ..., -k
    D_roots = [-j for j in range(1, k+1)]
    D_coeff = polynomial_from_roots(D_roots)  # [c0, c1, ..., ck]

    # P(t) = (-1)^k ∏_{m=0}^{k-1} (t - m - 1/2)
    P_roots = [m + 0.5 for m in range(k)]   # roots: 0.5, 1.5, ..., k-0.5
    P_coeff = polynomial_from_roots(P_roots)
    # multiply by (-1)^k
    sign = (-1) ** k
    P_coeff = [c * sign for c in P_coeff]

    # constant term C = leading coefficient of P / leading coeff of D
    # Both are monic, so C = (-1)^k * 1 / 1 = (-1)^k
    C = sign

    # Compute residues λ_j = P(-j) / ∏_{l≠j} (l - j)
    alpha = [0.0] * (k+1)  # 1‑based
    beta = [0.0] * (k+1)
    for j in range(1, k+1):
        t = -j
        P_val = eval_poly(P_coeff, t)
        # product over l ≠ j of (l - j)
        prod = 1.0
        for l in range(1, k+1):
            if l == j:
                continue
            prod *= (l - j)   # note: l - j = -(j - l)
        lambda_j = P_val / prod
        beta[j] = C / k
        alpha[j] = lambda_j + C * j / k

    return alpha, beta

def build_parity_transformer(k, M=1000.0):
    """
    Construct a one‑layer transformer with k heads that computes k‑bit parity.
    Uses the additive embedding from the proof.
    Returns a SingleLayerAttention instance.
    """
    d = k + 3
    # Embedding: standard basis
    def e(i):
        vec = np.zeros(d)
        vec[i] = 1.0
        return vec

    # query token
    emb_query = e(d-1)   # index d-1 (0‑based) = last coordinate

    # position and value embeddings:
    # bit 0 at position i: e0 + e_{i+2}
    # bit 1 at position i: e1 + e_{i+2}
    emb_dict = {}
    for i in range(k):
        emb_dict[(0, i)] = e(0) + e(i+2)
        emb_dict[(1, i)] = e(1) + e(i+2)

    # compute α, β
    alpha, beta = get_alpha_beta(k)

    # instantiate model
    model = SingleLayerAttention(k, d, d_out=2)
    model.set_embedding(emb_query, emb_dict)
    model.set_output(np.eye(2))  # WO = I_2, so w0=[1,0], w1=[0,1]

    # Build each head
    for j in range(1, k+1):
        # ξ_j = log(j/k) e0 + log(1+j/k) e1 - M e_{d-1}
        xi = math.log(j/k) * e(0) + math.log(1 + j/k) * e(1) - M * e(d-1)

        # A_j: we need v_? A_j = ξ_j. Since v_? = e_{d-1}, set last row of A_j = ξ_j,
        # all other rows zero.
        A_j = np.zeros((d, d))
        A_j[d-1, :] = xi

        # δ_j = η0,j e0 + η1,j e1
        eta0 = alpha[j] / j
        eta1 = (beta[j] + alpha[j] / k) / (1 + j / k)
        delta = eta0 * e(0) + eta1 * e(1)

        # V_j: shape d x 2, column0 = delta/2, column1 = -delta/2
        V_j = np.zeros((d, 2))
        V_j[:, 0] = delta / 2.0
        V_j[:, 1] = -delta / 2.0

        model.add_head(A_j, V_j)

    return model

def test_parity(k):
    """Test that the built transformer computes k‑bit parity correctly."""
    model = build_parity_transformer(k)
    n = k
    correct = 0
    total = 2**k
    for bits in range(total):
        x = [(bits >> i) & 1 for i in range(k)]
        pred = model.forward(x)
        true_parity = sum(x) % 2
        if pred == true_parity:
            correct += 1
    print(f"k={k}: {correct}/{total} correct")
    return correct == total

# -------------------------------------------------------------------
# 3.  Universal upper bound (monomial vote, approximate)
#     (Theorem 5, Lemma 2)
# -------------------------------------------------------------------

def multilinear_coefficients(f, n):
    """
    Compute the Möbius coefficients α_S for f: {0,1}^n -> {0,1}.
    Returns a dict mapping frozenset S -> α_S (int).
    """
    alpha = {}
    for S_mask in range(1 << n):
        S = frozenset(i for i in range(n) if (S_mask >> i) & 1)
        # compute α_S = Σ_{T ⊆ S} (-1)^{|S|-|T|} f(1_T)
        val = 0
        T_mask = S_mask
        while True:
            T = frozenset(i for i in range(n) if (T_mask >> i) & 1)
            x = [1 if i in T else 0 for i in range(n)]
            sign = -1 if ((len(S) - len(T)) % 2) else 1
            val += sign * f(x)
            if T_mask == 0:
                break
            T_mask = (T_mask - 1) & S_mask
        if val != 0:
            alpha[S] = val
    return alpha

def build_monomial_head(S, R=20.0):
    """
    Build a single head that approximates the monomial ∏_{i∈S} x_i.
    Uses the additive embedding construction from Lemma 2.
    Returns (A, V) and also the required embedding parameters.
    This is a placeholder: we construct a head that directly computes the product
    using a large bias, but the implementation is simplified.
    In the paper, they use the query token itself as a reference and set scores
    so that softmax concentrates on positions in S that are 1.
    We'll implement a version that uses the scalar normal form: we set the head's
    numerator and denominator to yield the product exactly via a different mechanism.
    Since the paper's construction is approximate, we'll use a large R to make it
    arbitrarily close.  For simplicity, we'll just return a dummy head that
    computes the product using the scalar gap with a custom formula.
    """
    # This is a sketch; a full implementation would set embeddings and matrices
    # as described in the proof.  For the purpose of the code, we can omit it.
    raise NotImplementedError("Monomial head construction is approximate; use parity test instead.")

def build_universal_transformer(f, n):
    """
    Build a transformer with at most 2^n heads that computes f exactly
    (using the monomial vote).  This is the upper bound from Theorem 5.
    """
    alpha = multilinear_coefficients(f, n)
    # For each S with nonzero coefficient, we add a head that approximates the monomial.
    # We'll return a model with those heads.
    # This is not fully implemented here; see parity example for a concrete construction.
    raise NotImplementedError("Universal construction not implemented in this demo.")

# -------------------------------------------------------------------
# 4.  Main demo
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Test parity for k=1,2,3
    for k in range(1, 5):
        success = test_parity(k)
        print(f"Parity with {k} heads: {'PASS' if success else 'FAIL'}")

    # Additional: show the scalar gap for a few inputs (optional)
    # model = build_parity_transformer(2)
    # print(model.scalar_gap([0,0]))  # etc.