import numpy as np
from scipy.linalg import svd, qr, pinv
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix
import warnings
from typing import Tuple, Optional

class LpSubspaceCoreset:
    """
    Implementation of Nearly Optimal Strong Coresets for ℓp Subspace Approximation
    Based on the paper by Lin, Mirrokni, and Woodruff
    """
    
    def __init__(self, p: float, k: int, epsilon: float, delta: float = 0.1):
        """
        Initialize the coreset construction
        
        Args:
            p: ℓp norm parameter (1 <= p < 2 or p > 2)
            k: rank of subspace approximation
            epsilon: approximation error
            delta: failure probability
        """
        assert p != 2, "p=2 is the Euclidean case (PCA), which is handled separately"
        assert p >= 1, "p must be >= 1"
        
        self.p = p
        self.k = k
        self.epsilon = epsilon
        self.delta = delta
        self.is_small_p = p < 2
        
        # Constants from the paper
        self.C_p = self._get_constant()
    
    def _get_constant(self) -> float:
        """Get constant C_p depending on p (simplified)"""
        if self.p < 2:
            return max(10, 100/self.p)
        else:
            return max(10, 10*self.p)
    
    def _compute_lewis_weights(self, M: np.ndarray, max_iter: int = 10) -> np.ndarray:
        """
        Compute ℓp Lewis weights using the Cohen-Peng algorithm (Algorithm 2.1 from paper)
        
        Args:
            M: Input matrix of shape (n, r)
            max_iter: Maximum iterations
            
        Returns:
            Lewis weights array of shape (n,)
        """
        n, r = M.shape
        
        # Remove zero rows
        zero_rows = np.all(np.abs(M) < 1e-10, axis=1)
        M_clean = M[~zero_rows]
        n_clean = M_clean.shape[0]
        
        if n_clean == 0:
            return np.zeros(n)
        
        # Initialize weights to ones
        w = np.ones(n_clean)
        
        for _ in range(max_iter):
            # Construct W^{1/2 - 1/p}
            exponent = 0.5 - 1.0/self.p
            W = np.diag(w ** exponent)
            
            # Compute leverage scores of W^{1/2-1/p} * M
            M_scaled = W @ M_clean
            
            # Compute SVD for leverage scores
            U, S, Vt = svd(M_scaled, full_matrices=False)
            
            # Leverage scores are row norms of U
            tau = np.sum(U**2, axis=1)
            
            # Update weights: w_i = (w_i^(2/p-1) * tau_i)^(p/2)
            exponent2 = 2.0/self.p - 1
            w_new = (w ** exponent2 * tau) ** (self.p/2)
            
            # Normalize to avoid numerical issues
            w_new = w_new / np.sum(w_new) * r
            
            # Check for convergence
            if np.max(np.abs(w_new - w) / (np.abs(w) + 1e-10)) < 1e-6:
                w = w_new
                break
            w = w_new
        
        # Restore zero rows
        lewis_weights = np.zeros(n)
        lewis_weights[~zero_rows] = w
        
        return lewis_weights
    
    def _compute_leverage_scores(self, M: np.ndarray) -> np.ndarray:
        """Compute standard leverage scores"""
        n, r = M.shape
        
        if n == 0 or r == 0:
            return np.zeros(n)
        
        # Compute leverage scores via SVD
        U, S, Vt = svd(M, full_matrices=False)
        leverage_scores = np.sum(U**2, axis=1)
        
        return leverage_scores
    
    def _bicriteria_subspace(self, A: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Lemma 2.5: Fast bicriteria construction for low-rank split
        
        Args:
            A: Input matrix of shape (n, d)
            
        Returns:
            F0: Basis for the bicriteria subspace
            r: Dimension of F0
        """
        n, d = A.shape
        
        # Simplified implementation: use top k right singular vectors
        # In practice, this uses the Woodruff-Yasuda algorithm
        U, S, Vt = svd(A, full_matrices=False)
        
        # For p < 2, we need O(k log N) rows
        # Use top k dimensions
        r = min(2 * self.k, d)
        F0 = Vt[:r].T  # Basis for subspace
        
        return F0, r
    
    def _compute_residual_and_low_rank(self, A: np.ndarray, F0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute the low-rank and residual decomposition A = B + E
        
        Args:
            A: Input matrix of shape (n, d)
            F0: Basis for bicriteria subspace (d, r)
            
        Returns:
            B: Low-rank matrix (n, d)
            E: Residual matrix (n, d)
            R: Residual ℓ_p norm
        """
        n, d = A.shape
        r = F0.shape[1]
        
        # Project each row onto F0
        # b_i = P_{F0} a_i
        P = F0 @ F0.T  # Projection matrix
        B = A @ P
        E = A - B
        
        # Compute residual ℓ_{p,2} norm
        residual_norms = np.sum(E**2, axis=1) ** (self.p/2)
        R = np.sum(residual_norms)
        
        return B, E, R
    
    def _construct_sampling_distribution(self, B: np.ndarray, E: np.ndarray, R: float) -> np.ndarray:
        """
        Construct sampling distribution q_i ∝ w_i + D * rho_i
        
        Args:
            B: Low-rank matrix (n, d)
            E: Residual matrix (n, d)
            R: Residual ℓ_p norm
            
        Returns:
            Sampling probabilities q of shape (n,)
        """
        n, d = B.shape
        D = self.k + min(B.shape[0], B.shape[1])  # k + rank(B)
        
        # Compute Lewis weights for B
        # For efficiency, if B has rank r, we can compute Lewis weights on a basis
        w = self._compute_lewis_weights(B)
        
        # Compute residual fractions
        if R > 0:
            residual_norms = np.sum(E**2, axis=1) ** (self.p/2)
            rho = residual_norms / R
        else:
            rho = np.zeros(n)
        
        # Balanced scores: s_i = w_i + D * rho_i
        s = w + D * rho
        
        # Normalize to get probabilities
        q = s / np.sum(s)
        
        return q
    
    def _sample_rows(self, A: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample rows according to distribution q (Definition 2.3)
        
        Args:
            A: Input matrix of shape (n, d)
            q: Sampling probabilities of shape (n,)
            
        Returns:
            SA: Sampled and rescaled matrix
            weights: Sampling weights
        """
        n, d = A.shape
        
        # Sample size m = O_p(k * eps^{-2})
        m = int(self.C_p * self.k * self.epsilon**(-2) * np.log(self.C_p * self.k / self.epsilon))
        m = max(m, 1)
        m = min(m, n)  # Can't sample more than n rows
        
        # Sample m indices with replacement
        indices = np.random.choice(n, size=m, p=q)
        
        # Create sampling matrix S
        SA = np.zeros((m, d))
        weights = np.zeros(m)
        
        for t, idx in enumerate(indices):
            # Rescale by (m * q_i)^{-1/p}
            if q[idx] > 0:
                scale = (m * q[idx]) ** (-1.0/self.p)
            else:
                scale = 0
            SA[t] = scale * A[idx]
            weights[t] = scale
        
        return SA, weights
    
    def _pre_sparsify(self, A: np.ndarray) -> np.ndarray:
        """
        Theorem 2.4: Pre-sparsification to remove log n dependence
        
        Args:
            A: Input matrix
            
        Returns:
            Sparsified matrix with O(poly(k/eps)) rows
        """
        n, d = A.shape
        
        # Simple implementation: uniform sampling
        # In practice, use the Woodruff-Yasuda algorithm
        target_rows = int(self.C_p * self.k * self.epsilon ** (-2))
        target_rows = min(target_rows, n)
        
        if target_rows < n:
            indices = np.random.choice(n, size=target_rows, replace=False)
            return A[indices]
        else:
            return A
    
    def _compute_ridge_leverage_scores(self, B: np.ndarray, lambda_reg: float) -> np.ndarray:
        """
        Compute ridge leverage scores for p > 2 case
        
        Args:
            B: Input matrix (N, d)
            lambda_reg: Regularization parameter
            
        Returns:
            Ridge leverage scores of shape (N,)
        """
        N, d = B.shape
        
        # Compute (B^T B + lambda_reg * I)^{-1}
        BtB = B.T @ B
        
        # Add regularization to avoid singularities
        if N < d or d == 0:
            BtB = BtB + lambda_reg * np.eye(d)
        
        try:
            # Use SVD for stability
            U, S, Vt = svd(BtB, full_matrices=False)
            # Regularized inverse
            S_inv = 1.0 / (S + lambda_reg + 1e-10)
            # Compute ridge leverage scores
            tau = np.zeros(N)
            for i in range(N):
                Bi = B[i:i+1, :]  # Row as row vector
                tau[i] = Bi @ Vt.T @ np.diag(S_inv) @ Vt @ Bi.T
                tau[i] = tau[i].item()
        except:
            # Fallback: use pseudoinverse
            try:
                inv = pinv(BtB + lambda_reg * np.eye(d))
                for i in range(N):
                    Bi = B[i:i+1, :]
                    tau[i] = Bi @ inv @ Bi.T
                    tau[i] = tau[i].item()
            except:
                tau = np.ones(N) / N  # Fallback to uniform
        
        return np.clip(tau, 0, 1)
    
    def _one_round_sampling_p_gt_2(self, B: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        One-round sampling primitive for p > 2 (from Section 6)
        
        Args:
            B: Active matrix (N, d)
            alpha: Sampling threshold
            
        Returns:
            Sampled and rescaled matrix
            Sampling probabilities
        """
        N, d = B.shape
        
        if N == 0 or d == 0:
            return B, np.ones(N)
        
        # Compute rank-k approximation
        if self.k < min(N, d):
            U, S, Vt = svd(B, full_matrices=False)
            B_k = U[:, :self.k] @ np.diag(S[:self.k]) @ Vt[:self.k, :]
        else:
            B_k = B
        
        # Compute ridge leverage scores
        lambda_reg = np.sum((B - B_k)**2) / (self.k * N) if self.k > 0 else 1.0
        tau = self._compute_ridge_leverage_scores(B, lambda_reg)
        
        # Sampling probabilities: q_i = min(1, N^{p/2-1} * tau_i^{p/2} / alpha)
        term1 = N ** (self.p/2 - 1) if N > 0 else 0
        term2 = tau ** (self.p/2)
        q = np.minimum(1, term1 * term2 / (alpha + 1e-10))
        
        # Sample rows
        sampled_indices = []
        for i in range(N):
            if np.random.random() < q[i]:
                sampled_indices.append(i)
        
        if len(sampled_indices) == 0:
            # If no rows sampled, sample one row with highest probability
            idx = np.argmax(q)
            sampled_indices = [idx]
        
        # Rescale rows
        SA = np.zeros((len(sampled_indices), d))
        for j, i in enumerate(sampled_indices):
            if q[i] > 0:
                SA[j] = B[i] / (q[i] ** (1.0/self.p))
            else:
                SA[j] = B[i]
        
        return SA, q
    
    def fit_transform(self, A: np.ndarray) -> np.ndarray:
        """
        Main method to construct strong row coreset
        
        Args:
            A: Input matrix of shape (n, d)
            
        Returns:
            SA: Coreset matrix with O(poly(k/eps)) rows
        """
        n, d = A.shape
        
        if n <= 1 or self.k >= min(n, d):
            return A
        
        # Step 1: Pre-sparsification to remove log n dependence (Theorem 2.4)
        A0 = self._pre_sparsify(A)
        
        if self.is_small_p:
            # Case 1 <= p < 2
            
            # Step 2: Get bicriteria subspace (Lemma 2.5)
            F0, r = self._bicriteria_subspace(A0)
            
            # Step 3: Compute low-rank and residual decomposition
            B, E, R = self._compute_residual_and_low_rank(A0, F0)
            
            # Step 4: Construct sampling distribution
            q = self._construct_sampling_distribution(B, E, R)
            
            # Step 5: Sample rows
            SA, _ = self._sample_rows(A0, q)
            
        else:
            # Case p > 2 (Section 6-7)
            
            # Set sampling threshold alpha based on target error
            log_factor = np.log(self.C_p * self.k / (self.epsilon * self.delta + 1e-10))
            alpha = self.epsilon**2 / (log_factor**3 + np.log(1/(self.delta + 1e-10)) + 1e-10)
            
            # One-round sampling (could be applied recursively)
            SA, _ = self._one_round_sampling_p_gt_2(A0, alpha)
            
            # For simplicity, we do one round
            # In practice, recursive application would be used
        
        return SA

# Helper functions for specific use cases

def pca_coreset_p_lt_2(A: np.ndarray, k: int, epsilon: float, p: float = 1.5) -> np.ndarray:
    """
    Construct coreset for ℓp subspace approximation (1 <= p < 2)
    
    Args:
        A: Input matrix (n, d)
        k: Rank of subspace
        epsilon: Approximation error
        p: ℓp norm (1 <= p < 2)
        
    Returns:
        Coreset matrix
    """
    coreset = LpSubspaceCoreset(p=p, k=k, epsilon=epsilon)
    return coreset.fit_transform(A)

def pca_coreset_p_gt_2(A: np.ndarray, k: int, epsilon: float, p: float = 3.0) -> np.ndarray:
    """
    Construct coreset for ℓp subspace approximation (p > 2)
    
    Args:
        A: Input matrix (n, d)
        k: Rank of subspace
        epsilon: Approximation error
        p: ℓp norm (p > 2)
        
    Returns:
        Coreset matrix
    """
    coreset = LpSubspaceCoreset(p=p, k=k, epsilon=epsilon)
    return coreset.fit_transform(A)

# Example usage and testing

def example_usage():
    """Example demonstrating the coreset construction"""
    np.random.seed(42)
    
    # Generate synthetic data
    n, d = 1000, 50
    k = 10
    
    # Data with low-rank structure plus noise
    U = np.random.randn(n, k)
    V = np.random.randn(d, k)
    noise = 0.1 * np.random.randn(n, d)
    A = U @ V.T + noise
    
    print(f"Original matrix shape: {A.shape}")
    print(f"k={k}, p<2 and p>2 tests")
    print("-" * 50)
    
    # Test for p < 2
    try:
        coreset_lt_2 = pca_coreset_p_lt_2(A, k=k, epsilon=0.1, p=1.5)
        print(f"Coreset shape (p < 2): {coreset_lt_2.shape}")
    except Exception as e:
        print(f"Error in p<2: {e}")
    
    # Test for p > 2
    try:
        coreset_gt_2 = pca_coreset_p_gt_2(A, k=k, epsilon=0.1, p=3.0)
        print(f"Coreset shape (p > 2): {coreset_gt_2.shape}")
    except Exception as e:
        print(f"Error in p>2: {e}")
    
    # Verify subspace cost preservation (simplified check)
    try:
        U_orig, S_orig, Vt_orig = svd(A, full_matrices=False)
        print(f"\nTop 5 singular values (original): {S_orig[:5]}")
        
        if 'coreset_lt_2' in locals():
            U_core, S_core, Vt_core = svd(coreset_lt_2, full_matrices=False)
            print(f"Top 5 singular values (coreset p<2): {S_core[:5]}")
    except Exception as e:
        print(f"Error in verification: {e}")
    
    return A

if __name__ == "__main__":
    example_usage()