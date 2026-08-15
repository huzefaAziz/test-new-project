import networkx as nx
import math
import random
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import functools
from datetime import datetime

@dataclass
class Quaternion:
    """Represents a quaternion a + bi + cj + dk"""
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    
    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)
        return Quaternion(self.a + other, self.b, self.c, self.d)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.a * other, self.b * other, self.c * other, self.d * other)
        
        # Quaternion multiplication: (a1 + b1i + c1j + d1k)(a2 + b2i + c2j + d2k)
        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = other.a, other.b, other.c, other.d
        
        return Quaternion(
            a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2
        )
    
    def __abs__(self):
        return math.sqrt(self.a**2 + self.b**2 + self.c**2 + self.d**2)
    
    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)
        return Quaternion(self.a - other, self.b, self.c, self.d)
    
    def norm(self):
        return abs(self)
    
    def conjugate(self):
        return Quaternion(self.a, -self.b, -self.c, -self.d)
    
    def __pow__(self, n):
        if n == 0:
            return Quaternion(1, 0, 0, 0)
        if n == 1:
            return self
        result = Quaternion(1, 0, 0, 0)
        for _ in range(n):
            result = result * self
        return result
    
    def __str__(self):
        return f"({self.a:.4f}, {self.b:.4f}, {self.c:.4f}, {self.d:.4f})"

class QuaternionicPolynomial:
    """Quaternionic polynomial in canonical generalized form"""
    
    def __init__(self, roots: List[Quaternion] = None, coefficients: List[Quaternion] = None):
        """
        Initialize polynomial either from roots or coefficients
        For canonical generalized form: P(q) = (q - α1)(q - α2)...(q - αn)
        """
        self.creation_time = time.time()
        self.last_access_time = time.time()
        self.access_count = 0
        
        if roots is not None:
            self.roots = roots
            self.coefficients = self._compute_coefficients_from_roots(roots)
        elif coefficients is not None:
            self.coefficients = coefficients
            self.roots = self._find_roots(coefficients)
        else:
            self.roots = []
            self.coefficients = [Quaternion(0, 0, 0, 0)]
        
        self.degree = len(self.coefficients) - 1
        self._norm_cache = None
        self._derivative_cache = None
    
    def _compute_coefficients_from_roots(self, roots: List[Quaternion]) -> List[Quaternion]:
        """Compute polynomial coefficients from roots using convolution product"""
        n = len(roots)
        coeffs = [Quaternion(1, 0, 0, 0)]
        
        for root in roots:
            # Multiply by (q - root) using convolution product
            new_coeffs = [Quaternion(0, 0, 0, 0)] * (len(coeffs) + 1)
            
            # First term: q * current polynomial
            for i, coeff in enumerate(coeffs):
                new_coeffs[i + 1] = new_coeffs[i + 1] + coeff
            
            # Second term: -root * current polynomial
            for i, coeff in enumerate(coeffs):
                new_coeffs[i] = new_coeffs[i] + (coeff * (-1) * root)
            
            coeffs = new_coeffs
        
        return coeffs
    
    def _find_roots(self, coeffs: List[Quaternion]) -> List[Quaternion]:
        """Find roots of quaternionic polynomial (simplified version)"""
        # This is a placeholder - root finding for quaternionic polynomials is complex
        # For caching purposes, we'll just return dummy roots
        return [Quaternion(1, 0, 0, 0) for _ in range(len(coeffs) - 1)]
    
    def evaluate(self, q: Quaternion) -> Quaternion:
        """Evaluate polynomial at quaternion q"""
        self.last_access_time = time.time()
        self.access_count += 1
        
        result = Quaternion(0, 0, 0, 0)
        for i, coeff in enumerate(reversed(self.coefficients)):
            result = result + coeff * (q ** i)
        return result
    
    def derivative(self):
        """Compute derivative of polynomial"""
        if self._derivative_cache is None:
            derivative_coeffs = []
            for i in range(1, len(self.coefficients)):
                derivative_coeffs.append(self.coefficients[i] * i)
            self._derivative_cache = QuaternionicPolynomial(coefficients=derivative_coeffs)
        return self._derivative_cache
    
    def norm(self) -> float:
        """Compute norm of polynomial on unit sphere"""
        if self._norm_cache is not None:
            return self._norm_cache
        
        max_val = 0.0
        # Sample points on unit sphere with increased resolution
        for theta in [0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6, math.pi, 
                      7*math.pi/6, 4*math.pi/3, 3*math.pi/2, 5*math.pi/3, 11*math.pi/6]:
            for phi in [0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6, math.pi,
                       7*math.pi/6, 4*math.pi/3, 3*math.pi/2, 5*math.pi/3, 11*math.pi/6]:
                q = Quaternion(
                    0,
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta)
                )
                val = abs(self.evaluate(q))
                max_val = max(max_val, val)
        
        self._norm_cache = max_val
        return max_val
    
    def get_age(self) -> float:
        """Get age of polynomial in seconds since creation"""
        return time.time() - self.creation_time
    
    def get_last_access_age(self) -> float:
        """Get time since last access in seconds"""
        return time.time() - self.last_access_time

class TimedCacheMemory:
    """Cache memory system with time-based management using Turán-type inequalities"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600, eviction_check_interval: int = 60):
        """
        Initialize cache with time-based parameters
        - max_size: Maximum number of items in cache
        - ttl: Time-to-live in seconds for cache entries
        - eviction_check_interval: How often to check for expired entries (seconds)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.eviction_check_interval = eviction_check_interval
        self.cache = {}
        self.access_count = {}
        self.priority_graph = nx.DiGraph()
        self._initialize_graph()
        self.access_history = []
        self.hit_count = 0
        self.miss_count = 0
        self.cache_creation_times = {}
        self.cache_last_access_times = {}
        self.last_eviction_check = time.time()
        self.total_evaluation_time = 0
        self.total_cache_time = 0
        self.stats_timestamps = []
        self.performance_history = []
    
    def _initialize_graph(self):
        """Initialize priority graph for cache management"""
        self.priority_graph.add_nodes_from(range(self.max_size))
        
        # Create a ring structure for priority ordering
        for i in range(self.max_size - 1):
            self.priority_graph.add_edge(i, i + 1)
        self.priority_graph.add_edge(self.max_size - 1, 0)
    
    def _get_key(self, q: Quaternion) -> str:
        """Generate cache key from quaternion"""
        return f"({q.a:.6f},{q.b:.6f},{q.c:.6f},{q.d:.6f})"
    
    def _should_cache(self, polynomial: QuaternionicPolynomial) -> bool:
        """
        Determine if polynomial should be cached based on Turán-type inequality
        Implements the caching criteria from the paper
        """
        n = polynomial.degree
        if n < 1:
            return True
        
        # Compute bounds from the paper
        derivative = polynomial.derivative()
        norm_p = polynomial.norm()
        norm_d = derivative.norm()
        
        # Check Turán-type inequality conditions
        # For degree 3: k >= 2^(1/3) ≈ 1.26
        if n == 3:
            # Compute k (radius bound of zeros)
            max_root_norm = max(abs(root) for root in polynomial.roots) if polynomial.roots else 0
            
            # Check condition from Theorem 3.1: k >= 2^(1/3)
            if max_root_norm >= 2**(1/3):
                # Turán inequality should hold, cache friendly
                cache_score = norm_d / (n/(1 + max_root_norm**n) * norm_p)
                return cache_score >= 0.8
        
        # For degree 4: check k >= 3^(1/4) ≈ 1.316
        elif n == 4:
            max_root_norm = max(abs(root) for root in polynomial.roots) if polynomial.roots else 0
            if max_root_norm >= 3**(1/4):
                cache_score = norm_d / (n/(1 + max_root_norm**n) * norm_p)
                return cache_score >= 0.8
        
        # Default caching policy based on derivative bound efficiency
        if norm_p > 0:
            efficiency = norm_d / (n * norm_p)
            return efficiency >= 0.6
        
        return True
    
    def _check_and_evict_expired(self):
        """Check for expired cache entries and remove them"""
        current_time = time.time()
        
        # Only check periodically to avoid overhead
        if current_time - self.last_eviction_check < self.eviction_check_interval:
            return
        
        expired_keys = []
        for key in self.cache.keys():
            if key in self.cache_creation_times:
                age = current_time - self.cache_creation_times[key]
                last_access_age = current_time - self.cache_last_access_times.get(key, 0)
                
                # Evict if too old or not accessed recently
                if age > self.ttl or last_access_age > self.ttl * 2:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            del self.cache_creation_times[key]
            if key in self.cache_last_access_times:
                del self.cache_last_access_times[key]
            if key in self.access_count:
                del self.access_count[key]
            if key in self.priority_graph.nodes:
                self.priority_graph.remove_node(key)
        
        self.last_eviction_check = current_time
    
    def _update_priority_graph(self, key: str):
        """Update priority graph based on access pattern"""
        if key in self.cache:
            # Update access time
            self.cache_last_access_times[key] = time.time()
            
            # Move key to higher priority
            nodes = list(self.priority_graph.nodes())
            if len(nodes) > 1 and key in nodes:
                try:
                    current_pos = nodes.index(key)
                    # Swap with previous node to increase priority
                    if current_pos > 0:
                        self.priority_graph.remove_edge(nodes[current_pos - 1], nodes[current_pos])
                        self.priority_graph.add_edge(key, nodes[current_pos - 1])
                    elif current_pos == 0 and len(nodes) > 1:
                        self.priority_graph.remove_edge(nodes[-1], nodes[0])
                        self.priority_graph.add_edge(key, nodes[-1])
                except (ValueError, nx.NetworkXError):
                    # Key not in graph properly, re-add it
                    if key in self.priority_graph.nodes:
                        self.priority_graph.remove_node(key)
                    self.priority_graph.add_node(key)
                    if nodes:
                        self.priority_graph.add_edge(key, nodes[0])
    
    def get(self, q: Quaternion) -> Optional[Quaternion]:
        """Retrieve value from cache with priority-based access and time tracking"""
        start_time = time.time()
        key = self._get_key(q)
        
        # Check for expired entries
        self._check_and_evict_expired()
        
        if key in self.cache:
            self.hit_count += 1
            self.access_history.append(('hit', key, time.time()))
            self._update_priority_graph(key)
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.cache_last_access_times[key] = time.time()
            
            # Track performance
            cache_time = time.time() - start_time
            self.total_cache_time += cache_time
            
            return self.cache[key]
        else:
            self.miss_count += 1
            self.access_history.append(('miss', key, time.time()))
            return None
    
    def put(self, q: Quaternion, polynomial: QuaternionicPolynomial) -> bool:
        """Store quaternionic polynomial evaluation in cache using Turán-based criteria"""
        start_time = time.time()
        
        # Check for expired entries before adding
        self._check_and_evict_expired()
        
        if len(self.cache) >= self.max_size:
            # Need eviction - use priority graph for LRU-like eviction
            nodes = list(self.priority_graph.nodes())
            if nodes:
                # Find node with lowest priority (least accessed and oldest)
                min_score = float('inf')
                evict_key = None
                for node in nodes:
                    if node in self.cache:
                        access_count = self.access_count.get(node, 0)
                        last_access = self.cache_last_access_times.get(node, 0)
                        age = time.time() - last_access if last_access > 0 else 0
                        
                        # Score combines access frequency and age
                        score = access_count * 0.3 + age * 0.7
                        if score < min_score:
                            min_score = score
                            evict_key = node
                
                if evict_key:
                    # Update performance history
                    if evict_key in self.cache_creation_times:
                        lifespan = time.time() - self.cache_creation_times[evict_key]
                        self.performance_history.append({
                            'evicted': evict_key,
                            'lifespan': lifespan,
                            'access_count': self.access_count.get(evict_key, 0)
                        })
                    
                    del self.cache[evict_key]
                    if evict_key in self.cache_creation_times:
                        del self.cache_creation_times[evict_key]
                    if evict_key in self.cache_last_access_times:
                        del self.cache_last_access_times[evict_key]
                    if evict_key in self.access_count:
                        del self.access_count[evict_key]
                    if evict_key in self.priority_graph.nodes:
                        self.priority_graph.remove_node(evict_key)
        
        key = self._get_key(q)
        value = polynomial.evaluate(q)
        
        # Check if we should cache this polynomial
        if self._should_cache(polynomial):
            self.cache[key] = value
            self.access_count[key] = 0
            self.cache_creation_times[key] = time.time()
            self.cache_last_access_times[key] = time.time()
            
            # Add to priority graph
            nodes = list(self.priority_graph.nodes())
            if nodes:
                self.priority_graph.add_node(key)
                self.priority_graph.add_edge(key, nodes[0])
            else:
                self.priority_graph.add_node(key)
            
            # Track performance
            self.total_evaluation_time += time.time() - start_time
            
            return True
        return False
    
    def compute_turan_bound(self, polynomial: QuaternionicPolynomial, k: float) -> Tuple[float, float]:
        """
        Compute Turán-type bound for polynomial derivative
        Implements Theorems 3.1, 3.2, 3.3 from the paper
        """
        n = polynomial.degree
        derivative = polynomial.derivative()
        norm_p = polynomial.norm()
        norm_d = derivative.norm()
        
        if n == 1:
            # Theorem 3.1: For degree 3 polynomials
            if k >= 2**(1/3):
                bound = n / (1 + k**n)
                return norm_d, bound * norm_p
        elif n == 3:
            # Theorem 3.2: For degree 3 polynomials with sum of roots = 0
            if abs(sum(root.a for root in polynomial.roots)) < 1e-10:
                A_k = self._compute_A(k)
                if any(abs(root) <= A_k for root in polynomial.roots):
                    bound = n / (1 + k**n)
                    return norm_d, bound * norm_p
        elif n == 4:
            # Theorem 3.3: For degree 4 polynomials
            if len(polynomial.roots) == 4:
                roots = polynomial.roots
                # Check conditions from Theorem 3.3
                sum1 = sum(root.a for root in roots)
                sum2 = sum(roots[i].a * roots[j].a for i in range(len(roots)) for j in range(i+1, len(roots)))
                
                if abs(sum1) < 1e-10 and abs(sum2) < 1e-10:
                    if k >= 3**(1/4):
                        bound = n / (1 + k**n)
                        return norm_d, bound * norm_p
        
        # General case
        return norm_d, norm_d
    
    def _compute_A(self, k: float) -> float:
        """
        Compute A(k) from Theorem 3.2
        A(k) = [-3k^2 + sqrt(9k^4 - 4(2-k^3)((2-k^3)k^2 - 3k^3))] / [2(2-k^3)]
        """
        if abs(2 - k**3) < 1e-10:
            return k  # A(k) = k when k = 2^(1/3)
        
        discriminant = 9*k**4 - 4*(2 - k**3)*((2 - k**3)*k**2 - 3*k**3)
        if discriminant < 0:
            return 0
        
        numerator = -3*k**2 + math.sqrt(discriminant)
        denominator = 2*(2 - k**3)
        return numerator / denominator
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics with time measurements"""
        total_requests = self.hit_count + self.miss_count
        current_time = time.time()
        
        # Calculate average lifespan of evicted items
        if self.performance_history:
            avg_lifespan = sum(d['lifespan'] for d in self.performance_history) / len(self.performance_history)
            avg_access = sum(d['access_count'] for d in self.performance_history) / len(self.performance_history)
        else:
            avg_lifespan = avg_access = 0
        
        # Calculate current cache age
        cache_ages = []
        for key in self.cache_creation_times:
            age = current_time - self.cache_creation_times[key]
            cache_ages.append(age)
        
        avg_cache_age = sum(cache_ages) / len(cache_ages) if cache_ages else 0
        max_cache_age = max(cache_ages) if cache_ages else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': self.hit_count / total_requests if total_requests > 0 else 0,
            'total_requests': total_requests,
            'access_distribution': dict(sorted(self.access_count.items(), key=lambda x: x[1], reverse=True)[:5]),
            'total_evaluation_time': self.total_evaluation_time,
            'total_cache_time': self.total_cache_time,
            'avg_cache_lifespan': avg_lifespan,
            'avg_cache_accesses': avg_access,
            'avg_cache_age': avg_cache_age,
            'max_cache_age': max_cache_age,
            'cache_ttl': self.ttl,
            'performance_metrics': {
                'evaluation_time_ratio': self.total_evaluation_time / (self.total_evaluation_time + self.total_cache_time) if (self.total_evaluation_time + self.total_cache_time) > 0 else 0,
                'cache_efficiency': self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0
            },
            'timestamp': datetime.now().isoformat()
        }

class TimedTuranCachingAlgorithm:
    """
    Efficient caching algorithm based on Turán-type inequalities with time tracking
    """
    
    def __init__(self, max_cache_size: int = 1000, ttl: int = 3600):
        self.cache_memory = TimedCacheMemory(max_size=max_cache_size, ttl=ttl)
        self.polynomial_cache = {}
        self.performance_data = []
        self.start_time = time.time()
        self.total_operations = 0
        self.operation_times = []
    
    def evaluate_polynomial(self, polynomial: QuaternionicPolynomial, q: Quaternion) -> Quaternion:
        """
        Evaluate polynomial with caching using Turán-type inequality optimization
        """
        self.total_operations += 1
        start_time = time.time()
        
        key = f"poly_{id(polynomial)}_{self.cache_memory._get_key(q)}"
        
        # Check if result is in cache
        cached_result = self.cache_memory.get(q)
        if cached_result is not None:
            cache_time = time.time() - start_time
            self.operation_times.append(('cache_hit', cache_time))
            return cached_result
        
        # Compute polynomial evaluation
        eval_start = time.time()
        result = polynomial.evaluate(q)
        eval_time = time.time() - eval_start
        
        # Cache the result if beneficial
        cache_decision_start = time.time()
        self.cache_memory.put(q, polynomial)
        decision_time = time.time() - cache_decision_start
        
        total_time = time.time() - start_time
        self.operation_times.append(('cache_miss', total_time))
        
        return result
    
    def optimize_derivative_bounds(self, polynomial: QuaternionicPolynomial) -> Tuple[float, float]:
        """
        Compute and cache derivative bounds using Turán inequalities with time tracking
        """
        start_time = time.time()
        
        # Check if polynomial is cached
        poly_key = id(polynomial)
        if poly_key in self.polynomial_cache:
            return self.polynomial_cache[poly_key]
        
        # Compute root radius bound
        max_root_norm = max(abs(root) for root in polynomial.roots) if polynomial.roots else 0
        k = max(1.0, max_root_norm)
        
        # Compute Turán bound from the paper
        derivative = polynomial.derivative()
        norm_d = derivative.norm()
        
        # Use the derived bounds
        n = polynomial.degree
        if n >= 1 and k >= 1:
            turan_bound = n / (1 + k**n) * polynomial.norm()
        else:
            turan_bound = norm_d
        
        self.polynomial_cache[poly_key] = (norm_d, turan_bound)
        
        # Update performance data with timing
        computation_time = time.time() - start_time
        self.performance_data.append({
            'degree': n,
            'k': k,
            'norm_d': norm_d,
            'turan_bound': turan_bound,
            'ratio': norm_d / turan_bound if turan_bound > 0 else float('inf'),
            'computation_time': computation_time,
            'timestamp': datetime.now().isoformat()
        })
        
        return norm_d, turan_bound
    
    def get_caching_decision(self, polynomial: QuaternionicPolynomial) -> bool:
        """
        Decide whether to cache polynomial based on Turán-type analysis
        """
        n = polynomial.degree
        max_root_norm = max(abs(root) for root in polynomial.roots) if polynomial.roots else 0
        k = max(1.0, max_root_norm)
        
        # Apply Theorems from paper
        if n == 3:
            # Theorem 3.1: k >= 2^(1/3)
            if k >= 2**(1/3):
                return True
            # Theorem 3.2: Check if sum of roots = 0 and condition (7)
            if len(polynomial.roots) == 3:
                roots = polynomial.roots
                if abs(sum(root.a for root in roots)) < 1e-10:
                    A_k = self.cache_memory._compute_A(k)
                    if any(abs(root) <= A_k for root in roots):
                        return True
        
        elif n == 4:
            # Theorem 3.3
            if len(polynomial.roots) == 4:
                roots = polynomial.roots
                sum1 = sum(root.a for root in roots)
                sum2 = sum(roots[i].a * roots[j].a for i in range(len(roots)) for j in range(i+1, len(roots)))
                
                if abs(sum1) < 1e-10 and abs(sum2) < 1e-10:
                    if k >= 3**(1/4):
                        return True
        
        # Default: cache if derivative bound is efficient
        derivative = polynomial.derivative()
        norm_p = polynomial.norm()
        norm_d = derivative.norm()
        if norm_p > 0:
            efficiency = norm_d / (n * norm_p)
            return efficiency >= 0.6
        
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of caching algorithm with time metrics"""
        cache_stats = self.cache_memory.get_statistics()
        
        if self.performance_data:
            ratios = [d['ratio'] for d in self.performance_data if d['ratio'] != float('inf')]
            avg_ratio = sum(ratios) / len(ratios) if ratios else 0
            max_ratio = max(ratios) if ratios else 0
            min_ratio = min(ratios) if ratios else 0
            
            avg_time = sum(d['computation_time'] for d in self.performance_data) / len(self.performance_data)
            total_time = sum(d['computation_time'] for d in self.performance_data)
        else:
            avg_ratio = max_ratio = min_ratio = 0
            avg_time = total_time = 0
        
        # Calculate operation time statistics
        if self.operation_times:
            hit_times = [t for op, t in self.operation_times if op == 'cache_hit']
            miss_times = [t for op, t in self.operation_times if op == 'cache_miss']
            avg_hit_time = sum(hit_times) / len(hit_times) if hit_times else 0
            avg_miss_time = sum(miss_times) / len(miss_times) if miss_times else 0
        else:
            avg_hit_time = avg_miss_time = 0
        
        runtime = time.time() - self.start_time
        
        return {
            'cache_statistics': cache_stats,
            'polynomial_processed': len(self.performance_data),
            'derivative_bounds': {
                'average_ratio': avg_ratio,
                'max_ratio': max_ratio,
                'min_ratio': min_ratio,
                'average_computation_time': avg_time,
                'total_computation_time': total_time
            },
            'operation_times': {
                'average_hit_time': avg_hit_time,
                'average_miss_time': avg_miss_time,
                'total_operations': self.total_operations
            },
            'runtime_seconds': runtime,
            'cache_hit_rate': cache_stats['hit_rate'],
            'cache_efficiency': cache_stats['performance_metrics']['cache_efficiency'],
            'timestamp': datetime.now().isoformat()
        }

# Example usage and testing
def test_timed_turan_caching():
    """Test the timed Turán caching algorithm with example polynomials"""
    
    print("=" * 60)
    print("TESTING TIMED TURÁN CACHING ALGORITHM")
    print("=" * 60)
    
    # Create sample quaternions
    q1 = Quaternion(0.5, 0.3, 0.2, 0.1)
    q2 = Quaternion(0.2, 0.5, 0.3, 0.4)
    q3 = Quaternion(0.1, 0.2, 0.4, 0.3)
    
    # Create polynomials with roots
    roots = [q1, q2, q3]
    poly3 = QuaternionicPolynomial(roots=roots)  # Degree 3
    
    # Test caching algorithm with time tracking
    caching_alg = TimedTuranCachingAlgorithm(max_cache_size=100, ttl=60)
    
    print(f"Created polynomial of degree {poly3.degree}")
    print(f"Polynomial age: {poly3.get_age():.4f} seconds")
    
    # Test multiple evaluations with time tracking
    test_points = [
        Quaternion(0.3, 0.1, 0.2, 0.5),
        Quaternion(0.7, 0.4, 0.1, 0.3),
        Quaternion(0.1, 0.8, 0.2, 0.4),
        Quaternion(0.5, 0.3, 0.6, 0.2),
        Quaternion(0.9, 0.2, 0.3, 0.1)
    ]
    
    print("\nPerforming evaluations with caching...")
    for i, q in enumerate(test_points):
        start_time = time.time()
        result = caching_alg.evaluate_polynomial(poly3, q)
        elapsed = time.time() - start_time
        print(f"Evaluation {i+1} at {q}: {result}")
        print(f"  Time: {elapsed:.6f} seconds")
    
    # Optimize derivative bounds
    print("\nComputing derivative bounds...")
    start_time = time.time()
    norm_d, bound = caching_alg.optimize_derivative_bounds(poly3)
    elapsed = time.time() - start_time
    print(f"Derivative norm: {norm_d:.4f}")
    print(f"Turán bound: {bound:.4f}")
    print(f"Efficiency ratio: {norm_d/bound:.4f}" if bound > 0 else "Bound is zero")
    print(f"Computation time: {elapsed:.6f} seconds")
    
    # Get caching decision
    should_cache = caching_alg.get_caching_decision(poly3)
    print(f"\nShould cache polynomial: {should_cache}")
    
    # Get performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    summary = caching_alg.get_performance_summary()
    
    print(f"Cache hit rate: {summary['cache_statistics']['hit_rate']:.2%}")
    print(f"Cache efficiency: {summary['cache_efficiency']:.2%}")
    print(f"Polynomials processed: {summary['polynomial_processed']}")
    print(f"Total operations: {summary['operation_times']['total_operations']}")
    print(f"Average hit time: {summary['operation_times']['average_hit_time']:.6f} seconds")
    print(f"Average miss time: {summary['operation_times']['average_miss_time']:.6f} seconds")
    print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")
    
    print(f"\nCache Statistics:")
    print(f"  Size: {summary['cache_statistics']['size']}/{summary['cache_statistics']['max_size']}")
    print(f"  Hit count: {summary['cache_statistics']['hit_count']}")
    print(f"  Miss count: {summary['cache_statistics']['miss_count']}")
    print(f"  Average cache lifespan: {summary['cache_statistics']['avg_cache_lifespan']:.2f} seconds")
    print(f"  Average cache age: {summary['cache_statistics']['avg_cache_age']:.2f} seconds")
    print(f"  Max cache age: {summary['cache_statistics']['max_cache_age']:.2f} seconds")
    
    print(f"\nDerivative Bounds:")
    print(f"  Average ratio: {summary['derivative_bounds']['average_ratio']:.4f}")
    print(f"  Average computation time: {summary['derivative_bounds']['average_computation_time']:.6f} seconds")
    
    # Test TTL eviction
    print("\n" + "=" * 60)
    print("TESTING TTL EVICTION")
    print("=" * 60)
    
    # Create a cache with short TTL
    test_cache = TimedCacheMemory(max_size=10, ttl=2, eviction_check_interval=1)
    
    # Add some entries
    for i in range(5):
        q = Quaternion(i * 0.1, i * 0.2, i * 0.3, i * 0.4)
        poly = QuaternionicPolynomial(roots=[Quaternion(i, i, i, i)])
        test_cache.put(q, poly)
    
    print(f"Initial cache size: {len(test_cache.cache)}")
    
    # Wait for TTL to expire
    print("Waiting 3 seconds for TTL to expire...")
    time.sleep(3)
    
    # Check for expired entries
    test_cache._check_and_evict_expired()
    print(f"Cache size after TTL eviction: {len(test_cache.cache)}")
    
    # Add more entries to test size limit
    print("\nTesting size limit eviction...")
    for i in range(10):
        q = Quaternion(i * 0.1, i * 0.2, i * 0.3, i * 0.4)
        poly = QuaternionicPolynomial(roots=[Quaternion(i, i, i, i)])
        test_cache.put(q, poly)
    
    print(f"Cache size after adding items: {len(test_cache.cache)}")
    
    # Get final statistics
    stats = test_cache.get_statistics()
    print(f"Total evaluations: {stats['total_evaluation_time']:.6f} seconds")
    print(f"Total cache time: {stats['total_cache_time']:.6f} seconds")
    
    print("\nTest completed successfully!")
    return caching_alg

# Run test
if __name__ == "__main__":
    test_timed_turan_caching()