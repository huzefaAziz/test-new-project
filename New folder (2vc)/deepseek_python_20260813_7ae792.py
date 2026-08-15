import networkx as nx
import math
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple, Optional, Set
import heapq
from datetime import datetime, timedelta

class TimeAwareTopologicalCache:
    """
    A cache memory algorithm with time-aware topological persistence.
    
    Extends the topological cache with:
    - Temporal local flatness: access patterns over time
    - Time-stamped topological signatures
    - Aging based on temporal distance
    - Predictive topological pre-fetching
    """
    
    def __init__(self, capacity: int, 
                 topological_radius: float = 0.5,
                 persistence_threshold: int = 3,
                 time_window: int = 3600,  # 1 hour in seconds
                 aging_factor: float = 0.95,
                 predictive_window: int = 10):
        """
        Initialize the time-aware topological cache.
        
        Args:
            capacity: Maximum number of items in cache
            topological_radius: Radius for topological neighborhood analysis
            persistence_threshold: Minimum accesses before considering persistent
            time_window: Time window for considering access patterns (seconds)
            aging_factor: Factor for aging old access patterns
            predictive_window: Number of future accesses to predict
        """
        self.capacity = capacity
        self.topological_radius = topological_radius
        self.persistence_threshold = persistence_threshold
        self.time_window = time_window
        self.aging_factor = aging_factor
        self.predictive_window = predictive_window
        
        # Core data structures with time awareness
        self.cache = {}  # key -> (data, access_history)
        self.access_graph = nx.Graph()  # Topological access graph
        self.topological_classes = defaultdict(set)  # key -> topological class
        self.persistence_scores = defaultdict(float)  # key -> persistence score
        self.access_timestamps = {}  # key -> last access time
        self.access_count = 0
        
        # Time-specific structures
        self.time_series = defaultdict(lambda: deque(maxlen=1000))  # key -> access times
        self.temporal_patterns = defaultdict(lambda: deque(maxlen=100))  # key -> pattern signatures
        self.access_frequencies = defaultdict(float)  # key -> frequency per time unit
        self.last_access_time = time.time()
        
        # For predictive caching
        self.access_sequence = deque(maxlen=1000)  # Recent access sequence
        self.transition_graph = nx.DiGraph()  # Key transitions over time
        self.transition_probabilities = defaultdict(float)  # Transition weights
        
        # For Milnor fibration analysis over time
        self.temporal_branches = defaultdict(set)  # Time-based branches
        self.temporal_links = defaultdict(set)  # Temporal link components
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.time_evictions = 0
        self.predictive_hits = 0
        
        # Performance tracking
        self.access_times = deque(maxlen=1000)  # Response times
        
        # Data source for prefetching (optional)
        self._data_source = None
    
    def set_data_source(self, data_source):
        """Set a data source for prefetching."""
        self._data_source = data_source
    
    def _get_time_bucket(self, timestamp: float = None) -> int:
        """
        Get time bucket for temporal analysis.
        """
        if timestamp is None:
            timestamp = time.time()
        return int(timestamp / self.time_window)  # Bucket by time window
    
    def _compute_temporal_signature(self, key: Any, data: Any, timestamp: float = None) -> str:
        """
        Compute temporal signature combining topological and time information.
        """
        # Get topological signature
        topo_sig = self._compute_topological_signature(key, data)
        
        # Add time component
        if timestamp is None:
            timestamp = time.time()
        
        time_bucket = self._get_time_bucket(timestamp)
        
        # Frequency component
        freq = self.access_frequencies.get(key, 0)
        freq_bucket = int(freq * 10) / 10.0
        
        return f"{topo_sig}_T{time_bucket}_F{freq_bucket}"
    
    def _compute_topological_signature(self, key: Any, data: Any) -> str:
        """
        Compute topological signature using local flatness concept.
        Enhanced with data size and complexity measures.
        """
        # Get base signature
        if isinstance(data, (int, float)):
            # For numerical data, use threshold-based topology
            return f"num_{math.floor(data / self.topological_radius)}"
        elif isinstance(data, str):
            # For strings, use length and first character topology
            return f"str_{len(data)}_{data[0] if data else 'empty'}"
        elif isinstance(data, (list, tuple)):
            # For sequences, use topology of values
            if len(data) > 0:
                sig = "_".join(str(type(x).__name__) for x in data[:3])
                return f"seq_{len(data)}_{sig}"
        elif isinstance(data, dict):
            # For dicts, use topological signature of keys
            keys = sorted(data.keys())[:3]
            sig = "_".join(str(k) for k in keys)
            return f"dict_{len(data)}_{sig}"
        
        # Default: use type and length
        return f"{type(data).__name__}_{len(str(data))}"
    
    def _topological_distance(self, sig1: str, sig2: str) -> float:
        """
        Compute topological distance between two signatures.
        Inspired by Milnor fibration topology.
        """
        parts1 = sig1.split('_')
        parts2 = sig2.split('_')
        
        # If types differ, distance is large
        if parts1[0] != parts2[0]:
            return 2.0
        
        # Compare lengths and other properties
        dist = 0.0
        for i in range(min(len(parts1), len(parts2))):
            try:
                # Try numeric comparison
                d = abs(float(parts1[i]) - float(parts2[i]))
                dist += d
            except:
                # String comparison
                if parts1[i] != parts2[i]:
                    dist += 0.5
        
        return dist / max(len(parts1), len(parts2))
    
    def _compute_topological_stability(self, key: Any) -> float:
        """
        Compute topological stability using local flatness concept.
        In the paper, local flatness is an obstruction to degeneracy.
        """
        if key not in self.cache:
            return 0.0
        
        data, history = self.cache[key]
        
        if len(history) < 2:
            return 1.0
        
        # Check if access patterns form a locally flat structure
        # by computing variation of access intervals
        intervals = [history[i+1] - history[i] for i in range(len(history)-1)]
        if not intervals:
            return 1.0
        
        avg_interval = sum(intervals) / len(intervals)
        if avg_interval == 0:
            return 1.0
        
        # Less variation = more stable = higher score
        variation = sum(abs(i - avg_interval) for i in intervals) / len(intervals)
        stability = 1.0 / (1.0 + variation / (avg_interval + 1))
        
        # Check if the data itself has stable topology
        topo_sig = self._compute_topological_signature(key, data)
        
        # If the signature appears in many different contexts, it's less stable
        context_count = sum(1 for sig in self.topological_classes if sig == topo_sig)
        context_stability = 1.0 / (1.0 + math.log(context_count + 1))
        
        # Check temporal stability
        if key in self.time_series:
            times = list(self.time_series[key])
            if len(times) >= 2:
                time_intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
                if time_intervals:
                    avg_time_interval = sum(time_intervals) / len(time_intervals)
                    if avg_time_interval > 0:
                        time_variation = sum(abs(t - avg_time_interval) for t in time_intervals) / len(time_intervals)
                        time_stability = 1.0 / (1.0 + time_variation / (avg_time_interval + 1))
                    else:
                        time_stability = 1.0
                else:
                    time_stability = 1.0
            else:
                time_stability = 1.0
        else:
            time_stability = 1.0
        
        return stability * context_stability * time_stability
    
    def _compute_temporal_persistence(self, key: Any) -> float:
        """
        Compute temporal persistence score.
        Items that are accessed consistently over time have higher scores.
        """
        if key not in self.cache:
            return 0.0
        
        data, history = self.cache[key]
        current_time = time.time()
        
        # Get access times for this key
        access_times = list(self.time_series.get(key, []))
        if not access_times:
            return 0.0
        
        # Only consider accesses within time window
        recent_accesses = [t for t in access_times if current_time - t <= self.time_window]
        if not recent_accesses:
            return 0.0
        
        # Compute temporal density
        time_span = current_time - min(recent_accesses) if recent_accesses else 1
        density = len(recent_accesses) / max(time_span, 1)
        
        # Compute regularity (low variance in intervals)
        if len(recent_accesses) >= 2:
            intervals = [recent_accesses[i+1] - recent_accesses[i] 
                        for i in range(len(recent_accesses)-1)]
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                if avg_interval > 0:
                    variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
                    regularity = 1.0 / (1.0 + math.sqrt(variance) / avg_interval)
                else:
                    regularity = 1.0
            else:
                regularity = 1.0
        else:
            regularity = 1.0
        
        # Compute recency factor
        last_access = max(recent_accesses)
        recency = current_time - last_access
        recency_factor = 1.0 / (1.0 + math.log(recency + 1))
        
        # Combine factors
        persistence = density * regularity * recency_factor
        
        # Add topological persistence component
        topo_score = self._compute_topological_stability(key)
        persistence *= (1.0 + topo_score)
        
        # Age factor - older but regular patterns get slight boost
        age = current_time - min(recent_accesses)
        age_factor = math.exp(-age / (self.time_window * 2))
        persistence *= (1.0 + age_factor * 0.5)
        
        return persistence
    
    def _compute_temporal_flatness(self, key: Any) -> float:
        """
        Compute temporal local flatness.
        In the paper, local flatness determines if a hypersurface is well-behaved.
        """
        if key not in self.cache:
            return 0.0
        
        access_times = list(self.time_series.get(key, []))
        if len(access_times) < 2:
            return 1.0
        
        current_time = time.time()
        recent_times = [t for t in access_times if current_time - t <= self.time_window]
        
        if len(recent_times) < 2:
            return 1.0
        
        # Check if access pattern is locally flat (regular intervals)
        intervals = [recent_times[i+1] - recent_times[i] 
                    for i in range(len(recent_times)-1)]
        
        if not intervals:
            return 1.0
        
        # Low variation indicates local flatness
        avg_interval = sum(intervals) / len(intervals)
        if avg_interval == 0:
            return 1.0
        
        variation = sum(abs(i - avg_interval) for i in intervals) / len(intervals)
        flatness = 1.0 / (1.0 + variation / (avg_interval + 1))
        
        return flatness
    
    def _update_time_series(self, key: Any, timestamp: float = None):
        """
        Update time series data for a key.
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Update time series
        self.time_series[key].append(timestamp)
        
        # Update access frequency
        time_span = max(1, timestamp - self.last_access_time)
        self.access_frequencies[key] = len(self.time_series[key]) / time_span
        
        # Update temporal pattern
        if key in self.cache:
            data, _ = self.cache[key]
            pattern = self._compute_temporal_signature(key, data, timestamp)
            self.temporal_patterns[key].append(pattern)
        
        # Update access sequence for predictive analysis
        self.access_sequence.append((key, timestamp))
        if len(self.access_sequence) >= 2:
            prev_key, prev_time = self.access_sequence[-2]
            if prev_key != key:
                # Update transition graph
                if self.transition_graph.has_edge(prev_key, key):
                    self.transition_graph[prev_key][key]['weight'] += 1
                else:
                    self.transition_graph.add_edge(prev_key, key, weight=1)
                
                # Update transition probabilities
                if prev_key in self.transition_graph:
                    total_transitions = sum(self.transition_graph[prev_key][k]['weight'] 
                                           for k in self.transition_graph[prev_key])
                    if total_transitions > 0:
                        self.transition_probabilities[(prev_key, key)] = (
                            self.transition_graph[prev_key][key]['weight'] / total_transitions
                        )
    
    def _predict_next_accesses(self) -> List[Tuple[Any, float]]:
        """
        Predict future accesses based on temporal patterns.
        Uses Markov chain on transition graph with temporal weighting.
        """
        if len(self.access_sequence) == 0:
            return []
        
        current_key = self.access_sequence[-1][0]
        
        # Get possible next keys
        if current_key not in self.transition_graph:
            return []
        
        # Get transition probabilities
        predictions = []
        total_weight = 0
        
        for next_key in self.transition_graph[current_key]:
            weight = self.transition_graph[current_key][next_key]['weight']
            
            # Apply temporal decay factor
            timestamp = time.time()
            if next_key in self.time_series:
                recent_accesses = [t for t in self.time_series[next_key] 
                                 if timestamp - t <= self.time_window]
                if recent_accesses:
                    recency = max(recent_accesses)
                    decay = math.exp(-(timestamp - recency) / self.time_window)
                    weight *= (1.0 + decay)
            
            predictions.append((next_key, weight))
            total_weight += weight
        
        # Normalize probabilities
        if total_weight > 0:
            predictions = [(k, w / total_weight) for k, w in predictions]
        
        # Sort by probability
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:self.predictive_window]
    
    def _prefetch_predictions(self):
        """
        Prefetch predicted items based on temporal patterns.
        """
        predictions = self._predict_next_accesses()
        
        for key, probability in predictions:
            if probability > 0.3 and key not in self.cache:
                # Check if we can pre-load this item
                if self._data_source is not None:
                    try:
                        data = self._data_source.get(key)
                        if data is not None:
                            # Add with initial access count and timestamp
                            current_time = time.time()
                            self.cache[key] = (data, [self.access_count])
                            self.time_series[key].append(current_time)
                            self.access_timestamps[key] = current_time
                            self.predictive_hits += 1
                    except:
                        pass
    
    def _evict_with_time_awareness(self):
        """
        Evict items using time-aware topological persistence.
        """
        if len(self.cache) < self.capacity:
            return
        
        current_time = time.time()
        
        # Compute scores considering both persistence and time
        scores = []
        for key in self.cache:
            # Base persistence score
            persistence = self._compute_temporal_persistence(key)
            
            # Time-aware adjustments
            times = self.time_series.get(key, [0])
            last_access = max(times) if times else 0
            time_since_access = current_time - last_access
            
            # Age penalty (older items are more likely to be evicted)
            age_penalty = math.exp(-time_since_access / self.time_window)
            
            # Temporal flatness bonus
            flatness = self._compute_temporal_flatness(key)
            
            # Combine scores
            score = persistence * (0.5 + 0.5 * age_penalty) * (1.0 + flatness)
            
            scores.append((score, key))
        
        # Sort by score (ascending) and remove lowest
        scores.sort()
        
        # Remove items with lowest scores, but protect topologically essential items
        removed_count = 0
        target_removal = len(scores) // 3 + 1  # Remove 1/3 + 1
        
        for score, key in scores:
            if removed_count >= target_removal:
                break
            
            # Check if topologically essential
            if self._is_topologically_essential(key):
                continue
            
            # Remove from cache
            data, history = self.cache.pop(key)
            self.access_timestamps.pop(key, None)
            self.time_series.pop(key, None)
            
            # Update graph
            if key in self.access_graph:
                self.access_graph.remove_node(key)
            
            # Remove from topological classes
            sig = self._compute_topological_signature(key, data)
            if sig in self.topological_classes:
                self.topological_classes[sig].discard(key)
            
            removed_count += 1
            self.time_evictions += 1
    
    def _update_topological_graph_with_time(self, key: Any, data: Any, 
                                           access_pattern: List[Any], timestamp: float = None):
        """
        Update the topological access graph with time awareness.
        """
        if timestamp is None:
            timestamp = time.time()
        
        sig = self._compute_temporal_signature(key, data, timestamp)
        
        # Add to temporal branches
        time_bucket = self._get_time_bucket(timestamp)
        self.temporal_branches[time_bucket].add(sig)
        
        # Add to topological class
        self.topological_classes[sig].add(key)
        
        # Update access graph with time-weighted edges
        for other_sig in list(self.topological_classes.keys()):
            if other_sig != sig:
                distance = self._topological_distance(sig, other_sig)
                time_distance = abs(self._get_time_bucket_from_sig(sig) - 
                                   self._get_time_bucket_from_sig(other_sig))
                
                # Time-aware distance
                weighted_distance = distance * (1.0 + time_distance * 0.1)
                
                if weighted_distance < self.topological_radius:
                    for k1 in self.topological_classes[sig]:
                        for k2 in self.topological_classes[other_sig]:
                            if k1 != k2:
                                # Add time-weighted edge
                                weight = self.topological_radius / (weighted_distance + 0.1)
                                if self.access_graph.has_edge(k1, k2):
                                    # Update weight with time decay
                                    new_weight = (self.access_graph[k1][k2]['weight'] + weight) / 2
                                    self.access_graph[k1][k2]['weight'] = new_weight
                                else:
                                    self.access_graph.add_edge(k1, k2, weight=weight)
    
    def _get_time_bucket_from_sig(self, sig: str) -> int:
        """
        Extract time bucket from temporal signature.
        """
        parts = sig.split('_')
        for part in parts:
            if part.startswith('T'):
                try:
                    return int(part[1:])
                except:
                    pass
        return 0
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve an item from cache with time tracking.
        """
        self.access_count += 1
        current_time = time.time()
        start_time = current_time
        
        # Try predictive pre-fetch
        if self.access_count % 5 == 0:  # Check every 5 accesses
            self._prefetch_predictions()
        
        if key in self.cache:
            data, history = self.cache[key]
            history.append(self.access_count)
            self.cache[key] = (data, history)
            self.access_timestamps[key] = current_time
            
            # Update time series
            self._update_time_series(key, current_time)
            
            # Update topological graph with time
            self._update_topological_graph_with_time(key, data, history, current_time)
            
            # Update persistence score
            self.persistence_scores[key] = self._compute_temporal_persistence(key)
            
            self.hits += 1
            
            # Track response time
            elapsed = time.time() - start_time
            self.access_times.append(elapsed)
            
            return data
        
        self.misses += 1
        
        # Track response time for miss
        elapsed = time.time() - start_time
        self.access_times.append(elapsed)
        
        return None
    
    def put(self, key: Any, data: Any, timestamp: float = None):
        """
        Insert an item into cache with time awareness.
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.access_count += 1
        
        # If already in cache, update
        if key in self.cache:
            _, history = self.cache[key]
            history.append(self.access_count)
            self.cache[key] = (data, history)
            self.access_timestamps[key] = timestamp
            
            # Update time series
            self._update_time_series(key, timestamp)
            
            # Update topological information with time
            self._update_topological_graph_with_time(key, data, history, timestamp)
            self.persistence_scores[key] = self._compute_temporal_persistence(key)
            return
        
        # If cache is full, evict with time awareness
        if len(self.cache) >= self.capacity:
            self._evict_with_time_awareness()
        
        # Add new item
        history = [self.access_count]
        self.cache[key] = (data, history)
        self.access_timestamps[key] = timestamp
        
        # Initialize time series
        self.time_series[key].append(timestamp)
        self.access_frequencies[key] = 1.0 / max(1, timestamp - self.last_access_time)
        
        # Initialize topological information with time
        sig = self._compute_temporal_signature(key, data, timestamp)
        self.topological_classes[sig].add(key)
        
        # Add to access graph
        self.access_graph.add_node(key)
        
        # Connect to topologically and temporally similar items
        for other_key in list(self.cache.keys())[:-1]:
            if other_key != key:
                other_data, _ = self.cache[other_key]
                other_sig = self._compute_temporal_signature(other_key, other_data, timestamp)
                distance = self._topological_distance(sig, other_sig)
                time_distance = abs(self._get_time_bucket_from_sig(sig) - 
                                   self._get_time_bucket_from_sig(other_sig))
                weighted_distance = distance * (1.0 + time_distance * 0.1)
                
                if weighted_distance < self.topological_radius:
                    self.access_graph.add_edge(key, other_key, 
                                              weight=self.topological_radius / (weighted_distance + 0.1))
        
        # Compute initial persistence score
        self.persistence_scores[key] = self._compute_temporal_persistence(key)
        self.last_access_time = timestamp
    
    def _is_topologically_essential(self, key: Any) -> bool:
        """
        Check if an item is topologically essential using Lewy-type theorem.
        Enhanced with temporal considerations.
        """
        if key not in self.cache:
            return False
        
        data, history = self.cache[key]
        current_time = time.time()
        
        # Check temporal significance
        access_times = list(self.time_series.get(key, []))
        recent_accesses = [t for t in access_times if current_time - t <= self.time_window]
        
        # Essential if accessed frequently and regularly
        if len(recent_accesses) > self.persistence_threshold * 2:
            if len(recent_accesses) >= 2:
                intervals = [recent_accesses[i+1] - recent_accesses[i] 
                            for i in range(len(recent_accesses)-1)]
                if intervals and len(set(intervals)) <= 2:
                    return True
        
        # Check topological significance
        sig = self._compute_temporal_signature(key, data, current_time)
        class_connections = set()
        
        for neighbor in self.access_graph.neighbors(key):
            if neighbor in self.cache:
                neigh_data, _ = self.cache[neighbor]
                neigh_sig = self._compute_temporal_signature(neighbor, neigh_data, current_time)
                if neigh_sig != sig:
                    class_connections.add(neigh_sig)
        
        # If it connects different classes, it's topologically essential
        if len(class_connections) >= 2:
            return True
        
        # Check transition importance
        if key in self.transition_graph:
            total_weight = sum(self.transition_graph[key][n]['weight'] 
                             for n in self.transition_graph[key])
            if total_weight > 5:  # High transition count
                return True
        
        return False
    
    def analyze_topology(self) -> Dict[str, Any]:
        """
        Analyze the topological structure of the cache.
        This is inspired by Milnor fibration analysis in the paper.
        """
        if len(self.cache) == 0:
            return {'classes': 0, 'connected_components': 0, 'average_degree': 0}
        
        # Find connected components (like Milnor fibers)
        components = list(nx.connected_components(self.access_graph))
        
        # Analyze each component
        component_analysis = []
        for comp in components:
            if len(comp) > 0:
                # Compute topological signature of component
                sigs = []
                for key in comp:
                    if key in self.cache:
                        data, _ = self.cache[key]
                        sigs.append(self._compute_topological_signature(key, data))
                
                # Unique signatures in component (like local branches)
                unique_sigs = set(sigs)
                
                component_analysis.append({
                    'size': len(comp),
                    'branches': len(unique_sigs),
                    'persistence_score': sum(self.persistence_scores.get(k, 0) for k in comp),
                })
        
        # Compute degree distribution
        degrees = [d for n, d in self.access_graph.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        
        return {
            'classes': len(self.topological_classes),
            'connected_components': len(components),
            'average_degree': avg_degree,
            'max_degree': max(degrees) if degrees else 0,
            'components': component_analysis,
        }
    
    def get_time_stats(self) -> Dict[str, Any]:
        """
        Get time-aware statistics.
        """
        current_time = time.time()
        
        # Calculate average response time
        avg_response = sum(self.access_times) / len(self.access_times) if self.access_times else 0
        
        # Calculate access rate
        if len(self.access_times) > 1:
            time_span = max(self.access_times) - min(self.access_times)
            access_rate = len(self.access_times) / max(time_span, 1)
        else:
            access_rate = 0
        
        # Count active items
        active_items = sum(1 for key in self.cache 
                          if current_time - max(self.time_series.get(key, [0])) <= self.time_window)
        
        return {
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            'cache_size': len(self.cache),
            'capacity': self.capacity,
            'time_evictions': self.time_evictions,
            'predictive_hits': self.predictive_hits,
            'avg_response_time': avg_response,
            'access_rate': access_rate,
            'active_items': active_items,
            'active_percentage': active_items / len(self.cache) if self.cache else 0,
            'oldest_item_age': current_time - min(min(self.time_series.get(k, [current_time])) 
                                                   for k in self.cache) if self.cache else 0,
        }
    
    def get_temporal_analysis(self) -> Dict[str, Any]:
        """
        Analyze temporal patterns in the cache.
        """
        current_time = time.time()
        
        # Analyze temporal branches
        branch_analysis = {}
        for time_bucket, branches in self.temporal_branches.items():
            branch_analysis[time_bucket] = {
                'branches': len(branches),
                'age': current_time - (time_bucket * self.time_window),
            }
        
        # Analyze transition graph
        if self.transition_graph:
            # Find cycles in temporal transitions (like Milnor fibers over time)
            try:
                cycles = list(nx.simple_cycles(self.transition_graph))
            except:
                cycles = []
            
            # Analyze strongly connected components
            scc = list(nx.strongly_connected_components(self.transition_graph))
            
            return {
                'temporal_branches': len(self.temporal_branches),
                'total_branches': sum(len(b) for b in self.temporal_branches.values()),
                'temporal_cycles': len(cycles),
                'strongly_connected_components': len(scc),
                'branch_analysis': branch_analysis,
                'transition_count': self.transition_graph.number_of_edges(),
                'average_edges_per_node': (self.transition_graph.number_of_edges() / 
                                          max(1, self.transition_graph.number_of_nodes())),
            }
        
        return {
            'temporal_branches': 0,
            'total_branches': 0,
            'temporal_cycles': 0,
            'strongly_connected_components': 0,
            'branch_analysis': {},
            'transition_count': 0,
            'average_edges_per_node': 0,
        }


class TimeAwareCacheManager:
    """
    Manager for time-aware topological caches with automatic tuning.
    """
    
    def __init__(self):
        self.caches = {}
        self.cache_stats = defaultdict(list)
        self.global_time = time.time()
    
    def create_cache(self, name: str, capacity: int, 
                     topological_radius: float = 0.5,
                     persistence_threshold: int = 3,
                     time_window: int = 3600) -> TimeAwareTopologicalCache:
        """Create a new time-aware topological cache."""
        cache = TimeAwareTopologicalCache(capacity, topological_radius, 
                                         persistence_threshold, time_window)
        self.caches[name] = cache
        return cache
    
    def get_cache(self, name: str) -> Optional[TimeAwareTopologicalCache]:
        """Get an existing cache."""
        return self.caches.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        stats = {}
        for name, cache in self.caches.items():
            stats[name] = cache.get_time_stats()
            stats[name]['temporal_analysis'] = cache.get_temporal_analysis()
            stats[name]['topology'] = cache.analyze_topology()
        return stats
    
    def optimize_time_parameters(self, name: str):
        """
        Optimize time-related parameters based on access patterns.
        """
        if name not in self.caches:
            return
        
        cache = self.caches[name]
        temporal = cache.get_temporal_analysis()
        
        # Adjust time window based on temporal patterns
        if temporal['temporal_branches'] > 10:
            # Too many branches, increase window to merge patterns
            cache.time_window = min(86400, cache.time_window * 1.2)  # Max 24 hours
        elif temporal['temporal_branches'] < 3 and cache.time_window > 300:
            # Too few branches, decrease window to split patterns
            cache.time_window = max(60, cache.time_window * 0.8)
        
        # Adjust aging factor based on temporal cycles
        if temporal['temporal_cycles'] > 2:
            # Many cycles, increase aging to favor recent patterns
            cache.aging_factor = min(0.99, cache.aging_factor * 1.01)
        else:
            cache.aging_factor = max(0.85, cache.aging_factor * 0.99)


# Example usage with time simulation
def test_time_aware_cache():
    """Test the time-aware topological cache with simulated time."""
    print("Testing Time-Aware Topological Cache...")
    
    # Create cache with time awareness
    cache = TimeAwareTopologicalCache(
        capacity=10, 
        topological_radius=0.5, 
        persistence_threshold=3,
        time_window=60,  # 1 minute window
        aging_factor=0.95
    )
    
    # Create a simple data source for prefetching
    class SimpleDataSource:
        def get(self, key):
            return f"prefetched_data_{key}"
    
    cache.set_data_source(SimpleDataSource())
    
    print("\nSimulating time-based access patterns...")
    
    # Test data
    test_data = {i: f"data_{i}" for i in range(1, 11)}
    
    # Simulate accesses over time
    patterns = [
        # Pattern 1: Regular access to key 1,2,3
        ([1, 2, 3], 0, 5),  # keys, start_time, duration
        # Pattern 2: Access to key 4,5,6
        ([4, 5, 6], 10, 4),
        # Pattern 3: Return to key 1,2,3
        ([1, 2, 3], 20, 3),
        # Pattern 4: Access to key 7,8,9,10
        ([7, 8, 9, 10], 30, 6),
        # Pattern 5: Mixed access
        ([1, 4, 7, 2, 5, 8], 40, 8),
    ]
    
    current_time = 0
    
    for keys, start_offset, duration in patterns:
        # Advance time
        current_time = start_offset
        
        # Access keys over the duration
        for step in range(duration * 2):  # 2 accesses per time unit
            key = keys[step % len(keys)]
            
            # Simulate real time
            time.sleep(0.01)  # Small delay for realistic timing
            
            # Access or put
            if current_time % 2 == 0:
                cache.put(key, test_data[key])
            else:
                cache.get(key)
            
            current_time += 0.5
    
    print(f"\nCache size: {len(cache.cache)}")
    print(f"Hits: {cache.hits}, Misses: {cache.misses}")
    print(f"Predictive hits: {cache.predictive_hits}")
    print(f"Time evictions: {cache.time_evictions}")
    
    stats = cache.get_time_stats()
    print(f"\nTime Statistics:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Active items: {stats['active_items']}/{stats['cache_size']}")
    print(f"  Active percentage: {stats['active_percentage']:.1%}")
    print(f"  Average response time: {stats['avg_response_time']:.4f}s")
    print(f"  Access rate: {stats['access_rate']:.2f} accesses/s")
    
    temporal = cache.get_temporal_analysis()
    print(f"\nTemporal Analysis:")
    print(f"  Temporal branches: {temporal['temporal_branches']}")
    print(f"  Total branches: {temporal['total_branches']}")
    print(f"  Temporal cycles: {temporal['temporal_cycles']}")
    print(f"  Transition count: {temporal['transition_count']}")
    
    print("\nCache contents with time information:")
    for key, (data, history) in sorted(cache.cache.items()):
        times = list(cache.time_series.get(key, []))
        last_access = max(times) if times else 0
        access_count = len(times)
        persistence = cache._compute_temporal_persistence(key)
        stability = cache._compute_topological_stability(key)
        print(f"  Key: {key}, Data: {data[:10]}..., "
              f"Accesses: {access_count}, "
              f"Last access: {last_access:.1f}, "
              f"Persistence: {persistence:.3f}, "
              f"Stability: {stability:.3f}")
    
    # Test cache manager
    print("\nTesting Time-Aware Cache Manager...")
    manager = TimeAwareCacheManager()
    cache2 = manager.create_cache("test2", capacity=5, time_window=30)
    
    # Simulate real-time pattern
    for i in range(20):
        key = i % 5 + 1
        if i % 3 == 0:
            cache2.put(key, f"data_{key}")
        else:
            cache2.get(key)
        time.sleep(0.01)
    
    print(f"Cache2 stats: {cache2.get_time_stats()}")
    
    manager.optimize_time_parameters("test2")
    print(f"Cache2 optimized stats: {cache2.get_time_stats()}")
    
    print("\nAll time-aware tests complete!")

if __name__ == "__main__":
    test_time_aware_cache()