import random
import numpy as np

# ============================================
# 1. A LIVING NEURON (no algebra, just thresholds)
# ============================================
class Neuron:
    def __init__(self, id):
        self.id = id
        self.threshold = random.uniform(0.3, 0.7)  # fires if input >= this
        self.connections = []  # (target_neuron, weight)
    
    def fire(self, input_signal):
        # No multiplication! Just sum and compare
        total = sum(input_signal)  # raw sum, no weights in forward pass
        return 1.0 if total >= self.threshold else 0.0

# ============================================
# 2. A GROWING NETWORK (mutates itself)
# ============================================
class LivingNetwork:
    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(i) for i in range(input_size + output_size + 5)]
        self.input_ids = list(range(input_size))
        self.output_ids = list(range(input_size, input_size + output_size))
        self.fitness = 0
        self.age = 0
        
        # Random initial connections (sparse)
        for _ in range(10):
            self._mutate_connection()
    
    def _mutate_connection(self):
        # Adds a random connection between any two neurons
        src = random.choice(self.neurons)
        tgt = random.choice(self.neurons)
        if src != tgt:
            weight = random.uniform(-1.0, 1.0)
            src.connections.append((tgt, weight))
    
    def _mutate_neuron(self):
        # Adds a new neuron by splitting an existing connection
        if len(self.neurons) < 50:  # prevent explosion
            new_id = len(self.neurons)
            new_n = Neuron(new_id)
            self.neurons.append(new_n)
            # Copy a random connection's target
            src = random.choice(self.neurons)
            if src.connections:
                tgt = random.choice(src.connections)[0]
                src.connections.append((new_n, random.uniform(-1,1)))
                new_n.connections.append((tgt, random.uniform(-1,1)))
    
    def mutate(self):
        # Evolution happens here — no gradients!
        self.age += 1
        if random.random() < 0.3:
            self._mutate_connection()
        if random.random() < 0.1:
            self._mutate_neuron()
        # Randomly tweak thresholds (living adaptation)
        for n in random.sample(self.neurons, min(3, len(self.neurons))):
            n.threshold += random.uniform(-0.1, 0.1)
            n.threshold = max(0.1, min(0.9, n.threshold))
    
    def think(self, inputs):
        # Forward pass — NO multiplication, just sums and thresholds
        values = {n.id: 0.0 for n in self.neurons}
        for i, val in zip(self.input_ids, inputs):
            values[i] = val
        
        # Simple spreading activation (3 passes)
        for _ in range(3):
            for n in self.neurons:
                if n.id in self.input_ids:
                    continue
                # Sum inputs from connected neurons
                incoming = [values[src.id] * w for src, w in n.connections]
                if incoming:
                    values[n.id] = n.fire(incoming)
        
        # Read outputs
        return [values[oid] for oid in self.output_ids]

# ============================================
# 3. EVOLUTION ENGINE (survival of the fittest)
# ============================================
class Evolution:
    def __init__(self, population_size=50, input_size=4, output_size=2):
        self.population = [LivingNetwork(input_size, output_size) for _ in range(population_size)]
        self.generation = 0
    
    def evaluate(self, network):
        # Fitness: how well it solves a simple XOR-like pattern
        # (completely non-algebraic: just reward correct guesses)
        score = 0
        tests = [
            ([0,0,0,0], [0,0]),
            ([1,1,0,0], [1,0]),
            ([0,0,1,1], [0,1]),
            ([1,0,1,0], [1,1]),
        ]
        for inp, expected in tests:
            out = network.think(inp)
            for e, o in zip(expected, out):
                score += 1.0 if abs(e - o) < 0.5 else 0.0
        return score
    
    def next_generation(self):
        self.generation += 1
        # Score everyone
        scored = [(self.evaluate(n), n) for n in self.population]
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Keep top 20% as parents
        keep = int(len(self.population) * 0.2)
        parents = [n for _, n in scored[:keep]]
        
        # Fill rest with mutated copies
        new_pop = parents.copy()
        while len(new_pop) < len(self.population):
            parent = random.choice(parents)
            child = LivingNetwork(4, 2)  # fresh start
            # Copy parent's structure (simplified crossover)
            child.neurons = [Neuron(n.id) for n in parent.neurons]
            for src, tgt, w in [(s, t, w) for s in parent.neurons for t, w in s.connections]:
                # Rebuild connections
                src_child = next(n for n in child.neurons if n.id == src.id)
                tgt_child = next(n for n in child.neurons if n.id == tgt.id)
                src_child.connections.append((tgt_child, w))
            child.mutate()  # mutate the copy
            new_pop.append(child)
        
        self.population = new_pop
        return scored[0][1]  # return best

# ============================================
# 4. RUN IT — No Algebra, Just Living Code
# ============================================
if __name__ == "__main__":
    evo = Evolution(population_size=30)
    
    print("🧬 Evolving living networks without math...\n")
    for gen in range(20):
        best = evo.next_generation()
        best_fitness = evo.evaluate(best)
        print(f"Gen {gen+1}: Best fitness = {best_fitness}/8 | Neurons = {len(best.neurons)}")
        
        if best_fitness >= 8:
            print("\n✅ Perfect! The network learned without a single equation.")
            break
    
    # Test the final brain
    print("\n🔮 Final predictions:")
    for inp in [[0,0,0,0], [1,1,0,0], [0,0,1,1], [1,0,1,0]]:
        out = best.think(inp)
        print(f"  {inp} → {[round(o) for o in out]}")