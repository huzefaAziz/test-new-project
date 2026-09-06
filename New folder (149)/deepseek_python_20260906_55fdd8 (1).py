#!/usr/bin/env python3
"""
Comparison: Brute-force (simulating infinite computation) vs. Genetic Algorithm
Problem: maximise f(x, y) = -(x^2 + y^2) over the domain [-10, 10] x [-10, 10].
Global optimum: (0, 0) with value 0.0.
"""

import itertools
import random
import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------------------------------------------------
# Objective function
# ----------------------------------------------------------------------
def fitness(x, y):
    """Return the objective value to maximise."""
    return -(x * x + y * y)

# ----------------------------------------------------------------------
# 1. Brute-force exhaustive search - the "infinite-fast" strategy
# ----------------------------------------------------------------------
def brute_force_infinite(step=0.1):
    """
    Enumerate every discrete combination in the search space.
    With infinite speed, we could take step->0, but we keep it finite
    for illustrative purposes.
    """
    # Build the list of x and y values from -10.0 to 10.0 with given step
    values = [i * step for i in range(int(-10 / step), int(10 / step) + 1)]

    best_x = best_y = None
    best_value = float('-inf')
    total_evaluations = 0

    start = time.time()

    for x, y in itertools.product(values, repeat=2):
        total_evaluations += 1
        val = fitness(x, y)
        if val > best_value:
            best_value = val
            best_x, best_y = x, y

    elapsed = time.time() - start

    return {
        'best_x': best_x,
        'best_y': best_y,
        'best_value': best_value,
        'evaluations': total_evaluations,
        'time': elapsed
    }

# ----------------------------------------------------------------------
# 2. Genetic Algorithm (comparison)
# ----------------------------------------------------------------------
def genetic_algorithm(pop_size=50, generations=100, mutation_rate=0.1, mutation_std=0.5):
    """
    A simple GA with:
      - real-valued encoding
      - roulette-wheel selection
      - single-point crossover (swap y coordinates)
      - Gaussian mutation (clipped to bounds)
    """
    # Initialise random population
    population = [(random.uniform(-10, 10), random.uniform(-10, 10))
                  for _ in range(pop_size)]

    best_ever_x = best_ever_y = None
    best_ever_value = float('-inf')
    total_evaluations = 0

    start = time.time()

    for _ in range(generations):
        # Evaluate all individuals
        fitness_scores = []
        for x, y in population:
            val = fitness(x, y)
            fitness_scores.append(val)
            total_evaluations += 1
            if val > best_ever_value:
                best_ever_value = val
                best_ever_x, best_ever_y = x, y

        # Selection (roulette wheel)
        total_fit = sum(fitness_scores)
        if total_fit == 0:
            # Avoid division by zero - fallback to uniform random choice
            selected = random.choices(population, k=pop_size)
        else:
            probs = [f / total_fit for f in fitness_scores]
            selected = random.choices(population, weights=probs, k=pop_size)

        # Crossover and mutation to form next generation
        new_population = []
        for i in range(0, pop_size, 2):
            p1 = selected[i]
            p2 = selected[i + 1] if i + 1 < pop_size else selected[0]

            # Single-point crossover: swap y coordinates
            c1 = (p1[0], p2[1])
            c2 = (p2[0], p1[1])

            # Mutation (Gaussian noise)
            if random.random() < mutation_rate:
                c1 = (c1[0] + random.gauss(0, mutation_std),
                      c1[1] + random.gauss(0, mutation_std))
            if random.random() < mutation_rate:
                c2 = (c2[0] + random.gauss(0, mutation_std),
                      c2[1] + random.gauss(0, mutation_std))

            # Clip to the allowed domain
            c1 = (max(-10, min(10, c1[0])), max(-10, min(10, c1[1])))
            c2 = (max(-10, min(10, c2[0])), max(-10, min(10, c2[1])))

            new_population.extend([c1, c2])

        # Keep population size constant
        population = new_population[:pop_size]

    elapsed = time.time() - start

    return {
        'best_x': best_ever_x,
        'best_y': best_ever_y,
        'best_value': best_ever_value,
        'evaluations': total_evaluations,
        'time': elapsed
    }

# ----------------------------------------------------------------------
# Main: run both and compare
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("COMPARISON: INFINITE-FAST BRUTE FORCE vs. GENETIC ALGORITHM")
    print("=" * 60)
    print("Problem: Maximise f(x,y) = -(x^2 + y^2) on [-10,10]^2")
    print("Global optimum: (0, 0) with value 0.0")
    print("-" * 60)

    # Run brute force (simulated infinite computation)
    bf_result = brute_force_infinite(step=0.1)
    print("\n[1] BRUTE FORCE (exhaustive search, step = 0.1)")
    print(f"    Best solution: x = {bf_result['best_x']:.4f}, y = {bf_result['best_y']:.4f}")
    print(f"    Best value:    {bf_result['best_value']:.8f}")
    print(f"    Evaluations:   {bf_result['evaluations']}")
    print(f"    Time:          {bf_result['time']:.6f} seconds")
    print("    >> Guaranteed global optimum within the discretisation.")

    # Run genetic algorithm
    ga_result = genetic_algorithm(pop_size=50, generations=100,
                                  mutation_rate=0.1, mutation_std=0.5)
    print("\n[2] GENETIC ALGORITHM")
    print(f"    Best solution: x = {ga_result['best_x']:.4f}, y = {ga_result['best_y']:.4f}")
    print(f"    Best value:    {ga_result['best_value']:.8f}")
    print(f"    Evaluations:   {ga_result['evaluations']}")
    print(f"    Time:          {ga_result['time']:.6f} seconds")
    print("    >> May get close but can get stuck in local optima, and needs tuning.")

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("If computation is infinitely fast and memory unlimited, brute-force")
    print("exhaustive search is always superior because it guarantees the global")
    print("optimum without any parameter tuning. Genetic algorithms are only")
    print("useful when you cannot afford to evaluate all possibilities.")
    print("=" * 60)