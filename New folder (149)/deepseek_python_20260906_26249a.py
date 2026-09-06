import time
import math
import sys

class ZenoEngine:
    """
    Conceptual simulator for the Transfinite Temporal Compression (TTC) Algorithm.
    Step k executes at theoretical time t = T * (1 - 2^(-k)).
    The infinite series converges to T, yielding infinite asymptotic speedup.
    """

    def __init__(self, total_time: float = 1.0):
        self.T = total_time  # Total convergence time (e.g., 1 second)

    def get_theoretical_time(self, step: int) -> float:
        """Returns the absolute theoretical timestamp of the k-th step."""
        if step < 0:
            return 0.0
        # As step -> infinity, 2^-step -> 0, so time -> self.T
        return self.T * (1.0 - 2.0 ** (-step))

    def simulate_sat_bruteforce(self, num_vars: int, target_assignment: int, max_demo_steps: int = 40):
        """
        Simulates solving a SAT problem via brute-force assignment checking.
        In classical computing, checking 2^num_vars assignments takes O(2^n).
        In this Zeno model, ALL assignments are checked by t = self.T.
        """
        search_space = 2 ** num_vars
        print(f"\n=== SAT Brute-Force Simulation ===")
        print(f"Variables: {num_vars} | Total assignments: {search_space}")
        print(f"Theoretical Zeno convergence: {self.T} second(s)")
        print("-" * 60)

        start_wall = time.perf_counter()
        step = 1

        while step <= search_space:
            # 1. Calculate theoretical execution time
            t_step = self.get_theoretical_time(step)

            # 2. Generate the k-th binary assignment (simulating CPU work)
            #    We convert step to binary bits representing True/False assignments.
            assignment_bits = [(step >> i) & 1 for i in range(num_vars)]
            current_value = step  # For simplicity, we treat the integer as the candidate

            # 3. Print progress (only for first few steps to show the pattern)
            if step <= 15 or step > search_space - 5 or t_step > 0.999:
                print(f"Step {step:4d} | Time: {t_step:.12f}s | Assign: {assignment_bits}")

            # 4. Check if we found the target
            if current_value == target_assignment:
                print(f"\n>>> SATISFIABLE! Found target {target_assignment} at step {step}.")
                print(f">>> Theoretical completion time: {t_step:.12f} seconds.")
                break

            step += 1

            # 5. Safety break for demonstration.
            #    In a real Zeno machine, this loop would run until step=search_space.
            #    We truncate to prevent Python from running for years.
            if step > max_demo_steps:
                remaining_steps = search_space - step
                t_at_max = self.get_theoretical_time(max_demo_steps)
                print(f"\n... (Simulation truncated at step {max_demo_steps} for demo)")
                print(f"Remaining {remaining_steps} assignments would execute between")
                print(f"t = {t_at_max:.12f}s and t = {self.T:.12f}s (the convergence limit).")
                print(f"Mathematically, the algorithm checks ALL {search_space} assignments")
                print(f"within the finite window of {self.T} second(s).")
                break
        else:
            print("Target not found in search space.")

        elapsed = time.perf_counter() - start_wall
        print(f"\nPhysical wall-clock time for this simulation: {elapsed:.4f} seconds.")
        print(f"(This is linear time because the CPU cannot physically accelerate.)")

    def brute_force_attack(self, password_length: int, target: str):
        """
        Simulates a password cracker using the Zeno schedule.
        Tries all combinations (A-Z) but demonstrates the infinite speedup.
        """
        import itertools
        import string

        chars = string.ascii_uppercase
        total_combos = len(chars) ** password_length
        print(f"\n=== Password Cracking (Zeno Mode) ===")
        print(f"Searching for '{target}' among {total_combos} combinations.")
        print(f"Converges in exactly {self.T} second(s) theoretically.")
        print("-" * 60)

        start_wall = time.perf_counter()
        step = 1

        # Generate all combinations lazily
        for combo in itertools.product(chars, repeat=password_length):
            candidate = ''.join(combo)
            t_step = self.get_theoretical_time(step)

            # Print first few attempts
            if step <= 10:
                print(f"Step {step:4d} | t={t_step:.12f}s | Trying: {candidate}")

            if candidate == target:
                print(f"\n>>> CRACKED! Password '{target}' found at step {step}.")
                print(f">>> Theoretical time: {t_step:.12f} seconds.")
                break

            step += 1

            # Artificial truncation for demo
            if step > 30:
                print(f"\n... (Simulation truncated at step 30)")
                print(f"Remaining {total_combos - step} combos checked between")
                print(f"t={self.get_theoretical_time(30):.12f}s and t={self.T}s.")
                break

        elapsed = time.perf_counter() - start_wall
        print(f"Simulation wall-clock time: {elapsed:.4f}s")

    def physical_attempt(self, steps: int = 30):
        """
        ATTEMPT to physically accelerate the CPU by sleeping decreasing amounts.
        This is purely educational: it will fail miserably once sleeps hit 0.0
        or the OS scheduler limit, proving why this is strictly a theoretical model.
        """
        print(f"\n=== Attempting Physical Acceleration (Sleep Test) ===")
        print("(This will break once sleep duration approaches the nanosecond limit)")
        start = time.perf_counter()
        for k in range(1, steps + 1):
            sleep_dur = 2.0 ** (-k)  # Shrinks exponentially
            try:
                # On most OS, sleep(1e-9) works, but sleep(1e-18) rounds to 0.
                if sleep_dur < 1e-9:
                    print(f"Step {k}: Sleep {sleep_dur:.2e}s -> Rounds to 0. Physics breaks.")
                    break
                time.sleep(sleep_dur)
                t_elapsed = time.perf_counter() - start
                print(f"Step {k:2d} | Slept: {sleep_dur:.10f}s | Total elapsed: {t_elapsed:.6f}s")
            except ValueError as e:
                print(f"Step {k}: Overflow/Error - {e}")
                break


if __name__ == "__main__":
    # Initialize the Zeno Engine (converges in exactly 1 second)
    engine = ZenoEngine(total_time=1.0)

    # --- DEMO 1: SAT Brute-Force ---
    # We have 6 variables (64 assignments). Target assignment is decimal 42 (binary 101010).
    engine.simulate_sat_bruteforce(num_vars=6, target_assignment=42, max_demo_steps=25)

    # --- DEMO 2: Password Cracking ---
    # Searching 4-letter password (26^4 = 456,976 combos) in < 1 second theoretically.
    engine.brute_force_attack(password_length=4, target="ZENO")

    # --- DEMO 3: Why it fails physically ---
    engine.physical_attempt(steps=40)