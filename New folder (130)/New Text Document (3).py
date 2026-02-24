from collections import defaultdict

class InfiniteDimensionalTuringMachine:
    def __init__(self, dimensions=2, blank=0):
        self.dimensions = dimensions
        self.blank = blank
        
        # Sparse infinite tape: coordinate tuple -> value
        self.tape = defaultdict(lambda: self.blank)
        
        # Head position in ℤ^n
        self.head = tuple([0] * self.dimensions)
        
        # State
        self.state = "q0"
        
        # Transition function
        # (state, symbol) -> (new_symbol, move_vector, new_state)
        self.transitions = {}
        
    def add_transition(self, state, symbol, new_symbol, move_vector, new_state):
        self.transitions[(state, symbol)] = (new_symbol, move_vector, new_state)
    
    def step(self):
        current_symbol = self.tape[self.head]
        
        key = (self.state, current_symbol)
        if key not in self.transitions:
            return False  # halt
        
        new_symbol, move_vector, new_state = self.transitions[key]
        
        # Write
        self.tape[self.head] = new_symbol
        
        # Move in n-dimensional space
        self.head = tuple(
            self.head[i] + move_vector[i]
            for i in range(self.dimensions)
        )
        
        # Update state
        self.state = new_state
        
        return True
    
    def run(self, max_steps=1000):
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return steps

    def get_non_blank_cells(self):
        return {k: v for k, v in self.tape.items() if v != self.blank}
    
    
tm = InfiniteDimensionalTuringMachine(dimensions=2, blank=0)

# Write 1 and move diagonally
tm.add_transition("q0", 0, 1, (1, 1), "q0")

tm.run(20)

print("Non-blank cells:")
print(tm.get_non_blank_cells())