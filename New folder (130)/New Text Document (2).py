# Base types for states
class State: pass
class Start(State): pass
class Halt(State): pass

# Transition type
class Transition:
    def __init__(self, from_state: type[State], input_symbol, to_state: type[State], write_symbol, move):
        self.from_state = from_state
        self.input_symbol = input_symbol
        self.to_state = to_state
        self.write_symbol = write_symbol
        self.move = move  # 'L' or 'R'

# Conceptually infinite tape
class Tape(list):
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            return '_'
        return super().__getitem__(idx)

    def __setitem__(self, idx, value):
        if idx < 0:
            raise IndexError("Negative index not supported")
        while idx >= len(self):
            self.append('_')
        super().__setitem__(idx, value)

# Iterative infinite machine runner
class InfiniteMachine:
    transitions: list[Transition] = []

    @classmethod
    def run(cls, tape: Tape, start_state: type[State] = Start, max_steps: int = 1000):
        state = start_state
        head = 0
        for _ in range(max_steps):
            if state is Halt:
                break
            for t in cls.transitions:
                if t.from_state is state and t.input_symbol == tape[head]:
                    tape[head] = t.write_symbol
                    head += 1 if t.move == 'R' else -1
                    state = t.to_state
                    break
            else:
                # No matching transition → halt
                break
        return tape

# Define machine with type-level transitions
class MyMachine(InfiniteMachine):
    transitions = [
        Transition(Start, '_', Start, '1', 'R'),  # Keep writing 1 infinitely
        Transition(Start, '1', Start, '1', 'R'),
    ]

# Run machine safely
tape = Tape()
result = MyMachine.run(tape, max_steps=50)
print(result)  # Prints first 50 steps