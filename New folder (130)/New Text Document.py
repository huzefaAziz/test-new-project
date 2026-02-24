import numpy as np

class HilbertTuringMachine:
    def __init__(self, alphabet, blank='_', max_tape=100):
        self.alphabet = alphabet
        self.blank = blank
        self.tape = [blank] * max_tape  # simulate Hilbert space with finite array
        self.head = max_tape // 2       # start head in middle
        self.state = 'q0'
        self.states = {'q0', 'q1', 'q_accept', 'q_reject'}

        # Hilbert-space representation of tape (one-hot encoding for each cell)
        self.tape_vectors = np.zeros((max_tape, len(alphabet)))
        self.update_tape_vectors()
        
    def update_tape_vectors(self):
        for i, symbol in enumerate(self.tape):
            idx = self.alphabet.index(symbol)
            self.tape_vectors[i] = np.zeros(len(self.alphabet))
            self.tape_vectors[i][idx] = 1.0

    def step(self):
        current_symbol = self.tape[self.head]

        # Define a simple transition function
        if self.state == 'q0':
            if current_symbol == '1':
                self.tape[self.head] = '0'
                self.state = 'q1'
                self.head += 1
            else:
                self.state = 'q_accept'
        elif self.state == 'q1':
            if current_symbol == '0':
                self.tape[self.head] = '1'
                self.state = 'q0'
                self.head -= 1
            else:
                self.state = 'q_reject'

        # Update Hilbert-space vectors
        self.update_tape_vectors()

    def run(self, max_steps=50):
        steps = 0
        while self.state not in ['q_accept', 'q_reject'] and steps < max_steps:
            self.step()
            steps += 1
        return self.state, ''.join(self.tape)

    def visualize_tape_vectors(self):
        print(np.round(self.tape_vectors, 2))

# Example usage
htm = HilbertTuringMachine(alphabet=['0', '1', '_'])
htm.tape[htm.head:htm.head+3] = ['1', '0', '1']  # initial input
final_state, final_tape = htm.run()
print("Final state:", final_state)
print("Final tape:", final_tape)
htm.visualize_tape_vectors()