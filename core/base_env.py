class QuantumEnvironment:
    def __init__(self, name, n_qubits, exact_energy):
        self.name = name
        self.n_qubits = n_qubits
        self.exact_energy = exact_energy

    def compute_energy(self, c):
        raise NotImplementedError
