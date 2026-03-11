class QuantumEnvironment:
    def __init__(self, name, n_qubits, exact_energy, use_mps: bool = False):
        self.name = name
        self.n_qubits = n_qubits
        self.exact_energy = exact_energy
        self.use_mps = use_mps

    def compute_energy(self, c):
        raise NotImplementedError
