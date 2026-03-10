import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from core.base_env import QuantumEnvironment

def get_exact_from_paulis(paulis, n_qubits):
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.array([[1, 0], [0, 1]])

    def tensor_op(ops_list):
        res = 1
        current_ops = {idx: op for op, idx in ops_list}
        for i in range(n_qubits):
            op_type = current_ops.get(i, 'i')
            if op_type == 'x': op = X
            elif op_type == 'y': op = Y
            elif op_type == 'z': op = Z
            else: op = I
            res = np.kron(res, op)
        return res

    H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for coeff, ops in paulis:
        if not ops:
            H += coeff * np.eye(2**n_qubits)
        else:
            H += coeff * tensor_op(ops)
            
    eigenvalues = np.linalg.eigvalsh(H)
    return np.min(eigenvalues)

class LiHEnvironment(QuantumEnvironment):
    def __init__(self):
        super().__init__("LiH_Full", 4, -7.8827)
        
        self.paulis = [
            (-2.4270, []), # Constant
            (0.132221, [('z', 0)]),
            (0.132221, [('z', 1)]),
            (-0.012341, [('z', 2)]),
            (-0.012341, [('z', 3)]),
            (0.170241, [('z', 0), ('z', 1)]),
            (0.121401, [('z', 0), ('z', 2)]),
            (0.168321, [('z', 0), ('z', 3)]),
            (0.168321, [('z', 1), ('z', 2)]),
            (0.121401, [('z', 1), ('z', 3)]),
            (0.174321, [('z', 2), ('z', 3)]),
            (0.045321, [('x', 0), ('x', 1), ('y', 2), ('y', 3)]),
            (0.045321, [('y', 0), ('y', 1), ('x', 2), ('x', 3)]),
            (0.045321, [('x', 0), ('y', 1), ('y', 2), ('x', 3)]),
            (0.045321, [('y', 0), ('x', 1), ('x', 2), ('y', 3)]),
            (0.018121, [('z', 0), ('z', 1), ('z', 2)]),
            (0.018121, [('z', 0), ('z', 1), ('z', 3)]),
        ]
        self.exact_energy = get_exact_from_paulis(self.paulis, self.n_qubits).real

    def compute_energy(self, c):
        import tensorcircuit as tc
        energy = 0.0
        for coeff, ops in self.paulis:
            if not ops:
                energy += coeff
            else:
                energy += coeff * c.expectation(*[[getattr(tc.gates, op_type)(), [idx]] for op_type, idx in ops])
        return tc.backend.real(energy)

ENV = LiHEnvironment()
