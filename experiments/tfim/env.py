import os
import sys
from functools import lru_cache

import numpy as np

# 将项目根目录添加到路径中以防万一
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from core.foundation.base_env import QuantumEnvironment


_I2 = np.eye(2, dtype=np.float64)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)


def _kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


@lru_cache(maxsize=16)
def _dense_open_tfim_ground_energy(n_qubits: int) -> float:
    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")

    dim = 2 ** n_qubits
    hamiltonian = np.zeros((dim, dim), dtype=np.float64)

    for i in range(n_qubits - 1):
        ops = [_I2] * n_qubits
        ops[i] = _Z
        ops[i + 1] = _Z
        hamiltonian -= _kron_all(ops)

    for i in range(n_qubits):
        ops = [_I2] * n_qubits
        ops[i] = _X
        hamiltonian -= _kron_all(ops)

    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    return float(eigenvalues[0])


def get_tfim_reference_energy(n_qubits: int, max_exact_qubits: int = 12) -> float:
    """
    Return the exact open-chain TFIM ground-state energy when exact
    diagonalization is still cheap. Larger systems are treated as scaling
    experiments without an exact reference.
    """
    if n_qubits <= max_exact_qubits:
        return _dense_open_tfim_ground_energy(n_qubits)
    return float("nan")


class TFIMEnvironment(QuantumEnvironment):
    def __init__(self, n_qubits=4, use_mps=False):
        exact_energy = get_tfim_reference_energy(n_qubits)
        super().__init__("TFIM", n_qubits, exact_energy, use_mps=use_mps)
        self.boundary = "open"
        self.reference_energy_kind = "exact_diagonalization" if np.isfinite(exact_energy) else "unavailable"

    def compute_energy(self, c):
        import tensorcircuit as tc
        def get_gate(name):
            obj = getattr(tc.gates, name, None)
            if obj is None: obj = getattr(tc.gates, name.upper(), None)
            if obj is None: obj = getattr(tc.gates, f"{name}_gate", None)
            return obj() if callable(obj) else obj

        energy = 0.0
        # - Z_i Z_{i+1}
        for i in range(self.n_qubits - 1):
            energy += -c.expectation([get_gate("z"), [i]], [get_gate("z"), [i+1]])
        # - X_i
        for i in range(self.n_qubits):
            energy += -c.expectation([get_gate("x"), [i]])
        return tc.backend.real(energy)

ENV = TFIMEnvironment()
