import numpy as np
import sys
import os

# 将项目根目录添加到路径中以防万一
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from core.foundation.base_env import QuantumEnvironment

class TFIMEnvironment(QuantumEnvironment):
    def __init__(self, n_qubits=4, use_mps=False):
        # 4-qubit TFIM exact energy
        super().__init__("TFIM", n_qubits, -4.758770, use_mps=use_mps)

    def compute_energy(self, c):
        import tensorcircuit as tc
        energy = 0.0
        # - Z_i Z_{i+1}
        for i in range(self.n_qubits - 1):
            energy += -c.expectation([tc.gates.z(), [i]], [tc.gates.z(), [i+1]])
        # - X_i
        for i in range(self.n_qubits):
            energy += -c.expectation([tc.gates.x(), [i]])
        return tc.backend.real(energy)

ENV = TFIMEnvironment()
