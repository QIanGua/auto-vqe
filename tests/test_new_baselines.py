import torch
import tensorcircuit as tc
import pytest
from baselines import givens, kupccgsd, qucc

class MockEnv:
    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
        self.name = "LiH_Mock"

@pytest.mark.parametrize("baseline_module, name", [
    (givens, "givens"),
    (kupccgsd, "kupccgsd"),
    (qucc, "qucc")
])
def test_baseline_build(baseline_module, name):
    tc.set_backend("pytorch")
    env = MockEnv(n_qubits=4)
    
    spec = baseline_module.build_ansatz(env)
    assert spec.name == name
    assert spec.num_params > 0
    
    params = torch.zeros(spec.num_params)
    circuit, param_count = spec.create_circuit(params)
    
    assert param_count == spec.num_params
    # Check circuit nqubits
    n_q = getattr(circuit, "_nqubits", None) or getattr(circuit, "nqubits", 0)
    assert n_q == 4
