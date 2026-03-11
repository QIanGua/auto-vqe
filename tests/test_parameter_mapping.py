import torch
import pytest
from core.parameter_mapping import IdentityMapper

def test_identity_mapper_layer_addition():
    mapper = IdentityMapper()
    n_qubits = 4
    
    old_config = {
        "layers": 1,
        "single_qubit_gates": ["ry"],
        "two_qubit_gate": "cnot",
        "entanglement": "linear"
    }
    # 4 sq gates * 1 layer = 4 parameters
    old_params = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    new_config = old_config.copy()
    new_config["layers"] = 2
    # 4 sq gates * 2 layers = 8 parameters
    
    mapped_params = mapper.map(old_config, old_params, new_config, n_qubits)
    
    assert mapped_params.shape == (8,)
    assert torch.allclose(mapped_params[:4], old_params)
    assert torch.all(mapped_params[4:] == 0)

def test_identity_mapper_different_gates():
    mapper = IdentityMapper()
    n_qubits = 4
    
    old_config = {"layers": 1, "single_qubit_gates": ["ry"]}
    old_params = torch.tensor([1.0, 1.0, 1.0, 1.0])
    
    new_config = {"layers": 1, "single_qubit_gates": ["rx", "ry"]}
    # new sq_gates = 2 * 4 = 8 parameters
    
    mapped_params = mapper.map(old_config, old_params, new_config, n_qubits)
    
    assert mapped_params.shape == (8,)
    # At least the first few are preserved in our current simple implementation
    assert torch.allclose(mapped_params[:4], old_params)

def test_identity_mapper_identity_preservation():
    mapper = IdentityMapper()
    n_qubits = 4
    config = {"layers": 1, "single_qubit_gates": ["ry"]}
    params = torch.tensor([0.1, 0.2, 0.3, 0.4])
    
    mapped = mapper.map(config, params, config, n_qubits)
    assert torch.allclose(mapped, params)
