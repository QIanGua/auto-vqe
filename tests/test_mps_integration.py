import pytest
import os
import sys
import numpy as np
import tensorcircuit as tc

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from experiments.tfim.env import TFIMEnvironment
from core.representation.compiler import build_ansatz

@pytest.mark.slow
def test_mps_statevector_consistency():
    """测试在相同结构和小规模下, MPS 模拟器和 Statevector 模拟器的能量结果是否一致"""
    n_qubits = 8
    
    # 相同的网络配置
    config = {
        "layers": 2,
        "single_qubit_gates": ["ry", "rz"],
        "two_qubit_gate": "cnot",
        "entanglement": "linear",
        "param_strategy": "independent"
    }
    
    # 1. SV Backend
    env_sv = TFIMEnvironment(n_qubits=n_qubits, use_mps=False)
    config_sv = dict(config)
    config_sv["use_mps"] = False
    
    create_circ_sv, num_params = build_ansatz(config_sv, n_qubits)
    
    # 2. MPS Backend
    env_mps = TFIMEnvironment(n_qubits=n_qubits, use_mps=True)
    config_mps = dict(config)
    config_mps["use_mps"] = True
    
    create_circ_mps, num_params_mps = build_ansatz(config_mps, n_qubits)
    
    assert num_params == num_params_mps
    
    # 生成随机参数, 对于两边保持一致
    np.random.seed(42)
    test_params = np.random.randn(num_params).astype(np.float32)
    
    import torch
    params_sv = torch.tensor(test_params, dtype=torch.float32)
    params_mps = torch.tensor(test_params, dtype=torch.float32)
    
    # Evaluate SV
    with torch.no_grad():
        c_sv, _ = create_circ_sv(params_sv)
        energy_sv = env_sv.compute_energy(c_sv)
    
        # Evaluate MPS
        c_mps, _ = create_circ_mps(params_mps)
        energy_mps = env_mps.compute_energy(c_mps)
    
    # Compare
    assert isinstance(c_sv, tc.Circuit)
    assert isinstance(c_mps, tc.MPSCircuit)
    
    # The difference should be extremely small
    diff = abs(energy_sv.item() - energy_mps.item())
    assert diff < 1e-4, f"Energy difference too large: sv={energy_sv.item()}, mps={energy_mps.item()}, diff={diff}"
