import pytest
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.circuit_factory import count_params, build_ansatz

def test_translational_param_count():
    """测试 translational 降维策略的参数计算与线路构建逻辑"""
    
    config = {
        "layers": 3,
        "single_qubit_gates": ["rx", "ry"],
        "two_qubit_gate": "rzz",
        "entanglement": "linear",
        "param_strategy": "translational"
    }
    
    # 期望参数量 = 3层 * (单比特2种 + 双比特1种) = 9
    expected_params = 3 * (2 + 1)
    
    # 分别测试 10, 50, 100 量子比特，确保参数量都是 O(1) 的固定 9 个
    for n_qubits in [10, 50, 100]:
        # 1. 测试 count_params 逻辑
        num_params = count_params(config, n_qubits)
        assert num_params == expected_params, f"Count failed for N={n_qubits}, expected {expected_params}, got {num_params}"
        
        # 2. 测试 build_ansatz 逻辑
        create_circ, num_params_builder = build_ansatz(config, n_qubits)
        assert num_params_builder == expected_params, f"Builder count failed for N={n_qubits}, expected {expected_params}, got {num_params_builder}"
        
        # 3. 实际构建一次电路测试
        import torch
        test_params = torch.randn(expected_params, dtype=torch.float32)
        c, actual_idx = create_circ(test_params)
        
        # 确保消耗了预期的所有参数
        assert actual_idx == expected_params, f"Circuit built with actual idx {actual_idx}, expected {expected_params}"

        # 检查线路的确被构建
        assert c._nqubits == n_qubits
