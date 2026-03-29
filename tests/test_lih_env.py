import pytest
import numpy as np
import torch
import os
import sys
import json

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.lih.env import LiHEnvironment, get_exact_from_paulis

def test_lih_env_initialization():
    """验证 LiH 环境的初始化，包括默认点和指定 R 值。"""
    # 默认初始化
    env_default = LiHEnvironment()
    assert env_default.n_qubits == 4
    assert hasattr(env_default, "paulis")
    assert len(env_default.paulis) > 0
    
    # 指定 R 值 (1.6 A 是常见的平衡位置)
    env_r16 = LiHEnvironment(R=1.6)
    assert "1.6" in env_r16.name
    assert env_r16.n_qubits == 4

def test_lih_env_exact_energy():
    """验证 LiH 环境的精确能量计算。"""
    env = LiHEnvironment(R=1.6)
    # 确保 exact_energy_active 已经计算
    assert hasattr(env, "exact_energy_active")
    # 验证 get_exact_from_paulis 结果一致性
    val = get_exact_from_paulis(env.paulis, env.n_qubits)
    assert np.isclose(val.real, env.exact_energy_active)

def test_lih_env_explicit_energy_fields():
    """新 schema 下应显式区分 full-space FCI 与 active-space exact。"""
    env = LiHEnvironment(R=1.6)
    assert hasattr(env, "full_fci_energy")
    assert hasattr(env, "nuclear_repulsion")
    assert env.full_fci_energy is not None
    assert env.exact_energy_active is not None
    assert env.full_fci_energy < env.exact_energy_active

def test_lih_env_legacy_json_fallback():
    """旧 schema 只有 exact_energy 时，仍应可回退加载。"""
    import experiments.lih.env as lih_env

    legacy_data = {
        "points": [
            {
                "R": 1.6,
                "exact_energy": -6.890117108408502,
                "n_qubits": 1,
                "paulis": [
                    {"coeff": -0.5, "ops": []},
                    {"coeff": 0.25, "ops": [["z", 0]]},
                ],
            }
        ]
    }

    old_cache = lih_env._PYS_CF_DATA_CACHE
    lih_env._PYS_CF_DATA_CACHE = legacy_data
    try:
        env = LiHEnvironment(R=1.6)
        assert np.isclose(env.full_fci_energy, -6.890117108408502)
        assert np.isclose(env.exact_energy_active, -0.75)
    finally:
        lih_env._PYS_CF_DATA_CACHE = old_cache

def test_lih_compute_energy():
    """验证 compute_energy 函数是否能正常运行。"""
    import tensorcircuit as tc
    env = LiHEnvironment(R=1.6)
    n = env.n_qubits
    
    # 构建一个简单的全零线路（对应 |0000> 态）
    c = tc.Circuit(n)
    
    energy = env.compute_energy(c)
    assert isinstance(energy, (float, np.float64, np.float32, torch.Tensor))
    
def test_lih_fallback():
    """
    即使数据文件被重命名/缺失，环境变量也应能回退到 toy 模型。
    (这个测试会临时模拟数据文件不存在的情况)
    """
    data_path = os.path.join(os.path.dirname(__file__), "../experiments/lih/data/lih_pyscf_data.json")
    backup_path = data_path + ".bak"
    
    if os.path.exists(data_path):
        os.rename(data_path, backup_path)
        try:
            # 清理全局缓存以强制重新加载
            import experiments.lih.env as lih_env
            old_cache = lih_env._PYS_CF_DATA_CACHE
            lih_env._PYS_CF_DATA_CACHE = None
            
            env = LiHEnvironment()
            assert env.name == "LiH_Full"
            assert env.n_qubits == 4
            # Toy 模型的实际精确能量约为 -3.046686
            assert np.isclose(env.exact_energy, -3.046686)
            
            # 恢复缓存以供后续使用（虽然在这里是最后一个测试）
            lih_env._PYS_CF_DATA_CACHE = old_cache
        finally:
            os.rename(backup_path, data_path)
    else:
        # 如果一开始就没有数据文件，直接验证 fallback
        env = LiHEnvironment()
        assert env.name == "LiH_Full"

def test_lih_data_file_uses_explicit_fields():
    """生成的数据文件不应继续依赖模糊的 exact_energy 字段。"""
    data_path = os.path.join(os.path.dirname(__file__), "../experiments/lih/data/lih_pyscf_data.json")
    if not os.path.exists(data_path):
        pytest.skip("lih_pyscf_data.json not found")

    with open(data_path, "r") as f:
        data = json.load(f)

    point = min(data["points"], key=lambda pt: abs(float(pt["R"]) - 1.6))
    assert "active_space_exact_energy" in point
    assert "full_fci_energy" in point
    assert "nuclear_repulsion" in point
    assert "exact_energy" not in point

if __name__ == "__main__":
    pytest.main([__file__])
