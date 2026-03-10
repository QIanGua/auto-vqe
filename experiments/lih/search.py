import sys
import os

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.engine import ansatz_search, tc
from experiments.lih.env import ENV

N_QUBITS = ENV.n_qubits


def make_lih_create_circuit_fn(config):
    """
    基于配置构造 LiH 的 ansatz。

    config 字段示例：
      - layers: 叠加的参数化层数
      - use_hf_init: 是否加入 Hartree–Fock 初始化
    """
    layers = int(config.get("layers", 4))
    use_hf_init = bool(config.get("use_hf_init", True))

    # 每层参数：
    # - 对每个 qubit 的 RY/RZ: 2 * N_QUBITS
    # - 对每个 (i, j) 二比特对的 RXX/RYY/RZZ: 3 * C(N_QUBITS, 2)
    # - 对每个 qubit 的 RX: N_QUBITS
    num_params_per_layer = 2 * N_QUBITS + 3 * (N_QUBITS * (N_QUBITS - 1) // 2) + N_QUBITS
    num_params = layers * num_params_per_layer

    def create_circuit(params):
        c = tc.Circuit(N_QUBITS)
        if use_hf_init:
            # Hartree–Fock 初始化
            c.x(0)
            c.x(1)
        idx = 0
        for _ in range(layers):
            for i in range(N_QUBITS):
                c.ry(i, theta=params[idx])
                idx += 1
                c.rz(i, theta=params[idx])
                idx += 1
            for i in range(N_QUBITS):
                for j in range(i + 1, N_QUBITS):
                    c.rxx(i, j, theta=params[idx])
                    idx += 1
                    c.ryy(i, j, theta=params[idx])
                    idx += 1
                    c.rzz(i, j, theta=params[idx])
                    idx += 1
            for i in range(N_QUBITS):
                c.rx(i, theta=params[idx])
                idx += 1
        return c, idx

    return create_circuit, num_params


def run_search():
    # 简单的离散搜索空间：层数 & 是否使用 HF 初始化
    config_list = [
        {"layers": 2, "use_hf_init": True},
        {"layers": 3, "use_hf_init": True},
        {"layers": 4, "use_hf_init": True},
        {"layers": 3, "use_hf_init": False},
    ]

    exp_dir = os.path.dirname(__file__)
    return ansatz_search(
        env=ENV,
        make_create_circuit_fn=make_lih_create_circuit_fn,
        config_list=config_list,
        exp_dir=exp_dir,
        base_exp_name="LiH_Search",
        trials_per_config=2,
        max_steps=600,
        lr=0.05,
        logger=None,
    )


if __name__ == "__main__":
    run_search()
