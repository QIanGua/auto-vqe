import sys
import os

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.engine import ansatz_search, tc
from experiments.tfim.env import ENV

N_QUBITS = ENV.n_qubits


def make_tfim_create_circuit_fn(config):
    """
    基于配置构造 TFIM 的 ansatz。

    config 字段示例：
      - layers: 纠缠+单比特层数
    """
    layers = int(config.get("layers", 3))

    num_params = layers * N_QUBITS

    def create_circuit(params):
        c = tc.Circuit(N_QUBITS)
        idx = 0
        for _ in range(layers):
            for i in range(N_QUBITS):
                c.ry(i, theta=params[idx])
                idx += 1
            for i in range(N_QUBITS - 1):
                c.cnot(i, i + 1)
            c.cnot(N_QUBITS - 1, 0)
        return c, idx

    return create_circuit, num_params


def run_search():
    # 一个简单的离散搜索空间：仅改变层数
    config_list = [
        {"layers": 1},
        {"layers": 2},
        {"layers": 3},
        {"layers": 4},
    ]

    exp_dir = os.path.dirname(__file__)
    return ansatz_search(
        env=ENV,
        make_create_circuit_fn=make_tfim_create_circuit_fn,
        config_list=config_list,
        exp_dir=exp_dir,
        base_exp_name="TFIM_Search",
        trials_per_config=2,
        max_steps=600,
        lr=0.01,
        logger=None,
    )


if __name__ == "__main__":
    run_search()
