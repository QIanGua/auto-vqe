"""
TFIM 多维网格搜索 (Multi-Dimensional Grid Search)

在结构化 config 空间上做穷举搜索，寻找符合奥卡姆剃刀原则的最优 ansatz：
- 先最小化能量误差
- 能量近似持平时，优先选择参数更少的配置
"""

import os
import sys
import json

# 将项目根目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from core.engine import ansatz_search  # type: ignore
from core.circuit_factory import build_ansatz, generate_config_grid  # type: ignore
from experiments.tfim.env import ENV  # type: ignore


N_QUBITS = ENV.n_qubits


def make_tfim_circuit_fn(config: dict):
    """
    为 TFIM 构造 (create_circuit_fn, num_params)。
    """
    return build_ansatz(config, N_QUBITS)


def run_multidim_search():
    """
    在多维配置网格上进行穷举搜索，并将最优配置写入：
    - experiments/tfim/multidim/best_config_multidim.json
    """
    # 主要考察层数 / 拓扑 / 门集合这三大维度
    dimensions = {
        "layers": [1, 2, 3, 4],
        "single_qubit_gates": [["ry"], ["ry", "rz"]],
        "two_qubit_gate": ["cnot", "rzz"],
        "entanglement": ["linear", "brick"],
    }

    config_list = generate_config_grid(dimensions)

    exp_dir = os.path.dirname(__file__)  # multidim 子目录

    result = ansatz_search(
        env=ENV,
        make_create_circuit_fn=make_tfim_circuit_fn,
        config_list=config_list,
        exp_dir=exp_dir,
        base_exp_name="TFIM_MultiDim",
        lr=0.01,
        trials_per_config=1,
        max_steps=600,
        sub_dir=None,  # 直接在 multidim/ 下写日志和报告
    )

    best_config = result["best_config"]
    if best_config is not None:
        out_path = os.path.join(exp_dir, "best_config_multidim.json")
        with open(out_path, "w") as f:
            json.dump(best_config, f, indent=4)

    return result


if __name__ == "__main__":
    run_multidim_search()

