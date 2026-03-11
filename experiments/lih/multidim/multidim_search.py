"""
LiH 多维网格搜索 (Multi-Dimensional Grid Search)

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
from experiments.lih.env import ENV  # type: ignore


N_QUBITS = ENV.n_qubits
HF_QUBITS = [0, 1]


def make_lih_circuit_fn(config: dict):
    """
    为 LiH 构造 (create_circuit_fn, num_params)。
    自动注入 HF 初始态所需的 hf_qubits。
    """
    if config.get("init_state") == "hf" and "hf_qubits" not in config:
        config = {**config, "hf_qubits": HF_QUBITS}
    return build_ansatz(config, N_QUBITS)


def run_multidim_search():
    """
    在多维配置网格上进行穷举搜索，并将最优配置写入：
    - experiments/lih/multidim/best_config_multidim.json
    """
    # 适中的搜索空间：兼顾表达力与可计算性
    dimensions = {
        "init_state": ["zero", "hf"],
        "layers": [1, 2, 3, 4],
        "single_qubit_gates": [["ry"], ["ry", "rz"]],
        "two_qubit_gate": ["cnot", "rzz"],
        "entanglement": ["linear", "ring"],
    }

    config_list = generate_config_grid(dimensions)

    exp_dir = os.path.dirname(__file__)  # multidim 子目录

    result = ansatz_search(
        env=ENV,
        make_create_circuit_fn=make_lih_circuit_fn,
        config_list=config_list,
        exp_dir=exp_dir,
        base_exp_name="LiH_MultiDim",
        lr=0.05,
        trials_per_config=1,
        max_steps=600,
        sub_dir=None,  # 直接在 multidim/ 下写日志和报告
    )

    return result


if __name__ == "__main__":
    run_multidim_search()

