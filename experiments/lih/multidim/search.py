"""
LiH 多维网格搜索 (Multi-Dimensional Grid Search)

在结构化 config 空间上做穷举搜索，寻找符合奥卡姆剃刀原则的最优 ansatz：
- 先最小化能量误差
- 能量近似持平时，优先选择参数更少的配置
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from core.engine import ansatz_search  # type: ignore
from core.circuit_factory import build_ansatz, generate_config_grid  # type: ignore
from experiments.lih.env import ENV  # type: ignore


N_QUBITS = ENV.n_qubits
HF_QUBITS = [0, 1]


def make_lih_circuit_fn(config: dict):
    if config.get("init_state") == "hf" and "hf_qubits" not in config:
        config = {**config, "hf_qubits": HF_QUBITS}
    return build_ansatz(config, N_QUBITS)


def run_multidim_search():
    dimensions = {
        "init_state": ["zero", "hf"],
        "layers": [1, 2, 3, 4],
        "single_qubit_gates": [["ry"], ["ry", "rz"]],
        "two_qubit_gate": ["cnot", "rzz"],
        "entanglement": ["linear", "ring"],
    }

    config_list = generate_config_grid(dimensions)

    from core.engine import prepare_experiment_dir

    strategy_dir = os.path.dirname(__file__)
    exp_dir = prepare_experiment_dir(strategy_dir, "lih_multidim_search")

    result = ansatz_search(
        env=ENV,
        make_create_circuit_fn=make_lih_circuit_fn,
        config_list=config_list,
        exp_dir=exp_dir,
        base_exp_name="LiH_MultiDim",
        lr=0.05,
        trials_per_config=1,
        max_steps=600,
        sub_dir=None,
    )

    best_config = result.get("best_config", {})
    import json

    session_root = os.environ.get("AGENT_VQE_SESSION_DIR")
    if session_root:
        target_path = os.path.join(session_root, "multidim", "best_config_multidim.json")
    else:
        target_path = os.path.join(strategy_dir, "best_config_multidim.json")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"\nSynced best MultiDim config to: {target_path}")

    return result

if __name__ == "__main__":
    run_multidim_search()
