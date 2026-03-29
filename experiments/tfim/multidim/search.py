"""
TFIM 多维网格搜索 (Multi-Dimensional Grid Search)

在结构化 config 空间上做穷举搜索，寻找符合奥卡姆剃刀原则的最优 ansatz：
- 先最小化能量误差
- 能量近似持平时，优先选择参数更少的配置
"""

import argparse
import os
import sys
# 将项目根目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

def make_tfim_circuit_fn(config: dict):
    """
    为 TFIM 构造 (create_circuit_fn, num_params)。
    """
    from core.representation.compiler import build_ansatz
    from experiments.tfim.env import ENV

    return build_ansatz(config, ENV.n_qubits)


def run_multidim_search():
    """
    在多维配置网格上进行穷举搜索，并将最优配置写入：
    - experiments/tfim/multidim/best_config_multidim.json
    """
    from core.evaluator.api import prepare_experiment_dir
    from core.generator.grid import ansatz_search
    from core.representation.search_space import generate_config_grid
    from experiments.tfim.env import ENV  # type: ignore

    # 主要考察层数 / 拓扑 / 门集合这三大维度
    dimensions = {
        "layers": [1, 2, 3, 4],
        "single_qubit_gates": [["ry"], ["ry", "rz"]],
        "two_qubit_gate": ["cnot", "rzz"],
        "entanglement": ["linear", "brick"],
    }

    config_list = generate_config_grid(dimensions)

    strategy_dir = os.path.dirname(__file__)
    exp_dir = prepare_experiment_dir(strategy_dir, "tfim_multidim_search")

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

    best_config = result.get("best_config", {})

    import json

    target_path = os.path.join(strategy_dir, "best_config_multidim.json")
    with open(target_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"\nSynced best MultiDim config to: {target_path}")

    return result


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run TFIM multidimensional grid ansatz search.")


def main() -> None:
    build_parser().parse_args()
    run_multidim_search()


if __name__ == "__main__":
    main()
