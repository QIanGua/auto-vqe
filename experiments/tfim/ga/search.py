"""
TFIM 遗传算法 (GA) 搜索

不再通过穷举网格，而是通过进化算法在多维空间中寻找最优 ansatz。
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from core.evaluator.api import prepare_experiment_dir
from core.generator.ga import GASearchStrategy
from core.representation.compiler import build_ansatz
from experiments.tfim.env import ENV

N_QUBITS = ENV.n_qubits

def make_tfim_circuit_fn(config):
    return build_ansatz(config, N_QUBITS)

def run_ga_search():
    # 定义搜索空间的范围
    dimensions = {
        "layers": [1, 2, 3, 4],
        "single_qubit_gates": [["ry"], ["ry", "rz"]],
        "two_qubit_gate": ["cnot", "cz", "rzz"],
        "entanglement": ["linear", "ring", "brick"],
        "param_strategy": ["independent", "tied"],
    }

    strategy_dir = os.path.dirname(__file__)
    exp_dir = prepare_experiment_dir(strategy_dir, "tfim_ga_search")
    
    # 使用 GASearchStrategy 对象化方式运行
    strategy = GASearchStrategy(
        env=ENV,
        make_circuit_fn=make_tfim_circuit_fn,
        dimensions=dimensions,
        pop_size=12,
        generations=8,
        mutation_rate=0.3,
        elite_count=2,
        trials_per_config=2,
        max_steps=600,
        lr=0.01,
        exp_dir=exp_dir,
        base_exp_name="TFIM_GA_Search"
    )
    result = strategy.run()
    best_config = result.get("best_config", {})

    import json

    target_path = os.path.join(strategy_dir, "best_config_ga.json")
    with open(target_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"\nSynced best GA config to: {target_path}")

    return result

if __name__ == "__main__":
    run_ga_search()
