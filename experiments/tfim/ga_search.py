"""
TFIM 遗传算法 (GA) 搜索

不再通过穷举网格，而是通过进化算法在多维空间中寻找最优 ansatz。
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.search_algorithms import ga_search
from core.circuit_factory import build_ansatz, SEARCH_DIMENSIONS
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

    from core.engine import prepare_experiment_dir
    base_dir = os.path.dirname(__file__)
    exp_dir = prepare_experiment_dir(base_dir, "tfim_ga_search")
    
    # 使用 GASearchStrategy 对象化方式运行
    from core.search_algorithms import GASearchStrategy
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
    
    return strategy.run()

if __name__ == "__main__":
    run_ga_search()
