"""
LiH 遗传算法 (GA) 搜索

利用进化算法在化学 ansatz 空间中高效搜索。
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.search_algorithms import GASearchStrategy
from core.circuit_factory import build_ansatz
from experiments.lih.env import ENV

N_QUBITS = ENV.n_qubits
HF_QUBITS = [0, 1]

def make_lih_circuit_fn(config):
    if config.get("init_state") == "hf" and "hf_qubits" not in config:
        config = {**config, "hf_qubits": HF_QUBITS}
    return build_ansatz(config, N_QUBITS)

def run_ga_search():
    # LiH 的搜索空间通常比 TFIM 更复杂
    dimensions = {
        "init_state": ["zero", "hf"],
        "layers": [2, 3, 4, 5],
        "single_qubit_gates": [["ry"], ["ry", "rz"], ["rx", "ry", "rz"]],
        "two_qubit_gate": ["cnot", "rzz", "rxx_ryy_rzz"],
        "entanglement": ["linear", "ring", "full"],
    }

    from core.engine import prepare_experiment_dir
    base_dir = os.path.dirname(__file__)
    exp_dir = prepare_experiment_dir(base_dir, "lih_ga_search")
    
    # 使用 GASearchStrategy 对象化方式运行
    strategy = GASearchStrategy(
        env=ENV,
        make_circuit_fn=make_lih_circuit_fn,
        dimensions=dimensions,
        pop_size=10,
        generations=6,
        mutation_rate=0.4,
        elite_count=2,
        trials_per_config=2,
        max_steps=600,
        lr=0.05,
        exp_dir=exp_dir,
        base_exp_name="LiH_GA_Search"
    )
    
    return strategy.run()

if __name__ == "__main__":
    run_ga_search()
