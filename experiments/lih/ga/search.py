"""
LiH 遗传算法 (GA) 搜索

利用进化算法在化学 ansatz 空间中高效搜索。
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

HF_QUBITS = [0, 1]
RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "runs")


def prepare_experiment_dir(*args, **kwargs):
    from core.evaluator.api import prepare_experiment_dir as _prepare_experiment_dir

    return _prepare_experiment_dir(*args, **kwargs)


class GASearchStrategy:
    def __new__(cls, *args, **kwargs):
        from core.generator.ga import GASearchStrategy as _GASearchStrategy

        return _GASearchStrategy(*args, **kwargs)


def make_lih_circuit_fn(config):
    from core.representation.compiler import build_ansatz
    from experiments.lih.env import ENV

    n_qubits = ENV.n_qubits
    if config.get("init_state") == "hf" and "hf_qubits" not in config:
        config = {**config, "hf_qubits": HF_QUBITS}
    return build_ansatz(config, n_qubits)


def run_ga_search():
    from experiments.lih.env import ENV

    dimensions = {
        "init_state": ["zero", "hf"],
        "layers": [2, 3, 4, 5],
        "single_qubit_gates": [["ry"], ["ry", "rz"], ["rx", "ry", "rz"]],
        "two_qubit_gate": ["cnot", "rzz", "rxx_ryy_rzz"],
        "entanglement": ["linear", "ring", "full"],
    }

    strategy_dir = os.path.dirname(__file__)
    exp_dir = prepare_experiment_dir(RUNS_DIR, "lih_ga_search")

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
        base_exp_name="LiH_GA_Search",
    )

    result = strategy.run()
    best_config = result.get("best_config", {})

    import json

    session_root = os.environ.get("AGENT_VQE_SESSION_DIR")
    if session_root:
        target_path = os.path.join(session_root, "ga", "best_config_ga.json")
    else:
        target_path = os.path.join(strategy_dir, "best_config_ga.json")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"\nSynced best GA config to: {target_path}")

    return result


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run LiH GA ansatz search.")


def main() -> None:
    build_parser().parse_args()
    run_ga_search()


if __name__ == "__main__":
    main()
