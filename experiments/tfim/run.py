import argparse
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.shared import (
    BaselineSpec,
    ExperimentManifest,
    OrchestrationPhase,
    OrchestrationSpec,
    SearchSpec,
    run_baseline_experiment,
    run_config_experiment,
    run_orchestration_experiment,
    run_research_step,
    run_search_experiment,
)


SYSTEM_DIR = os.path.dirname(__file__)
RUNS_DIR = os.path.join(SYSTEM_DIR, "artifacts", "runs")


def load_env():
    from experiments.tfim.env import ENV

    return ENV


def build_circuit(config):
    from core.representation.compiler import build_ansatz

    return build_ansatz(config, load_env().n_qubits)


def build_baseline(env, _config):
    from baselines.hea import build_ansatz

    return build_ansatz(env)


def run_100q_mps(max_steps: int = 100, method: str = "COBYLA"):
    from core.evaluator.logging_utils import print_results, setup_logger
    from core.evaluator.scipy_training import scipy_vqe_train

    n_qubits = 100
    env = load_env()(n_qubits=n_qubits, use_mps=True)
    config = {
        "layers": 3,
        "single_qubit_gates": ["ry", "rx"],
        "two_qubit_gate": "rzz",
        "entanglement": "linear",
        "param_strategy": "translational",
        "use_mps": True,
    }
    create_circuit_fn, num_params = build_circuit(config)

    def compute_energy_fn(params):
        circuit, _ = create_circuit_fn(params)
        return env.compute_energy(circuit)

    log_path = os.path.join(RUNS_DIR, "tfim_scale_100q.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path)

    print("Starting 100-qubit TFIM VQE with MPS backend...")
    print(f"Total optimizable parameters: {num_params} (translational tying)")
    start = time.time()
    results = scipy_vqe_train(
        create_circuit_fn=create_circuit_fn,
        compute_energy_fn=compute_energy_fn,
        n_qubits=n_qubits,
        num_params=num_params,
        max_steps=max_steps,
        logger=logger,
        method=method,
    )
    print("\nOptimization completed successfully!")
    print(f"Time Taken to optimize 100 qubits: {time.time() - start:.2f} seconds")
    print_results(results)
    return results


MANIFEST = ExperimentManifest(
    name="tfim",
    system_dir=SYSTEM_DIR,
    runs_dir=RUNS_DIR,
    fallback_config={
        "layers": 2,
        "single_qubit_gates": ["ry"],
        "two_qubit_gate": "rzz",
        "entanglement": "brick",
    },
    config_priority=("presets/ga.json", "presets/multidim.json", "best_config.json"),
    run_result_label="TFIM_ConfigMode",
    run_report_label="TFIM_Phase10_Report",
    run_default_trials=5,
    run_seed_base=200,
    run_max_steps=1500,
    run_lr=0.01,
    load_env=load_env,
    build_circuit=build_circuit,
    baseline=BaselineSpec(
        result_label="TFIM_Baseline",
        report_label="TFIM_Baseline_Report",
        run_slug="tfim_baseline",
        default_trials=5,
        max_steps=1500,
        lr=0.01,
        seed_base=50,
        builder=build_baseline,
    ),
    searches={
        "ga": SearchSpec(
            kind="ga",
            dimensions={
                "layers": [1, 2, 3, 4],
                "single_qubit_gates": [["ry"], ["ry", "rz"]],
                "two_qubit_gate": ["cnot", "cz", "rzz"],
                "entanglement": ["linear", "ring", "brick"],
                "param_strategy": ["independent", "tied"],
            },
            base_exp_name="TFIM_GA_Search",
            run_slug="tfim_ga_search",
            pop_size=12,
            generations=8,
            mutation_rate=0.3,
            elite_count=2,
            trials_per_config=2,
            max_steps=600,
            lr=0.01,
        ),
        "multidim": SearchSpec(
            kind="multidim",
            dimensions={
                "layers": [1, 2, 3, 4],
                "single_qubit_gates": [["ry"], ["ry", "rz"]],
                "two_qubit_gate": ["cnot", "rzz"],
                "entanglement": ["linear", "brick"],
            },
            base_exp_name="TFIM_MultiDim",
            run_slug="tfim_multidim_search",
            trials_per_config=1,
            max_steps=600,
            lr=0.01,
        ),
    },
    orchestration=OrchestrationSpec(
        run_slug="tfim_auto_search",
        max_runs=20,
        no_improvement_limit=5,
        phases=(
            OrchestrationPhase(
                kind="ga",
                dimensions={
                    "layers": [1, 2, 3],
                    "single_qubit_gates": [["ry"], ["ry", "rz"]],
                    "two_qubit_gate": ["cnot", "rzz"],
                    "entanglement": ["linear"],
                },
                base_exp_name="AutoSearch_Phase1_GA",
                pop_size=6,
                generations=3,
            ),
            OrchestrationPhase(
                kind="multidim",
                dimensions={
                    "layers": [1, 2, 3, 4],
                    "single_qubit_gates": [["ry", "rz"]],
                    "two_qubit_gate": ["rzz"],
                    "entanglement": ["linear", "brick"],
                },
                base_exp_name="AutoSearch_Phase2_Grid",
                trials_per_config=1,
                max_steps=400,
            ),
        ),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TFIM experiments from one entrypoint.")
    parser.add_argument("--config", type=str, help="Path to explicit ansatz config JSON.")
    parser.add_argument("--trials", type=int, default=MANIFEST.run_default_trials, help="Number of trials.")

    subparsers = parser.add_subparsers(dest="command")

    verify = subparsers.add_parser("verify", help="Verify the best-known or explicit config.")
    verify.add_argument("--config", type=str, help="Path to explicit ansatz config JSON.")
    verify.add_argument("--trials", type=int, default=MANIFEST.run_default_trials, help="Number of trials.")

    search = subparsers.add_parser("search", help="Run a structural search.")
    search.add_argument("strategy", choices=sorted(MANIFEST.searches.keys()))

    baseline = subparsers.add_parser("baseline", help="Run the baseline ansatz.")
    baseline.add_argument("--trials", type=int, default=MANIFEST.baseline.default_trials, help="Number of trials.")

    subparsers.add_parser("auto", help="Run the orchestration demo.")

    scale = subparsers.add_parser("scale-100q", help="Run the 100-qubit TFIM MPS scaling demo.")
    scale.add_argument("--max-steps", type=int, default=100)
    scale.add_argument("--method", type=str, default="COBYLA")

    research = subparsers.add_parser("research-step", help="Run one research-loop search+verify iteration.")
    research.add_argument("--strategy", required=True, choices=sorted(MANIFEST.searches.keys()))
    research.add_argument("--verify-trials", type=int, default=2)

    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.command is None:
        return run_config_experiment(MANIFEST, trials=args.trials, explicit_config_path=args.config)
    if args.command == "verify":
        return run_config_experiment(MANIFEST, trials=args.trials, explicit_config_path=args.config)
    if args.command == "search":
        return run_search_experiment(MANIFEST, args.strategy)
    if args.command == "baseline":
        return run_baseline_experiment(MANIFEST, trials=args.trials)
    if args.command == "auto":
        return run_orchestration_experiment(MANIFEST)
    if args.command == "scale-100q":
        return run_100q_mps(max_steps=args.max_steps, method=args.method)
    if args.command == "research-step":
        return run_research_step(MANIFEST, strategy_name=args.strategy, verify_trials=args.verify_trials)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
