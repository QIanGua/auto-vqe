from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class SearchSpec:
    kind: str
    dimensions: Dict[str, list[Any]]
    base_exp_name: str
    run_slug: str
    lr: float
    max_steps: int
    trials_per_config: int
    pop_size: Optional[int] = None
    generations: Optional[int] = None
    mutation_rate: Optional[float] = None
    elite_count: Optional[int] = None


@dataclass(frozen=True)
class BaselineSpec:
    result_label: str
    report_label: str
    run_slug: str
    default_trials: int
    max_steps: int
    lr: float
    seed_base: int
    builder: Callable[[Any, Optional[Dict[str, Any]]], Any]
    config: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class OrchestrationPhase:
    kind: str
    dimensions: Dict[str, list[Any]]
    base_exp_name: str
    max_steps: Optional[int] = None
    trials_per_config: Optional[int] = None
    pop_size: Optional[int] = None
    generations: Optional[int] = None


@dataclass(frozen=True)
class OrchestrationSpec:
    run_slug: str
    max_runs: int
    no_improvement_limit: int
    failure_limit: Optional[int] = None
    phases: tuple[OrchestrationPhase, ...] = ()


@dataclass(frozen=True)
class ExperimentManifest:
    name: str
    system_dir: str
    runs_dir: str
    fallback_config: Dict[str, Any]
    config_priority: tuple[str, ...]
    run_result_label: str
    run_report_label: str
    run_default_trials: int
    run_seed_base: int
    run_max_steps: int
    run_lr: float
    load_env: Callable[[], Any]
    build_circuit: Callable[[Dict[str, Any]], tuple[Callable[..., Any], int]]
    baseline: BaselineSpec
    searches: Dict[str, SearchSpec]
    orchestration: OrchestrationSpec
    notes: Dict[str, Any] = field(default_factory=dict)


def load_best_config(manifest: ExperimentManifest, explicit_path: Optional[str] = None) -> tuple[Dict[str, Any], str]:
    if explicit_path:
        if os.path.exists(explicit_path):
            with open(explicit_path, "r", encoding="utf-8") as f:
                return json.load(f), explicit_path
        print(f"Warning: Explicit config path {explicit_path} not found. Falling back.")

    for relative_path in manifest.config_priority:
        path = os.path.join(manifest.system_dir, relative_path)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path

    return dict(manifest.fallback_config), "fallback_default"


def persist_best_config(manifest: ExperimentManifest, strategy_name: str, best_config: Dict[str, Any]) -> str:
    session_root = os.environ.get("AGENT_VQE_SESSION_DIR")
    if session_root:
        target_path = os.path.join(session_root, strategy_name, f"best_config_{strategy_name}.json")
    else:
        target_path = os.path.join(manifest.system_dir, "presets", f"{strategy_name}.json")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=4)
    return target_path


def run_config_experiment(
    manifest: ExperimentManifest,
    *,
    trials: Optional[int] = None,
    explicit_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    import torch

    from core.evaluator.api import prepare_experiment_dir
    from core.evaluator.logging_utils import log_results, print_results, setup_logger
    from core.evaluator.report import generate_report
    from core.evaluator.training import vqe_train

    env = manifest.load_env()
    ansatz_config, config_path = load_best_config(manifest, explicit_config_path)
    create_circuit, num_params = manifest.build_circuit(ansatz_config)
    exp_dir = prepare_experiment_dir(manifest.runs_dir, f"{manifest.name}_vqe")

    log_path = os.path.join(exp_dir, "run.log")
    logger = setup_logger(log_path)
    logger.info(f"--- {manifest.name.upper()} Experiment ---")
    logger.info(f"Experiment Directory: {exp_dir}")
    logger.info(f"Config Source: {config_path}")
    logger.info(f"Config Content: {ansatz_config}")
    logger.info(f"Target Energy: {env.exact_energy:.6f}")
    logger.info(f"Total Params: {num_params}")

    best_results = None
    overall_best_energy = float("inf")

    def compute_energy_fn(params):
        circuit, _ = create_circuit(params)
        return env.compute_energy(circuit)

    for trial_idx in range(trials or manifest.run_default_trials):
        logger.info(f"\n--- Trial {trial_idx + 1}/{trials or manifest.run_default_trials} ---")
        torch.manual_seed(manifest.run_seed_base + trial_idx)
        results = vqe_train(
            create_circuit_fn=create_circuit,
            compute_energy_fn=compute_energy_fn,
            n_qubits=env.n_qubits,
            exact_energy=env.exact_energy,
            num_params=num_params,
            max_steps=manifest.run_max_steps,
            lr=manifest.run_lr,
            logger=logger,
        )
        if results["val_energy"] < overall_best_energy:
            overall_best_energy = results["val_energy"]
            best_results = results

    logger.info("\n=== Final Best ===")
    print_results(best_results, logger=logger)
    log_results(
        exp_dir,
        manifest.run_result_label,
        best_results,
        comment=f"Config: {ansatz_config}, source={config_path}",
        global_dir=os.path.join(manifest.system_dir, "artifacts"),
    )
    record_path = generate_report(
        exp_dir,
        manifest.run_report_label,
        best_results,
        create_circuit,
        ansatz_spec=ansatz_config,
        config_path=config_path,
    )
    logger.info(f"Run record generated at: {record_path}")
    return best_results


def run_baseline_experiment(manifest: ExperimentManifest, *, trials: Optional[int] = None) -> Dict[str, Any]:
    import json
    import torch

    from core.evaluator.api import prepare_experiment_dir
    from core.evaluator.logging_utils import log_results, print_results, setup_logger
    from core.evaluator.report import generate_report
    from core.evaluator.training import vqe_train

    env = manifest.load_env()
    exp_dir = prepare_experiment_dir(manifest.runs_dir, manifest.baseline.run_slug)

    ansatz_spec = manifest.baseline.builder(env, manifest.baseline.config)
    ansatz_spec_dict = ansatz_spec.to_logging_dict()
    config_path = os.path.join(exp_dir, "config_snapshot.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(ansatz_spec_dict, f, indent=4)

    log_path = os.path.join(exp_dir, "run.log")
    logger = setup_logger(log_path)
    logger.info(f"--- {manifest.name.upper()} Baseline Experiment ---")
    logger.info(f"Baseline family: {ansatz_spec.family}, name: {ansatz_spec.name}")
    logger.info(f"Config: {ansatz_spec.config}")
    logger.info(f"Target Energy: {env.exact_energy:.6f}")

    create_circuit = ansatz_spec.create_circuit
    num_params = ansatz_spec.num_params

    best_results = None
    overall_best_energy = float("inf")

    def compute_energy_fn(params):
        circuit, _ = create_circuit(params)
        return env.compute_energy(circuit)

    total_trials = trials or manifest.baseline.default_trials
    for trial_idx in range(total_trials):
        logger.info(f"\n--- Trial {trial_idx + 1}/{total_trials} ---")
        torch.manual_seed(manifest.baseline.seed_base + trial_idx)
        results = vqe_train(
            create_circuit_fn=create_circuit,
            compute_energy_fn=compute_energy_fn,
            n_qubits=env.n_qubits,
            exact_energy=env.exact_energy,
            num_params=num_params,
            max_steps=manifest.baseline.max_steps,
            lr=manifest.baseline.lr,
            logger=logger,
        )
        if results["val_energy"] < overall_best_energy:
            overall_best_energy = results["val_energy"]
            best_results = results

    logger.info("\n=== Baseline Best ===")
    print_results(best_results, logger=logger)
    log_results(
        exp_dir,
        manifest.baseline.result_label,
        best_results,
        comment=f"Baseline spec: {ansatz_spec_dict}",
    )
    record_path = generate_report(
        exp_dir,
        manifest.baseline.report_label,
        best_results,
        create_circuit,
        ansatz_spec=ansatz_spec_dict,
    )
    logger.info(f"Run record generated at: {record_path}")
    return best_results


def run_search_experiment(manifest: ExperimentManifest, strategy_name: str) -> Dict[str, Any]:
    from core.evaluator.api import prepare_experiment_dir
    from core.generator.ga import GASearchStrategy
    from core.generator.grid import ansatz_search
    from core.representation.search_space import generate_config_grid

    env = manifest.load_env()
    spec = manifest.searches[strategy_name]
    exp_dir = prepare_experiment_dir(manifest.runs_dir, spec.run_slug)

    if spec.kind == "ga":
        result = GASearchStrategy(
            env=env,
            make_circuit_fn=manifest.build_circuit,
            dimensions=spec.dimensions,
            pop_size=spec.pop_size,
            generations=spec.generations,
            mutation_rate=spec.mutation_rate or 0.3,
            elite_count=spec.elite_count or 2,
            trials_per_config=spec.trials_per_config,
            max_steps=spec.max_steps,
            lr=spec.lr,
            exp_dir=exp_dir,
            base_exp_name=spec.base_exp_name,
        ).run()
    elif spec.kind == "multidim":
        result = ansatz_search(
            env=env,
            make_create_circuit_fn=manifest.build_circuit,
            config_list=generate_config_grid(spec.dimensions),
            exp_dir=exp_dir,
            base_exp_name=spec.base_exp_name,
            lr=spec.lr,
            trials_per_config=spec.trials_per_config,
            max_steps=spec.max_steps,
            sub_dir=None,
        )
    else:
        raise ValueError(f"Unknown search kind: {spec.kind}")

    target_path = persist_best_config(manifest, strategy_name, result.get("best_config", {}))
    result["best_config_path"] = target_path
    print(f"\nSynced best {strategy_name} config to: {target_path}")
    return result


def run_orchestration_experiment(manifest: ExperimentManifest) -> list[dict[str, Any]]:
    from core.evaluator.api import prepare_experiment_dir
    from core.generator.ga import GASearchStrategy
    from core.generator.grid import GridSearchStrategy
    from core.orchestration.controller import SearchController, SearchOrchestrator
    from core.representation.search_space import generate_config_grid

    env = manifest.load_env()
    spec = manifest.orchestration
    exp_dir = prepare_experiment_dir(manifest.runs_dir, spec.run_slug)
    controller_kwargs = {
        "max_runs": spec.max_runs,
        "no_improvement_limit": spec.no_improvement_limit,
        "logger": None,
    }
    if spec.failure_limit is not None:
        controller_kwargs["failure_limit"] = spec.failure_limit
    controller = SearchController(**controller_kwargs)

    generators = []
    for phase in spec.phases:
        if phase.kind == "ga":
            generators.append(
                GASearchStrategy(
                    env=env,
                    make_circuit_fn=manifest.build_circuit,
                    dimensions=phase.dimensions,
                    pop_size=phase.pop_size or 6,
                    generations=phase.generations or 2,
                    exp_dir=exp_dir,
                    base_exp_name=phase.base_exp_name,
                    controller=controller,
                )
            )
            continue
        if phase.kind == "multidim":
            generators.append(
                GridSearchStrategy(
                    env=env,
                    make_create_circuit_fn=manifest.build_circuit,
                    config_list=generate_config_grid(phase.dimensions),
                    exp_dir=exp_dir,
                    base_exp_name=phase.base_exp_name,
                    trials_per_config=phase.trials_per_config or 1,
                    max_steps=phase.max_steps or 400,
                    controller=controller,
                )
            )
            continue
        raise ValueError(f"Unknown orchestration phase: {phase.kind}")

    results = SearchOrchestrator(generators=generators, controller=controller).run()
    print(f"\nAuto-Search Completed. Executed {len(results)} strategies.")
    return results


def run_research_step(
    manifest: ExperimentManifest,
    *,
    strategy_name: str,
    verify_trials: int,
) -> Dict[str, Any]:
    result = run_search_experiment(manifest, strategy_name)
    config_path = result.get("best_config_path")
    if not isinstance(config_path, str) or not os.path.exists(config_path):
        raise FileNotFoundError(f"No verifiable config produced for strategy '{strategy_name}'.")

    metrics = run_config_experiment(
        manifest,
        trials=verify_trials,
        explicit_config_path=config_path,
    )
    print("--- METRICS ---")
    print(f"METRIC val_energy={metrics['val_energy']}")
    print(f"METRIC energy_error={metrics['energy_error']}")
    print(f"METRIC num_params={metrics['num_params']}")
    print(f"METRIC selected_strategy={strategy_name}")
    print(f"METRIC selected_config_path={config_path}")
    print("---------------")
    return metrics
