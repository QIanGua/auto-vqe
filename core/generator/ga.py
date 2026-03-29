"""
Genetic search as a generator module.

This keeps the current GA behavior intact while moving the implementation
behind the generator namespace introduced by the Representation / Generator /
Evaluator split.
"""

import json
import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional

from core.orchestration.controller import SearchController
from core.evaluator.logging_utils import print_results, setup_logger
from core.evaluator.report import generate_report
from core.evaluator.training import vqe_train
from core.generator.base import GeneratorState, GeneratorStrategy
from core.representation.search_space import config_to_str, crossover_configs, get_random_config, mutate_config
from core.model.schemas import CandidateSpec, EvaluationResult


def _make_ansatz_spec_dict_for_ga(
    env,
    config: Dict[str, Any],
    num_params: int,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(config)

    return {
        "name": "ga",
        "family": "ga",
        "env_name": getattr(env, "name", "unknown"),
        "n_qubits": getattr(env, "n_qubits", None),
        "num_params": num_params,
        "config": cfg,
        "metadata": {
            "search": "ga",
        },
    }


class GASearchStrategy(GeneratorStrategy):
    def __init__(
        self,
        env,
        make_circuit_fn: Callable,
        dimensions: dict,
        pop_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.3,
        elite_count: int = 2,
        trials_per_config: int = 2,
        max_steps: int = 600,
        lr: float = 0.02,
        exp_dir: str = ".",
        base_exp_name: str = "GA_Search",
        sub_dir: str | None = None,
        logger: logging.Logger | None = None,
        controller: Optional[SearchController] = None,
    ):
        super().__init__(env, controller, logger, name=base_exp_name)
        self.make_circuit_fn = make_circuit_fn
        self.dimensions = dimensions
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.trials_per_config = trials_per_config
        self.max_steps = max_steps
        self.lr = lr
        self.base_exp_name = base_exp_name

        if sub_dir:
            self.exp_dir = os.path.join(exp_dir, sub_dir)
            os.makedirs(self.exp_dir, exist_ok=True)
        else:
            self.exp_dir = exp_dir

        if logger is None:
            log_path = os.path.join(self.exp_dir, "run.log")
            self.logger = setup_logger(log_path)
        else:
            self.logger = logger

        if controller is None:
            self.controller = SearchController(logger=self.logger)
        else:
            self.controller = controller

        self.population: List[dict] = []
        self.results_cache: Dict[str, Dict[str, Any]] = {}
        self.best_overall: Optional[Dict[str, Any]] = None
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_spec: Optional[Dict[str, Any]] = None

    def initialize(self) -> GeneratorState:
        state = GeneratorState(step_count=0)
        self.population = [get_random_config(self.dimensions) for _ in range(self.pop_size)]
        state.metadata["population"] = list(self.population)
        return state

    def _config_to_key(self, config: dict) -> str:
        return config_to_str(config)

    def evaluate(self, config: dict) -> Dict[str, Any]:
        key = self._config_to_key(config)
        if key in self.results_cache:
            return self.results_cache[key]

        if not self.controller.should_continue():
            return {
                "results": {"val_energy": 999.0, "num_params": 999},
                "ansatz_spec": {"name": "skipped"},
            }

        self.logger.info(f"  Evaluating: {key}")

        ansatz_obj = self.make_circuit_fn(config)
        if isinstance(ansatz_obj, tuple) and len(ansatz_obj) == 2:
            create_circuit_fn, num_params = ansatz_obj
            ansatz_spec_dict = _make_ansatz_spec_dict_for_ga(
                self.env, config, num_params
            )
        else:
            create_circuit_fn = getattr(ansatz_obj, "create_circuit")
            num_params = int(getattr(ansatz_obj, "num_params"))
            if hasattr(ansatz_obj, "to_logging_dict"):
                ansatz_spec_dict = ansatz_obj.to_logging_dict()  # type: ignore[attr-defined]
            else:
                ansatz_spec_dict = _make_ansatz_spec_dict_for_ga(
                    self.env, config, num_params
                )

        def compute_energy_fn(params):
            c, _ = create_circuit_fn(params)
            return self.env.compute_energy(c)

        best_trial = None
        proposal_failures = 0
        for _ in range(self.trials_per_config):
            if not self.controller.should_continue():
                break

            seed = random.randint(0, 10000)
            try:
                results = vqe_train(
                    create_circuit_fn=create_circuit_fn,
                    compute_energy_fn=compute_energy_fn,
                    n_qubits=self.env.n_qubits,
                    exact_energy=None,
                    num_params=num_params,
                    max_steps=self.max_steps,
                    lr=self.lr,
                    logger=None,
                    seed=seed,
                )
                self.controller.report_result(results)
                if best_trial is None or results["val_energy"] < best_trial["val_energy"]:
                    best_trial = results
            except Exception as e:
                proposal_failures += 1
                self.logger.error(f"  Trial failed: {e}")
                self.controller.report_result({}, is_failure=True)
                if proposal_failures >= 3:
                    break

        if best_trial is None:
            result = {
                "results": {"val_energy": 100.0, "num_params": num_params},
                "ansatz_spec": ansatz_spec_dict,
            }
            self.results_cache[key] = result
            return result

        best_trial["exact_energy"] = self.env.exact_energy
        best_trial["energy_error"] = abs(
            best_trial["val_energy"] - self.env.exact_energy
        )

        wrapped = {
            "results": best_trial,
            "ansatz_spec": ansatz_spec_dict,
        }
        self.results_cache[key] = wrapped
        return wrapped

    def propose(
        self,
        state: GeneratorState,
        budget: int = 1,
    ) -> List[CandidateSpec]:
        return []

    def observe(
        self,
        state: GeneratorState,
        results: List[EvaluationResult],
    ) -> GeneratorState:
        return state

    def should_stop(self, state: GeneratorState) -> bool:
        return state.step_count >= self.generations

    def run(self, dimensions: Optional[dict] = None) -> dict:
        if dimensions is not None:
            self.dimensions = dimensions
        if self.dimensions is None:
            raise ValueError("Dimensions must be provided to run GASearchStrategy.")

        self.logger.info(f"=== Starting GA Search: {self.base_exp_name} ===")
        self.logger.info(
            f"Population: {self.pop_size}, Gen: {self.generations}, Elite: {self.elite_count}"
        )

        state = self.initialize()
        self.population = list(state.metadata.get("population", []))

        for gen in range(self.generations):
            if not self.controller.should_continue():
                self.logger.info(f"GA search stopped: {self.controller.stop_reason}")
                break

            self.logger.info(f"\n--- Generation {gen + 1}/{self.generations} ---")
            scored_pop = []
            for config in self.population:
                eval_out = self.evaluate(config)
                results = eval_out["results"]
                ansatz_spec = eval_out["ansatz_spec"]
                scored_pop.append((config, results, ansatz_spec))

            scored_pop.sort(key=lambda item: (item[1]["val_energy"], item[1]["num_params"]))

            current_best_config, current_best_res, current_best_spec = scored_pop[0]
            if (
                self.best_overall is None
                or current_best_res["val_energy"] < self.best_overall["val_energy"]
            ):
                self.best_overall = current_best_res
                self.best_config = current_best_config
                self.best_spec = current_best_spec
                self.logger.info(
                    f"New Best Overall: Energy={self.best_overall['val_energy']:.6f}, "
                    f"Params={self.best_overall['num_params']}"
                )

            new_population = [item[0] for item in scored_pop[: self.elite_count]]
            while len(new_population) < self.pop_size:
                if random.random() < 0.7:
                    parent1 = random.choice(scored_pop[: max(3, self.pop_size // 2)])[0]
                    parent2 = random.choice(scored_pop[: max(3, self.pop_size // 2)])[0]
                    child = crossover_configs(parent1, parent2)
                else:
                    parent = random.choice(scored_pop[: max(3, self.pop_size // 2)])[0]
                    child = mutate_config(
                        parent,
                        self.dimensions,
                        mutation_rate=self.mutation_rate,
                    )
                new_population.append(child)

            self.population = new_population
            state.step_count += 1

        self.logger.info("\n=== GA Search Final Best ===")
        if self.best_overall is None or self.best_config is None or self.best_spec is None:
            self.logger.error("No valid results found during GA search.")
            return {}

        print_results(self.best_overall, logger=self.logger)
        self.logger.info(f"Best Config: {config_to_str(self.best_config)}")

        create_fn, _ = self.make_circuit_fn(self.best_config)
        record_path = generate_report(
            self.exp_dir,
            f"{self.base_exp_name}_Best",
            self.best_overall,
            create_fn,
            comment=f"GA Best: {config_to_str(self.best_config)}",
            ansatz_spec=self.best_spec,
        )

        return {
            "best_config": self.best_config,
            "best_results": self.best_overall,
            "record_path": record_path,
            "ansatz_spec": self.best_spec,
        }


GAOptimizer = GASearchStrategy


def ga_search(env, make_circuit_fn, dimensions, sub_dir: str | None = None, **kwargs):
    optimizer = GASearchStrategy(env, make_circuit_fn, dimensions, sub_dir=sub_dir, **kwargs)
    return optimizer.run()
