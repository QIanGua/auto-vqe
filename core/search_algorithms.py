"""
遗传算法搜索 (Genetic Algorithm Search)

基于 circuit_factory 定义的离散空间，通过进化策略寻找最优 ansatz 配置。

本模块已与 Baseline Zoo 的 `AnsatzSpec` 对齐：内部会为每个配置构造一个
标准化的 ansatz 说明字典，写入 `results.jsonl` 的 `ansatz_spec` 字段。
"""

import os
import random
import datetime
import json
import logging
from typing import Callable, Any, Dict, Optional, Tuple

from core.engine import (
    vqe_train,
    setup_logger,
    log_results,
    generate_report,
    print_results,
)
from core.circuit_factory import (
    build_ansatz,
    mutate_config,
    crossover_configs,
    get_random_config,
    config_to_str,
)

from core.controller import SearchController


def _make_ansatz_spec_dict_for_ga(
    env,
    config: Dict[str, Any],
    num_params: int,
) -> Dict[str, Any]:
    """
    将 GA 内部使用的裸 config dict 封装为统一的 AnsatzSpec 风格字典。

    这与 `baselines.AnsatzSpec.to_logging_dict()` 保持字段级兼容：
      - name/family 固定标记为 "ga"
      - env_name/n_qubits 来自 env
      - config 即 GA 所探索的结构化配置
    """
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


class GAOptimizer:
    def __init__(
        self,
        env,
        make_circuit_fn: Callable,
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
        self.env = env
        self.make_circuit_fn = make_circuit_fn # build_ansatz wrapping
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.trials_per_config = trials_per_config
        self.max_steps = max_steps
        self.lr = lr
        self.base_exp_name = base_exp_name
        
        # Resolve the actual experiment output directory.
        if sub_dir:
            self.exp_dir = os.path.join(exp_dir, sub_dir)
            os.makedirs(self.exp_dir, exist_ok=True)
        else:
            self.exp_dir = exp_dir
        
        if logger is None:
            log_path = os.path.join(
                self.exp_dir,
                f"{base_exp_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            )
            self.logger = setup_logger(log_path)
        else:
            self.logger = logger
            
        if controller is None:
            self.controller = SearchController(logger=self.logger)
        else:
            self.controller = controller

        self.population = []  # List[dict] of configs
        # config_str -> {"results": dict, "ansatz_spec": dict}
        self.results_cache: Dict[str, Dict[str, Any]] = {}
        self.best_overall: Optional[Dict[str, Any]] = None
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_spec: Optional[Dict[str, Any]] = None

    def _config_to_key(self, config: dict) -> str:
        return config_to_str(config)

    def evaluate(self, config: dict) -> Dict[str, Any]:
        key = self._config_to_key(config)
        if key in self.results_cache:
            return self.results_cache[key]

        if not self.controller.should_continue():
            return {"val_energy": 999.0, "num_params": 999}

        self.logger.info(f"  Evaluating: {key}")

        # make_circuit_fn 最初返回 (create_circuit_fn, num_params)，
        # 为保持向后兼容，这里仍支持该形式；若未来改为返回 AnsatzSpec，
        # 也可在此处扩展分支逻辑。
        ansatz_obj = self.make_circuit_fn(config)
        if isinstance(ansatz_obj, tuple) and len(ansatz_obj) == 2:
            create_circuit_fn, num_params = ansatz_obj
            ansatz_spec_dict = _make_ansatz_spec_dict_for_ga(
                self.env, config, num_params
            )
        else:
            # 宽松兼容：若调用方改为返回一个带 create_circuit/num_params 的对象
            #（例如 baselines.AnsatzSpec），则在此动态解析。
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
        for t in range(self.trials_per_config):
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
                    logger=None, # Silence per-step logs in GA
                    seed=seed,
                )
                self.controller.report_result(results)

                if best_trial is None or results["val_energy"] < best_trial["val_energy"]:
                    best_trial = results
            except Exception as e:
                proposal_failures += 1
                self.logger.error(f"  Trial {t+1} failed: {e}")
                self.controller.report_result({}, is_failure=True)
                if proposal_failures >= 3:
                    break

        if best_trial is None:
            # 仍然返回结构化字典，以避免调用端崩溃
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

    def run(self, dimensions: dict) -> dict:
        self.logger.info(f"=== Starting GA Search: {self.base_exp_name} ===")
        self.logger.info(f"Population: {self.pop_size}, Gen: {self.generations}, Elite: {self.elite_count}")

        # 1. Initialize Population
        self.population = [get_random_config(dimensions) for _ in range(self.pop_size)]

        for gen in range(self.generations):
            if not self.controller.should_continue():
                self.logger.info(f"GA search stopped: {self.controller.stop_reason}")
                break

            self.logger.info(f"\n--- Generation {gen+1}/{self.generations} ---")
            
            # 2. Evaluate
            scored_pop = []
            for config in self.population:
                eval_out = self.evaluate(config)
                results = eval_out["results"]
                ansatz_spec = eval_out["ansatz_spec"]
                scored_pop.append((config, results, ansatz_spec))

            # 3. Sort by Pareto logic
            def rank_key(item):
                _, res, _ = item
                return (res["val_energy"], res["num_params"])

            scored_pop.sort(key=rank_key)
            
            # Update best
            current_best_config, current_best_res, current_best_spec = scored_pop[0]
            if self.best_overall is None or current_best_res["val_energy"] < self.best_overall["val_energy"]:
                self.best_overall = current_best_res
                self.best_config = current_best_config
                self.best_spec = current_best_spec
                self.logger.info(f"New Best Overall: Energy={self.best_overall['val_energy']:.6f}, Params={self.best_overall['num_params']}")

            # 4. Selection & Evolution
            new_population = [item[0] for item in scored_pop[:self.elite_count]]  # Elites
            
            while len(new_population) < self.pop_size:
                if random.random() < 0.7: # Crossover
                    parent1 = random.choice(scored_pop[:max(3, self.pop_size//2)])[0]
                    parent2 = random.choice(scored_pop[:max(3, self.pop_size//2)])[0]
                    child = crossover_configs(parent1, parent2)
                else: # Mutation
                    parent = random.choice(scored_pop[:max(3, self.pop_size//2)])[0]
                    child = mutate_config(parent, dimensions, mutation_rate=self.mutation_rate)
                
                new_population.append(child)
            
            self.population = new_population

        self.logger.info("\n=== GA Search Final Best ===")
        if self.best_overall is None or self.best_config is None or self.best_spec is None:
            self.logger.error("No valid results found during GA search.")
            return {}
            
        print_results(self.best_overall, logger=self.logger)
        self.logger.info(f"Best Config: {config_to_str(self.best_config)}")
        
        # Save best config to JSON (Use GA specific name)
        config_path = os.path.join(self.exp_dir, "best_config_ga.json")
        with open(config_path, "w") as f:
            json.dump(self.best_config, f, indent=4)
        self.logger.info(f"Best GA config saved to {config_path}")

        # Final report
        create_fn, _ = self.make_circuit_fn(self.best_config)
        report_path = generate_report(
            self.exp_dir,
            f"{self.base_exp_name}_Best",
            self.best_overall,
            create_fn,
            comment=f"GA Best: {config_to_str(self.best_config)}",
            # 写入统一格式的 ansatz_spec 字典，便于后续和 Baseline Zoo 对齐分析
            ansatz_spec=self.best_spec,
        )
        
        return {
            "best_config": self.best_config,
            "best_results": self.best_overall,
            "report_path": report_path,
            "ansatz_spec": self.best_spec,
        }

def ga_search(env, make_circuit_fn, dimensions, sub_dir: str | None = None, **kwargs):
    optimizer = GAOptimizer(env, make_circuit_fn, sub_dir=sub_dir, **kwargs)
    return optimizer.run(dimensions)
