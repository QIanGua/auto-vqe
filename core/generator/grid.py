"""
Grid / exhaustive search as a generator module.
"""

import datetime
import json
import logging
import os
from typing import Any, Dict, Optional

from core.orchestration.controller import SearchController
from core.evaluator.logging_utils import log_results, print_results, setup_logger, summarize_config
from core.evaluator.report import generate_report
from core.evaluator.training import vqe_train
from core.generator.strategy import SearchStrategy


class GridSearchStrategy(SearchStrategy):
    """
    网格搜索策略。封装已有的 exhaustive config evaluation 流程。
    """

    def __init__(
        self,
        env,
        make_create_circuit_fn,
        config_list,
        exp_dir,
        base_exp_name,
        lr=0.01,
        trials_per_config=3,
        max_steps=1000,
        sub_dir: str | None = None,
        logger: logging.Logger | None = None,
        controller: Optional[SearchController] = None,
    ):
        super().__init__(env, controller, logger, name=base_exp_name)
        self.make_create_circuit_fn = make_create_circuit_fn
        self.config_list = config_list
        self.exp_dir = exp_dir
        self.base_exp_name = base_exp_name
        self.lr = lr
        self.trials_per_config = trials_per_config
        self.max_steps = max_steps
        self.sub_dir = sub_dir

    def run(self) -> dict:
        return ansatz_search(
            env=self.env,
            make_create_circuit_fn=self.make_create_circuit_fn,
            config_list=self.config_list,
            exp_dir=self.exp_dir,
            base_exp_name=self.base_exp_name,
            lr=self.lr,
            trials_per_config=self.trials_per_config,
            max_steps=self.max_steps,
            sub_dir=self.sub_dir,
            logger=self.logger,
            controller=self.controller,
        )


def ansatz_search(
    env,
    make_create_circuit_fn,
    config_list,
    exp_dir,
    base_exp_name,
    lr=0.01,
    trials_per_config=3,
    max_steps=1000,
    sub_dir: str | None = None,
    logger: logging.Logger | None = None,
    controller: Optional[SearchController] = None,
):
    """
    在给定的 ansatz 配置列表上进行自动搜索。
    """
    if sub_dir:
        exp_dir = os.path.join(exp_dir, sub_dir)
        os.makedirs(exp_dir, exist_ok=True)

    if logger is None:
        log_path = os.path.join(
            exp_dir,
            f"{base_exp_name}_search_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        logger = setup_logger(log_path)

    if controller is None:
        controller = SearchController(logger=logger)

    logger.info(f"=== Ansatz Search: {base_exp_name} ===")
    logger.info(f"Target Energy: {env.exact_energy:.6f}")

    best_overall: Optional[Dict[str, Any]] = None
    best_overall_config: Any = None
    best_overall_spec: Optional[Dict[str, Any]] = None

    for idx, config in enumerate(config_list):
        if not controller.should_continue():
            logger.info(f"Search interrupted by controller: {controller.stop_reason}")
            break

        cfg_str = summarize_config(config)
        logger.info(f"\n--- Config {idx + 1}/{len(config_list)}: {cfg_str} ---")

        ansatz_obj = make_create_circuit_fn(config)
        if isinstance(ansatz_obj, tuple) and len(ansatz_obj) == 2:
            create_circuit_fn, num_params = ansatz_obj
            if isinstance(config, dict):
                cfg_dict: Dict[str, Any] = dict(config)
            else:
                cfg_dict = {"raw_config": str(config)}
            ansatz_spec_dict = {
                "name": base_exp_name,
                "family": "multidim",
                "env_name": getattr(env, "name", "unknown"),
                "n_qubits": getattr(env, "n_qubits", None),
                "num_params": num_params,
                "config": cfg_dict,
                "metadata": {
                    "search": "multidim",
                },
            }
        else:
            create_circuit_fn = getattr(ansatz_obj, "create_circuit")
            num_params = int(getattr(ansatz_obj, "num_params"))
            if hasattr(ansatz_obj, "to_logging_dict"):
                ansatz_spec_dict = ansatz_obj.to_logging_dict()  # type: ignore[attr-defined]
            else:
                if isinstance(config, dict):
                    cfg_dict = dict(config)
                else:
                    cfg_dict = {"raw_config": str(config)}
                ansatz_spec_dict = {
                    "name": base_exp_name,
                    "family": "multidim",
                    "env_name": getattr(env, "name", "unknown"),
                    "n_qubits": getattr(env, "n_qubits", None),
                    "num_params": num_params,
                    "config": cfg_dict,
                    "metadata": {
                        "search": "multidim",
                    },
                }

        def compute_energy_fn(params):
            c, _ = create_circuit_fn(params)
            return env.compute_energy(c)

        best_for_cfg = None
        proposal_failures = 0

        for t in range(trials_per_config):
            if not controller.should_continue():
                break

            seed = 1000 + idx * 100 + t
            logger.info(f"  Trial {t + 1}/{trials_per_config}, seed={seed}")

            try:
                results = vqe_train(
                    create_circuit_fn=create_circuit_fn,
                    compute_energy_fn=compute_energy_fn,
                    n_qubits=env.n_qubits,
                    exact_energy=None,
                    num_params=num_params,
                    max_steps=max_steps,
                    lr=lr,
                    logger=logger,
                    seed=seed,
                )
                controller.report_result(results)

                if (best_for_cfg is None) or (results["val_energy"] < best_for_cfg["val_energy"]):
                    best_for_cfg = results

            except Exception as e:
                proposal_failures += 1
                logger.error(f"  Trial {t + 1} failed: {e}")
                controller.report_result({}, is_failure=True)
                if proposal_failures >= 3:
                    logger.warning(f"Too many failures for this proposal ({proposal_failures}), skipping.")
                    break

        if best_for_cfg is not None:
            best_for_cfg["exact_energy"] = env.exact_energy
            best_for_cfg["energy_error"] = abs(best_for_cfg["val_energy"] - env.exact_energy)
            comment = f"config: {cfg_str}"
            log_results(exp_dir, f"{base_exp_name}_cfg{idx}", best_for_cfg, comment=comment)
            logger.info(
                f"Best for config {idx + 1}: val_energy={best_for_cfg['val_energy']:.6f}, "
                f"num_params={best_for_cfg['num_params']}"
            )
        else:
            logger.warning(f"No successful trials for config {idx + 1}")

        def is_better(new, old):
            if new is None:
                return False
            if old is None:
                return True
            if new["val_energy"] + 1e-4 < old["val_energy"]:
                return True
            if abs(new["val_energy"] - old["val_energy"]) <= 1e-4:
                return new["num_params"] < old["num_params"]
            return False

        if is_better(best_for_cfg, best_overall):
            best_overall = best_for_cfg
            best_overall_config = config
            best_overall_spec = ansatz_spec_dict

    logger.info("\n=== Ansatz Search Final Best ===")
    if best_overall_config and best_overall is not None and best_overall_spec is not None:
        logger.info(f"Best config: {summarize_config(best_overall_config)}")
        print_results(best_overall, logger=logger)

        best_ansatz_obj = make_create_circuit_fn(best_overall_config)
        if isinstance(best_ansatz_obj, tuple) and len(best_ansatz_obj) == 2:
            best_create_fn = best_ansatz_obj[0]
        else:
            best_create_fn = getattr(best_ansatz_obj, "create_circuit")

        report_path = generate_report(
            exp_dir,
            f"{base_exp_name}_Best_Report",
            best_overall,
            best_create_fn,
            comment=f"Best config: {summarize_config(best_overall_config)}",
            ansatz_spec=best_overall_spec,
        )
        logger.info(f"Report generated at: {report_path}")

        config_path = os.path.join(exp_dir, "best_config_multidim.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(best_overall_config, f, indent=4)
            logger.info(f"Best MultiDim config saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save best config: {e}")

        return {
            "best_config": best_overall_config,
            "best_results": best_overall,
            "report_path": report_path,
            "ansatz_spec": best_overall_spec,
        }

    logger.error("No valid results found in search.")
    return {}


__all__ = ["GridSearchStrategy", "ansatz_search"]
