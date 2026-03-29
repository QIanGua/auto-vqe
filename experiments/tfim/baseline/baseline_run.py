"""
TFIM Baseline / Benchmark Runner

在一个固定、可解释的基线 Ansatz 上运行 VQE，用于和 GA / MultiDim 搜索结果做对比。
"""

import os
import sys
import json
import datetime

import torch

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from core.evaluator.api import prepare_experiment_dir
from core.evaluator.logging_utils import log_results, print_results, setup_logger
from core.evaluator.report import generate_report
from core.evaluator.training import vqe_train
from experiments.tfim.env import ENV
from baselines.hea import build_ansatz as build_hea


N_QUBITS = ENV.n_qubits
EXACT_ENERGY = ENV.exact_energy


def save_baseline_config(exp_dir: str, ansatz_spec_dict: dict):
    """
    将 baseline AnsatzSpec 以 JSON 形式持久化，便于后续分析和对照。

    这里直接保存 `AnsatzSpec.to_logging_dict()` 的结果，而不是仅仅
    保存 config，这样在离线分析时可以看到 family/name 等元信息。
    """
    path = os.path.join(exp_dir, "baseline_config.json")
    with open(path, "w") as f:
        json.dump(ansatz_spec_dict, f, indent=4)


def run_baseline(trials: int = 5):
    """
    在固定 Baseline 配置上运行多次 VQE，并记录最优结果。
    """
    base_dir = os.path.dirname(__file__)
    exp_dir = prepare_experiment_dir(base_dir, "tfim_baseline")

    # 使用 Baseline Zoo 中的 HEA 基线构造 ansatz，并持久化其结构化描述。
    hea_spec = build_hea(ENV)
    hea_spec_dict = hea_spec.to_logging_dict()
    save_baseline_config(exp_dir, hea_spec_dict)

    log_path = os.path.join(exp_dir, "experiment.log")
    logger = setup_logger(log_path)
    logger.info("--- TFIM Baseline Experiment ---")
    logger.info(f"Baseline family: {hea_spec.family}, name: {hea_spec.name}")
    logger.info(f"Config: {hea_spec.config}")
    logger.info(f"Target Energy: {EXACT_ENERGY:.6f}")

    create_circuit = hea_spec.create_circuit
    num_params = hea_spec.num_params

    best_results = None
    overall_best_energy = float("inf")

    def compute_energy_fn(params):
        c, _ = create_circuit(params)
        return ENV.compute_energy(c)

    for i in range(trials):
        logger.info(f"\n--- Trial {i+1}/{trials} ---")
        torch.manual_seed(50 + i)

        results = vqe_train(
            create_circuit_fn=create_circuit,
            compute_energy_fn=compute_energy_fn,
            n_qubits=N_QUBITS,
            exact_energy=EXACT_ENERGY,
            num_params=num_params,
            max_steps=1500,
            lr=0.01,
            logger=logger,
        )

        if results["val_energy"] < overall_best_energy:
            overall_best_energy = results["val_energy"]
            best_results = results

    logger.info("\n=== TFIM Baseline Best ===")
    print_results(best_results, logger=logger)

    # 记录到 baseline 目录自己的 results.tsv
    log_results(
        exp_dir,
        "TFIM_Baseline",
        best_results,
        comment=f"Baseline spec: {hea_spec_dict}",
    )
    report_path = generate_report(
        exp_dir,
        "TFIM_Baseline_Report",
        best_results,
        create_circuit,
        # 在 results.jsonl 中记录统一的 AnsatzSpec 描述
        ansatz_spec=hea_spec_dict,
    )
    logger.info(f"Report generated at: {report_path}")

    return best_results


if __name__ == "__main__":
    run_baseline()
