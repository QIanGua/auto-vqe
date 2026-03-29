"""
LiH Baseline / Benchmark Runner

在一个固定、可解释的 UCCSD baseline 上运行 VQE，用于和 GA / MultiDim
搜索结果做对比。
"""

import os
import sys
import json

import torch

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from core.evaluator.api import prepare_experiment_dir
from core.evaluator.logging_utils import log_results, print_results, setup_logger
from core.evaluator.report import generate_report
from core.evaluator.training import vqe_train
from experiments.lih.env import ENV
from baselines.uccsd import build_ansatz as build_uccsd


N_QUBITS = ENV.n_qubits
EXACT_ENERGY = ENV.exact_energy
RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "runs")

# 固定 LiH baseline 配置。
# 这里保留 HF 初始化，并使用显式 singles + doubles 的最小 UCCSD 骨架。
BASELINE_CONFIG = {
    "init_state": "hf",
    "hf_qubits": [0, 1],
    "occupied_orbitals": [0, 1],
    "virtual_orbitals": [2, 3],
    "layers": 1,
    "include_singles": True,
    "include_doubles": True,
    "mapping": "jordan_wigner",
    "trotter_order": 1,
}


def save_baseline_config(exp_dir: str, ansatz_spec_dict: dict):
    """
    将 baseline AnsatzSpec 以 JSON 形式持久化，便于后续分析和对照。
    """
    path = os.path.join(exp_dir, "baseline_config.json")
    with open(path, "w") as f:
        json.dump(ansatz_spec_dict, f, indent=4)


def run_baseline(trials: int = 5):
    """
    在固定 UCCSD baseline 配置上运行多次 VQE，并记录最优结果。
    """
    exp_dir = prepare_experiment_dir(RUNS_DIR, "lih_baseline_uccsd")

    uccsd_spec = build_uccsd(ENV, BASELINE_CONFIG)
    uccsd_spec_dict = uccsd_spec.to_logging_dict()
    save_baseline_config(exp_dir, uccsd_spec_dict)

    log_path = os.path.join(exp_dir, "experiment.log")
    logger = setup_logger(log_path)
    logger.info("--- LiH UCCSD Baseline Experiment ---")
    logger.info(f"Baseline family: {uccsd_spec.family}, name: {uccsd_spec.name}")
    logger.info(f"Config: {uccsd_spec.config}")
    logger.info(
        "Excitations: %s singles + %s doubles",
        uccsd_spec.metadata.get("singles_count"),
        uccsd_spec.metadata.get("doubles_count"),
    )
    logger.info(f"Target Energy: {EXACT_ENERGY:.6f}")

    create_circuit = uccsd_spec.create_circuit
    num_params = uccsd_spec.num_params

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

    logger.info("\n=== LiH UCCSD Baseline Best ===")
    print_results(best_results, logger=logger)

    log_results(
        exp_dir,
        "LiH_UCCSD_Baseline",
        best_results,
        comment=f"Baseline spec: {uccsd_spec_dict}",
    )
    report_path = generate_report(
        exp_dir,
        "LiH_UCCSD_Baseline_Report",
        best_results,
        create_circuit,
        ansatz_spec=uccsd_spec_dict,
    )
    logger.info(f"Report generated at: {report_path}")

    return best_results


if __name__ == "__main__":
    run_baseline()
