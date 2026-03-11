"""
LiH Baseline / Benchmark Runner

在一个固定、可解释的基线 Ansatz 上运行 VQE，用于和 GA / MultiDim 搜索结果做对比。
"""

import os
import sys
import json
import datetime

import torch

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from core.engine import vqe_train, print_results, setup_logger, log_results, generate_report  # type: ignore
from experiments.lih.env import ENV  # type: ignore
from baselines.uccsd import build_ansatz as build_uccsd  # type: ignore


N_QUBITS = ENV.n_qubits
EXACT_ENERGY = ENV.exact_energy

# --- 固定 Baseline 配置 (覆盖 UCCSD 默认设置以匹配历史 LiH 基线) ---
BASELINE_CONFIG = {
    "init_state": "hf",
    "hf_qubits": [0, 1],
    "layers": 2,
    "single_qubit_gates": ["ry", "rz"],
    "two_qubit_gate": "rzz",
    "entanglement": "linear",
}


def save_baseline_config(exp_dir: str, ansatz_spec_dict: dict):
    """
    将 baseline AnsatzSpec 以 JSON 形式持久化，便于后续分析和对照。

    这里直接保存 `AnsatzSpec.to_logging_dict()` 的结果，以便离线分析时
    能看到 family/name/config 等完整信息。
    """
    path = os.path.join(exp_dir, "baseline_config.json")
    with open(path, "w") as f:
        json.dump(ansatz_spec_dict, f, indent=4)


def run_baseline(trials: int = 3):
    """
    在固定 Baseline 配置上运行多次 VQE，并记录最优结果。
    """
    exp_dir = os.path.dirname(__file__)
    os.makedirs(exp_dir, exist_ok=True)

    # 使用 Baseline Zoo 中的 UCCSD 基线构造 ansatz，并持久化其结构化描述。
    # 通过 BASELINE_CONFIG 覆盖 UCCSD 默认配置，以保持与历史 LiH 基线一致。
    uccsd_spec = build_uccsd(ENV, BASELINE_CONFIG)
    uccsd_spec_dict = uccsd_spec.to_logging_dict()
    save_baseline_config(exp_dir, uccsd_spec_dict)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(exp_dir, f"baseline_lih_{timestamp}.log")

    logger = setup_logger(log_path)
    logger.info("--- LiH Baseline Experiment ---")
    logger.info(f"Baseline family: {uccsd_spec.family}, name: {uccsd_spec.name}")
    logger.info(f"Config: {uccsd_spec.config}")
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
        torch.manual_seed(100 + i)

        results = vqe_train(
            create_circuit_fn=create_circuit,
            compute_energy_fn=compute_energy_fn,
            n_qubits=N_QUBITS,
            exact_energy=EXACT_ENERGY,
            num_params=num_params,
            max_steps=800,
            lr=0.05,
            logger=logger,
        )

        if results["val_energy"] < overall_best_energy:
            overall_best_energy = results["val_energy"]
            best_results = results

    logger.info("\n=== LiH Baseline Best ===")
    print_results(best_results, logger=logger)

    # 记录到 baseline 目录自己的 results.tsv
    log_results(
        exp_dir,
        "LiH_Baseline",
        best_results,
        comment=f"Baseline spec: {uccsd_spec_dict}",
    )
    report_path = generate_report(
        exp_dir,
        "LiH_Baseline_Report",
        best_results,
        create_circuit,
        # 在 results.jsonl 中记录统一的 AnsatzSpec 描述
        ansatz_spec=uccsd_spec_dict,
    )
    logger.info(f"Report generated at: {report_path}")

    return best_results


if __name__ == "__main__":
    run_baseline()
