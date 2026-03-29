import sys
import os
import argparse
import json

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
SYSTEM_DIR = os.path.dirname(__file__)
ARTIFACT_RUNS_DIR = os.path.join(SYSTEM_DIR, "artifacts", "runs")


def _load_runtime_dependencies():
    import torch

    from core.evaluator.api import prepare_experiment_dir
    from core.evaluator.logging_utils import log_results, print_results, setup_logger
    from core.evaluator.report import generate_report
    from core.evaluator.training import vqe_train
    from core.representation.compiler import build_ansatz
    from experiments.lih.data.geom_grid import BOND_LENGTHS_ANGSTROM
    from experiments.lih.env import ENV, LiHEnvironment

    return {
        "torch": torch,
        "prepare_experiment_dir": prepare_experiment_dir,
        "log_results": log_results,
        "print_results": print_results,
        "setup_logger": setup_logger,
        "generate_report": generate_report,
        "vqe_train": vqe_train,
        "build_ansatz": build_ansatz,
        "LiHEnvironment": LiHEnvironment,
        "ENV": ENV,
        "BOND_LENGTHS_ANGSTROM": BOND_LENGTHS_ANGSTROM,
    }

# ---- 配置加载逻辑 (优先级: CLI > GA > MultiDim > Fallback) ----
def load_best_config(explicit_path=None):
    exp_dir = SYSTEM_DIR
    if explicit_path:
        if os.path.exists(explicit_path):
            with open(explicit_path, "r") as f:
                config = json.load(f)
            print(f"Loaded EXPLICIT config from {explicit_path}")
            return config, explicit_path
        print(f"Warning: Explicit config path {explicit_path} not found. Falling back.")

    # 按照优先级尝试加载不同策略产出的最优配置
    configs_to_try = [
        "ga/best_config_ga.json", 
        "multidim/best_config_multidim.json",
        "best_config_ga.json", 
        "best_config_multidim.json", 
        "best_config.json"
    ]
    
    for filename in configs_to_try:
        path = os.path.join(exp_dir, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                config = json.load(f)
            print(f"Loaded config from {path}")
            return config, path
            
    # Fallback default config
    return {
        "init_state": "hf",
        "hf_qubits": [0, 1],
        "layers": 2,
        "single_qubit_gates": ["ry", "rz"],
        "two_qubit_gate": "rzz",
        "entanglement": "linear"
    }, "fallback_default"

def get_default_ansatz_bundle():
    runtime = _load_runtime_dependencies()
    ansatz_config, config_path = load_best_config()
    create_circuit, num_params = runtime["build_ansatz"](ansatz_config, runtime["ENV"].n_qubits)
    return ansatz_config, config_path, create_circuit, num_params

def run_experiment(trials=2, explicit_config_path=None): # LiH 耗时较长，Trial 改为 2
    runtime = _load_runtime_dependencies()
    torch = runtime["torch"]
    build_ansatz = runtime["build_ansatz"]
    prepare_experiment_dir = runtime["prepare_experiment_dir"]
    setup_logger = runtime["setup_logger"]
    print_results = runtime["print_results"]
    log_results = runtime["log_results"]
    generate_report = runtime["generate_report"]
    vqe_train = runtime["vqe_train"]
    env = runtime["ENV"]

    ansatz_config, config_path = load_best_config(explicit_config_path)
    create_circuit, num_params = build_ansatz(ansatz_config, env.n_qubits)
    exp_dir = prepare_experiment_dir(ARTIFACT_RUNS_DIR, "lih_vqe")
    
    log_path = os.path.join(exp_dir, "experiment.log")
    logger = setup_logger(log_path)
    logger.info(f"--- LiH Experiment Phase 10 ---")
    logger.info(f"Experiment Directory: {exp_dir}")
    logger.info(f"Config Source: {config_path}")
    logger.info(f"Config Content: {ansatz_config}")
    logger.info(f"Target Energy: {env.exact_energy:.6f}")
    
    best_results = None
    overall_best_energy = float('inf')
    
    def compute_energy_fn(params):
        c, _ = create_circuit(params)
        return env.compute_energy(c)
        
    for i in range(trials):
        logger.info(f"\n--- Trial {i+1}/{trials} ---")
        torch.manual_seed(300 + i)
        
        results = vqe_train(
            create_circuit_fn=create_circuit,
            compute_energy_fn=compute_energy_fn,
            n_qubits=env.n_qubits,
            exact_energy=env.exact_energy,
            num_params=num_params,
            max_steps=800,
            lr=0.05,
            logger=logger
        )
        
        if results['val_energy'] < overall_best_energy:
            overall_best_energy = results['val_energy']
            best_results = results
            
    logger.info("\n=== Final Autonomous Best ===")
    print_results(best_results, logger=logger)
    
    # 记录到实验目录，并同步到实验根目录的汇总表
    log_results(
        exp_dir,
        "LiH_Phase10",
        best_results,
        comment=f"Config: {ansatz_config}, source={config_path}",
        global_dir=SYSTEM_DIR,
    )
    report_path = generate_report(
        exp_dir,
        "LiH_Phase10_Report",
        best_results,
        create_circuit,
        ansatz_spec=ansatz_config,
        config_path=config_path,
    )
    logger.info(f"Report generated at: {report_path}")
    return best_results


def run_geometry_scan(trials_per_R: int = 1, max_steps: int = 800, lr: float = 0.05):
    """
    对一组 Li–H 键长做完整 bond dissociation curve 扫描。
    """
    runtime = _load_runtime_dependencies()
    torch = runtime["torch"]
    prepare_experiment_dir = runtime["prepare_experiment_dir"]
    setup_logger = runtime["setup_logger"]
    log_results = runtime["log_results"]
    vqe_train = runtime["vqe_train"]
    LiHEnvironment = runtime["LiHEnvironment"]
    bond_lengths = runtime["BOND_LENGTHS_ANGSTROM"]

    default_ansatz_config, _, default_create_circuit, default_num_params = get_default_ansatz_bundle()
    exp_dir = prepare_experiment_dir(ARTIFACT_RUNS_DIR, "lih_geom_scan")

    log_path = os.path.join(exp_dir, "experiment.log")
    logger = setup_logger(log_path)
    logger.info(f"--- LiH Geometry Scan (structure transfer) ---")
    logger.info(f"Experiment Directory: {exp_dir}")
    logger.info(f"Ansatz config: {default_ansatz_config}")
    logger.info(f"--- LiH Geometry Scan (structure transfer) ---")
    logger.info(f"Ansatz config: {default_ansatz_config}")

    # 结果收集，用于后续画解离曲线
    scan_results = []

    for R in bond_lengths:
        env = LiHEnvironment(R=R)

        # active-space exact (4-qubit Hamiltonian) vs. full FCI
        exact_active = env.exact_energy  # QuantumEnvironment.exact_energy
        full_fci = getattr(env, "full_fci_energy", exact_active)

        logger.info(f"\n=== Geometry R = {R:.3f} Å ===")
        logger.info(
            "Target energies: "
            f"active_exact={exact_active:.8f}, full_FCI={full_fci:.8f}"
        )

        best_results = None
        overall_best_energy = float("inf")

        def compute_energy_fn(params):
            c, _ = default_create_circuit(params)
            return env.compute_energy(c)

        for t in range(trials_per_R):
            logger.info(f"--- Trial {t+1}/{trials_per_R} at R={R:.3f} Å ---")
            # 用不同 seed 做多次独立训练，考察稳定性
            torch.manual_seed(300 + t)

            results = vqe_train(
                create_circuit_fn=default_create_circuit,
                compute_energy_fn=compute_energy_fn,
                n_qubits=env.n_qubits,
                # 用 active-space exact 评估 ansatz 误差
                exact_energy=exact_active,
                num_params=default_num_params,
                max_steps=max_steps,
                lr=lr,
                logger=logger,
            )

            if results["val_energy"] < overall_best_energy:
                overall_best_energy = results["val_energy"]
                best_results = results

        if best_results is None:
            logger.warning(f"No successful trials for R={R:.3f} Å, skipping.")
            continue

        # 确保带上多种能量与误差，便于统一写入结果和后处理：
        #   - exact_energy / energy_error: ansatz 对 active-space 的误差
        #   - full_fci_energy / error_vs_full: ansatz 相对 full FCI 的误差
        #   - truncation_error: active-space 本身相对 full FCI 的截断误差
        best_results["exact_energy"] = exact_active
        best_results["energy_error"] = abs(best_results["val_energy"] - exact_active)
        best_results["full_fci_energy"] = full_fci
        best_results["error_vs_full"] = abs(best_results["val_energy"] - full_fci)
        best_results["truncation_error"] = abs(exact_active - full_fci)

        logger.info(
            f"Best at R={R:.3f} Å: "
            f"val={best_results['val_energy']:.8f}, "
            f"active_exact={exact_active:.8f}, ansatz_err={best_results['energy_error']:.3e}, "
            f"full_FCI={full_fci:.8f}, err_vs_full={best_results['error_vs_full']:.3e}, "
            f"trunc_err={best_results['truncation_error']:.3e}"
        )

        scan_results.append(
            {
                "R": R,
                "val_energy": float(best_results["val_energy"]),
                "exact_active": float(exact_active),
                "exact_fci": float(full_fci),
                "ansatz_error": float(best_results["energy_error"]),
                "error_vs_fci": float(best_results["error_vs_full"]),
                "truncation_error": float(best_results["truncation_error"]),
                "num_params": int(best_results["num_params"]),
            }
        )

        # 追加到全局 results.tsv，便于统一汇总分析
        comment = f"LiH geometry scan, R={R:.3f} Å"
        log_results(exp_dir, f"LiH_Geom_R_{R:.3f}", best_results, comment=comment)

    # 额外输出一个 TSV 文件专门用于解离曲线绘制
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    curve_path = os.path.join(exp_dir, f"lih_geometry_curve_{timestamp}.tsv")
    with open(curve_path, "w") as f:
        # 列顺序：
        #   R_A, vqe_energy, exact_active, exact_fci, error_vs_fci,
        #   ansatz_error, truncation_error, num_params
        f.write(
            "R_A\tvqe_energy\texact_active\texact_fci\terror_vs_fci\t"
            "ansatz_error\ttruncation_error\tnum_params\n"
        )
        for item in scan_results:
            f.write(
                f"{item['R']:.3f}\t"
                f"{item['val_energy']:.8f}\t"
                f"{item['exact_active']:.8f}\t"
                f"{item['exact_fci']:.8f}\t"
                f"{item['error_vs_fci']:.3e}\t"
                f"{item['ansatz_error']:.3e}\t"
                f"{item['truncation_error']:.3e}\t"
                f"{item['num_params']}\n"
            )

    logger.info(f"\nGeometry scan curve saved to: {curve_path}")
    return scan_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LiH VQE experiment with config.")
    parser.add_argument("--config", type=str, help="Path to explicit ansatz config JSON.")
    parser.add_argument("--trials", type=int, default=2, help="Number of trials with different seeds.")
    args = parser.parse_args()

    # 默认仍然保持原来的单点实验行为；如需扫描整条解离曲线，可以显式调用
    # run_geometry_scan()。
    run_experiment(trials=args.trials, explicit_config_path=args.config)
