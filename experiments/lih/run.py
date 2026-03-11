import sys
import os
import torch

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.engine import (
    vqe_train,
    print_results,
    setup_logger,
    log_results,
    generate_report,
    tc,
)
from core.circuit_factory import build_ansatz
from experiments.lih.env import LiHEnvironment, ENV
from experiments.lih.geom_grid import BOND_LENGTHS_ANGSTROM

N_QUBITS = ENV.n_qubits
EXACT_ENERGY = ENV.exact_energy

# ---- 配置加载逻辑 (优先级: GA > MultiDim > Fallback) ----
def load_best_config():
    exp_dir = os.path.dirname(__file__)
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
            import json
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

ANSATZ_CONFIG, CONFIG_PATH = load_best_config()

create_circuit, NUM_PARAMS = build_ansatz(ANSATZ_CONFIG, N_QUBITS)

def run_experiment(trials=2): # LiH 耗时较长，Trial 改为 2
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.dirname(__file__)
    log_path = os.path.join(exp_dir, f"vqe_lih_{timestamp}.log")
    
    logger = setup_logger(log_path)
    logger.info(f"--- LiH Experiment Phase 10 ---")
    logger.info(f"Config Source: {CONFIG_PATH}")
    logger.info(f"Config Content: {ANSATZ_CONFIG}")
    logger.info(f"Target Energy: {EXACT_ENERGY:.6f}")
    
    best_results = None
    overall_best_energy = float('inf')
    
    def compute_energy_fn(params):
        c, _ = create_circuit(params)
        return ENV.compute_energy(c)
        
    for i in range(trials):
        logger.info(f"\n--- Trial {i+1}/{trials} ---")
        torch.manual_seed(300 + i)
        
        results = vqe_train(
            create_circuit_fn=create_circuit,
            compute_energy_fn=compute_energy_fn,
            n_qubits=N_QUBITS,
            exact_energy=EXACT_ENERGY,
            num_params=NUM_PARAMS,
            max_steps=800,
            lr=0.05,
            logger=logger
        )
        
        if results['val_energy'] < overall_best_energy:
            overall_best_energy = results['val_energy']
            best_results = results
            
    logger.info("\n=== Final Autonomous Best ===")
    print_results(best_results, logger=logger)
    
    log_results(exp_dir, "LiH_Phase10", best_results, comment=f"Config: {ANSATZ_CONFIG}, source={CONFIG_PATH}")
    report_path = generate_report(
        exp_dir,
        "LiH_Phase10_Report",
        best_results,
        create_circuit,
        ansatz_spec=ANSATZ_CONFIG,
        config_path=CONFIG_PATH,
    )
    logger.info(f"Report generated at: {report_path}")


def run_geometry_scan(trials_per_R: int = 1, max_steps: int = 800, lr: float = 0.05):
    """
    对一组 Li–H 键长做完整 bond dissociation curve 扫描。

    只考察“结构可迁移性”：同一个 ansatz 结构（ANSATZ_CONFIG）在不同几何上
    各自独立优化参数，看在整个曲线上的误差表现。
    """
    import datetime

    exp_dir = os.path.dirname(__file__)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(exp_dir, f"vqe_lih_geom_scan_{timestamp}.log")

    logger = setup_logger(log_path)
    logger.info(f"--- LiH Geometry Scan (structure transfer) ---")
    logger.info(f"Ansatz config: {ANSATZ_CONFIG}")

    # 结果收集，用于后续画解离曲线
    scan_results = []

    for R in BOND_LENGTHS_ANGSTROM:
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
            c, _ = create_circuit(params)
            return env.compute_energy(c)

        for t in range(trials_per_R):
            logger.info(f"--- Trial {t+1}/{trials_per_R} at R={R:.3f} Å ---")
            # 用不同 seed 做多次独立训练，考察稳定性
            torch.manual_seed(300 + t)

            results = vqe_train(
                create_circuit_fn=create_circuit,
                compute_energy_fn=compute_energy_fn,
                n_qubits=env.n_qubits,
                # 用 active-space exact 评估 ansatz 误差
                exact_energy=exact_active,
                num_params=NUM_PARAMS,
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
    # 默认仍然保持原来的单点实验行为；如需扫描整条解离曲线，可以显式调用
    # run_geometry_scan()。
    run_experiment()
