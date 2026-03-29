import os
import sys
import argparse
import json

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def _load_runtime_dependencies():
    import torch

    from core.evaluator.api import prepare_experiment_dir
    from core.evaluator.logging_utils import log_results, print_results, setup_logger
    from core.evaluator.report import generate_report
    from core.evaluator.training import vqe_train
    from core.representation.compiler import build_ansatz
    from experiments.tfim.env import ENV

    return {
        "torch": torch,
        "prepare_experiment_dir": prepare_experiment_dir,
        "log_results": log_results,
        "print_results": print_results,
        "setup_logger": setup_logger,
        "generate_report": generate_report,
        "vqe_train": vqe_train,
        "build_ansatz": build_ansatz,
        "ENV": ENV,
    }

# ---- 配置加载逻辑 (优先级: CLI > GA > MultiDim > Fallback) ----
def load_best_config(explicit_path=None):
    exp_dir = os.path.dirname(__file__)
    
    if explicit_path:
        if os.path.exists(explicit_path):
            with open(explicit_path, "r") as f:
                config = json.load(f)
            print(f"Loaded EXPLICIT config from {explicit_path}")
            return config, explicit_path
        else:
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
            print(f"Loaded auto-discovered config from {path}")
            return config, path
            
    # Fallback default config
    return {
        "layers": 2,
        "single_qubit_gates": ["ry"],
        "two_qubit_gate": "rzz",
        "entanglement": "brick"
    }, "fallback_default"

def run_experiment(trials=5, explicit_config_path=None):
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

    # Load config inside run_experiment to support CLI override
    ansatz_config, config_path = load_best_config(explicit_config_path)
    create_circuit, num_params = build_ansatz(ansatz_config, env.n_qubits)
    
    base_dir = os.path.dirname(__file__)
    exp_dir = prepare_experiment_dir(base_dir, "tfim_vqe")
    
    log_path = os.path.join(exp_dir, "experiment.log")
    logger = setup_logger(log_path)
    logger.info(f"--- TFIM Experiment (Atomic/Config Mode) ---")
    logger.info(f"Experiment Directory: {exp_dir}")
    logger.info(f"Config Source: {config_path}")
    logger.info(f"Config Content: {ansatz_config}")
    logger.info(f"Total Params: {num_params}")
    logger.info(f"Target Energy: {env.exact_energy:.6f}")
    
    best_results = None
    overall_best_energy = float('inf')
    
    def compute_energy_fn(params):
        c, _ = create_circuit(params)
        return env.compute_energy(c)
        
    for i in range(trials):
        logger.info(f"\n--- Trial {i+1}/{trials} ---")
        torch.manual_seed(200 + i)
        
        results = vqe_train(
            create_circuit_fn=create_circuit,
            compute_energy_fn=compute_energy_fn,
            n_qubits=env.n_qubits,
            exact_energy=env.exact_energy,
            num_params=num_params,
            max_steps=1500,
            lr=0.01,
            logger=logger
        )
        
        if results['val_energy'] < overall_best_energy:
            overall_best_energy = results['val_energy']
            best_results = results
            
    logger.info("\n=== Final Autonomous Best ===")
    print_results(best_results, logger=logger)
    
    # 记录到实验目录，并同步到实验根目录的汇总表
    log_results(exp_dir, "TFIM_ConfigMode", best_results, comment=f"Config: {ansatz_config}, source={config_path}", global_dir=base_dir)
    report_path = generate_report(
        exp_dir,
        "TFIM_Phase10_Report",
        best_results,
        create_circuit,
        ansatz_spec=ansatz_config,
        config_path=config_path,
    )
    logger.info(f"Report generated at: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TFIM VQE experiment with config.")
    parser.add_argument("--config", type=str, help="Path to explicit ansatz config JSON.")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials with different seeds.")
    args = parser.parse_args()

    run_experiment(trials=args.trials, explicit_config_path=args.config)
