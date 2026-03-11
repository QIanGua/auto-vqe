import os
import sys
import torch

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.engine import vqe_train, print_results, setup_logger, log_results, generate_report, tc
from core.circuit_factory import build_ansatz
from experiments.tfim.env import ENV

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
        "layers": 2,
        "single_qubit_gates": ["ry"],
        "two_qubit_gate": "rzz",
        "entanglement": "brick"
    }, "fallback_default"

ANSATZ_CONFIG, CONFIG_PATH = load_best_config()

create_circuit, NUM_PARAMS = build_ansatz(ANSATZ_CONFIG, N_QUBITS)

def run_experiment(trials=5):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.dirname(__file__)
    log_path = os.path.join(exp_dir, f"vqe_tfim_{timestamp}.log")
    
    logger = setup_logger(log_path)
    logger.info(f"--- TFIM Experiment (Atomic/Config Mode) ---")
    logger.info(f"Config Source: {CONFIG_PATH}")
    logger.info(f"Config Content: {ANSATZ_CONFIG}")
    logger.info(f"Total Params: {NUM_PARAMS}")
    logger.info(f"Target Energy: {EXACT_ENERGY:.6f}")
    
    best_results = None
    overall_best_energy = float('inf')
    
    def compute_energy_fn(params):
        c, _ = create_circuit(params)
        return ENV.compute_energy(c)
        
    for i in range(trials):
        logger.info(f"\n--- Trial {i+1}/{trials} ---")
        torch.manual_seed(200 + i)
        
        results = vqe_train(
            create_circuit_fn=create_circuit,
            compute_energy_fn=compute_energy_fn,
            n_qubits=N_QUBITS,
            exact_energy=EXACT_ENERGY,
            num_params=NUM_PARAMS,
            max_steps=1500,
            lr=0.01,
            logger=logger
        )
        
        if results['val_energy'] < overall_best_energy:
            overall_best_energy = results['val_energy']
            best_results = results
            
    logger.info("\n=== Final Autonomous Best ===")
    print_results(best_results, logger=logger)
    
    log_results(exp_dir, "TFIM_ConfigMode", best_results, comment=f"Config: {ANSATZ_CONFIG}, source={CONFIG_PATH}")
    report_path = generate_report(
        exp_dir,
        "TFIM_Phase10_Report",
        best_results,
        create_circuit,
        ansatz_spec=ANSATZ_CONFIG,
        config_path=CONFIG_PATH,
    )
    logger.info(f"Report generated at: {report_path}")

if __name__ == "__main__":
    run_experiment()
