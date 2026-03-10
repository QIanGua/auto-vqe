import sys
import os
import torch

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.engine import vqe_train, print_results, setup_logger, log_results, generate_report, tc
from experiments.lih.env import ENV

N_QUBITS = ENV.n_qubits
EXACT_ENERGY = ENV.exact_energy

LAYERS = 2

# 每层参数：
# - 对每个 qubit 的 RY/RZ: 2 * N_QUBITS = 8
# - 对线性相邻 (i, i+1) 二比特对的 RZZ: 1 * (N_QUBITS - 1) = 3
# - 总计每层 11 个参数
NUM_PARAMS_PER_LAYER = 2 * N_QUBITS + (N_QUBITS - 1)
NUM_PARAMS = LAYERS * NUM_PARAMS_PER_LAYER

def create_circuit(params):
    c = tc.Circuit(N_QUBITS)
    c.x(0); c.x(1) # HF
    idx = 0
    for _ in range(LAYERS):
        for i in range(N_QUBITS):
            c.ry(i, theta=params[idx]); idx += 1
            c.rz(i, theta=params[idx]); idx += 1
        for i in range(N_QUBITS - 1):
            j = i + 1
            c.rzz(i, j, theta=params[idx]); idx += 1
    return c, idx

def run_experiment(trials=2): # LiH 耗时较长，Trial 改为 2
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.dirname(__file__)
    log_path = os.path.join(exp_dir, f"vqe_lih_{timestamp}.log")
    
    logger = setup_logger(log_path)
    logger.info(f"--- LiH Experiment Phase 10 ---")
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
    
    log_results(exp_dir, "LiH_Phase10", best_results, comment="Robust logging & visualization verification")
    report_path = generate_report(exp_dir, "LiH_Phase10_Report", best_results, create_circuit)
    logger.info(f"Report generated at: {report_path}")

if __name__ == "__main__":
    run_experiment()
