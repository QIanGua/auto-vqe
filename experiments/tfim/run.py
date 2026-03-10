import os
import sys

import torch

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.engine import vqe_train, print_results, setup_logger, log_results, generate_report, tc
from experiments.tfim.env import ENV

N_QUBITS = ENV.n_qubits
EXACT_ENERGY = ENV.exact_energy

LAYERS = 2
NUM_PARAMS = 12 # 2 layers * (4 RY + 2 RZZ) = 12 parameters

def create_circuit(params):
    c = tc.Circuit(N_QUBITS)
    idx = 0
    # Layer 1: Global RY + Brick RZZ (0,1) & (2,3)
    for i in range(N_QUBITS):
        c.ry(i, theta=params[idx]); idx += 1
    c.rzz(0, 1, theta=params[idx]); idx += 1
    c.rzz(2, 3, theta=params[idx]); idx += 1
    
    # Layer 2: Global RY + Brick RZZ (1,2) & (3,0)
    for i in range(N_QUBITS):
        c.ry(i, theta=params[idx]); idx += 1
    c.rzz(1, 2, theta=params[idx]); idx += 1
    c.rzz(3, 0, theta=params[idx]); idx += 1
    
    return c, idx

def run_experiment(trials=5):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.dirname(__file__)
    log_path = os.path.join(exp_dir, f"vqe_tfim_{timestamp}.log")
    
    logger = setup_logger(log_path)
    logger.info(f"--- TFIM Experiment Phase 2: RZZ Ansatz ---")
    logger.info(f"Layers: {LAYERS}, Total Params: {NUM_PARAMS}")
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
    
    log_results(exp_dir, "TFIM_Phase6_HybridSymmetry", best_results, comment=f"Hybrid Symmetry (2T+2U), layers={LAYERS}, params={NUM_PARAMS}")
    report_path = generate_report(exp_dir, "TFIM_Phase10_Report", best_results, create_circuit)
    logger.info(f"Report generated at: {report_path}")

if __name__ == "__main__":
    run_experiment()
