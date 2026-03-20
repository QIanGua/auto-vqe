import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.tfim.env import TFIMEnvironment
from core.circuit_factory import build_ansatz
from core.engine import vqe_train, setup_logger, print_results

def run_100q_mps():
    n_qubits = 100
    
    # 1. Initialize environment with MPS enabled
    env = TFIMEnvironment(n_qubits=n_qubits, use_mps=True)
    
    # 2. Define an ansatz configuration with parameter tying
    # Using 'tied' to compress 1000s of parameters into just a few
    config = {
        "layers": 3,
        "single_qubit_gates": ["ry", "rx"],
        "two_qubit_gate": "rzz",
        "entanglement": "linear",
        "param_strategy": "translational", 
        "use_mps": True
    }
    
    # 3. Build the circuit factory
    create_circuit_fn, num_params = build_ansatz(config, n_qubits)
    
    def compute_energy_fn(params):
        c, _ = create_circuit_fn(params)
        return env.compute_energy(c)
        
    print(f"Starting 100-qubit TFIM VQE with MPS backend...")
    print(f"Total optimizable parameters: {num_params} (Using Translational Invariance O(1))")
    
    import torch
    import time

    from core.scipy_optimizer import scipy_vqe_train

    log_path = os.path.join(os.path.dirname(__file__), "vqe_100q.log")
    logger = setup_logger(log_path)
    
    # 5. Run Scipy VQE training
    # Only 9 parameters! Gradient-free optimization will execute O(100) fast forward passes 
    start = time.time()
    results = scipy_vqe_train(
        create_circuit_fn=create_circuit_fn,
        compute_energy_fn=compute_energy_fn,
        n_qubits=n_qubits,
        num_params=num_params,
        max_steps=100, 
        logger=logger,
        method="COBYLA"
    )
    
    print("\nOptimization completed successfully!")
    print(f"Time Taken to optimize 100 qubits (100 evals): {time.time() - start:.2f} seconds")
    print_results(results)

if __name__ == "__main__":
    run_100q_mps()
