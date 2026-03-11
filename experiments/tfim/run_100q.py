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
        "param_strategy": "tied", 
        "use_mps": True
    }
    
    # 3. Build the circuit factory
    create_circuit_fn, num_params = build_ansatz(config, n_qubits)
    
    def compute_energy_fn(params):
        c, _ = create_circuit_fn(params)
        return env.compute_energy(c)
        
    print(f"Starting 100-qubit TFIM VQE with MPS backend...")
    print(f"Total optimizable parameters: {num_params}")
    
    import torch
    import time

    params = torch.randn(num_params, dtype=torch.float32)

    start = time.time()
    with torch.no_grad():
        energy = compute_energy_fn(params)
    
    print("\nForward pass eval on 100 qubits completed successfully!")
    print(f"Computed Energy: {energy.item():.4f}")
    print(f"Time Taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    run_100q_mps()
