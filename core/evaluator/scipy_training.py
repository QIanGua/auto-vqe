import time

import numpy as np
import scipy.optimize
import torch


def scipy_vqe_train(
    create_circuit_fn,
    compute_energy_fn,
    n_qubits,
    num_params,
    max_steps=500,
    logger=None,
    method="COBYLA",
):
    """
    Gradient-free training using Scipy for scenarios where AD (AutoGrad) is too
    slow (e.g. 100-q MPS SVD contractions).
    """
    np.random.seed(42)
    init_params = np.random.randn(num_params).astype(np.float32)

    start_time = time.time()
    energy_history = []

    def cost_fn(x):
        params_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            energy = compute_energy_fn(params_tensor).item()

        energy_history.append(energy)
        step = len(energy_history)

        if step % 10 == 0 and logger:
            logger.info(f"Step {step}, Energy: {energy:.6f}")

        return energy

    if logger:
        logger.info(f"Starting Scipy Optimization with method {method}...")

    res = scipy.optimize.minimize(
        cost_fn,
        init_params,
        method=method,
        options={"maxiter": max_steps},
    )

    end_time = time.time()

    return {
        "val_energy": res.fun,
        "energy_error": None,
        "num_params": num_params,
        "training_seconds": end_time - start_time,
        "actual_steps": res.nfev,
        "final_params": torch.tensor(res.x, dtype=torch.float32),
        "energy_history": energy_history,
        "success": res.success,
        "message": res.message,
    }
