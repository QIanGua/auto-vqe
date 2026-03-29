import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from core.model.schemas import AnsatzSpec, OptimizerSpec


def optimize_parameters(
    env: Any,
    ansatz: Optional[AnsatzSpec] = None,
    optimizer_spec: Optional[OptimizerSpec] = None,
    init_params: Optional[np.ndarray] = None,
    logger=None,
    seed: Optional[int] = None,
    create_circuit_fn: Optional[Callable] = None,
    compute_energy_fn: Optional[Callable] = None,
    num_params: Optional[int] = None,
) -> Dict[str, Any]:
    if optimizer_spec is None:
        optimizer_spec = OptimizerSpec()

    if create_circuit_fn is None:
        if ansatz is None:
            raise ValueError("Either ansatz or create_circuit_fn must be provided.")
        from core.representation.compiler import build_circuit_from_ansatz

        create_circuit_fn, num_params = build_circuit_from_ansatz(ansatz)

    if compute_energy_fn is None:
        def compute_energy_fn(params):
            c, _ = create_circuit_fn(params)
            return env.compute_energy(c)

    if num_params is None:
        num_params = 0

    if init_params is not None:
        params_tensor = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)
    else:
        if seed is not None:
            torch.manual_seed(seed)
        params_tensor = torch.randn(num_params, requires_grad=True)

    max_steps = optimizer_spec.max_steps
    lr = optimizer_spec.lr
    early_stop_window = optimizer_spec.early_stop_window
    early_stop_threshold = optimizer_spec.early_stop_threshold
    grad_clip_norm = optimizer_spec.grad_clip_norm

    optimizer = torch.optim.Adam([params_tensor], lr=lr)
    sched_spec = optimizer_spec.scheduler
    if sched_spec and sched_spec.type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_spec.mode,
            patience=sched_spec.patience,
            factor=sched_spec.factor,
            min_lr=sched_spec.min_lr,
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    start_time = time.time()
    energy_history = []
    actual_steps = max_steps

    for i in range(max_steps):
        optimizer.zero_grad()
        energy = compute_energy_fn(params_tensor)
        energy.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([params_tensor], max_norm=grad_clip_norm)
        optimizer.step()

        e_val = energy.item()
        energy_history.append(e_val)
        scheduler.step(e_val)

        if i % 100 == 0 and logger:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Step {i}, Energy: {e_val:.6f}, LR: {current_lr:.2e}")

        if len(energy_history) >= early_stop_window:
            recent = energy_history[-early_stop_window:]
            if max(recent) - min(recent) < early_stop_threshold:
                actual_steps = i + 1
                break

    end_time = time.time()
    final_energy = compute_energy_fn(params_tensor).item()
    return {
        "val_energy": final_energy,
        "exact_energy": env.exact_energy if hasattr(env, "exact_energy") else None,
        "energy_error": abs(final_energy - env.exact_energy) if hasattr(env, "exact_energy") else None,
        "num_params": num_params,
        "runtime_sec": end_time - start_time,
        "actual_steps": actual_steps,
        "final_params": params_tensor.detach().numpy(),
        "energy_history": energy_history,
    }


def vqe_train(
    create_circuit_fn,
    compute_energy_fn,
    n_qubits,
    exact_energy=None,
    num_params=0,
    max_steps=1000,
    lr=0.01,
    logger=None,
    seed=None,
    early_stop_window=50,
    early_stop_threshold=1e-8,
    grad_clip_norm=1.0,
    optimizer_spec_obj: Optional[OptimizerSpec] = None,
):
    if optimizer_spec_obj is None:
        optimizer_spec_obj = OptimizerSpec(
            lr=lr,
            max_steps=max_steps,
            early_stop_window=early_stop_window,
            early_stop_threshold=early_stop_threshold,
            grad_clip_norm=grad_clip_norm,
        )

    if seed is not None:
        torch.manual_seed(seed)

    params = torch.randn(num_params, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=optimizer_spec_obj.lr)

    start_time = time.time()
    energy_history = []
    actual_steps = optimizer_spec_obj.max_steps

    for i in range(optimizer_spec_obj.max_steps):
        optimizer.zero_grad()
        energy = compute_energy_fn(params)
        energy.backward()
        if optimizer_spec_obj.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([params], max_norm=optimizer_spec_obj.grad_clip_norm)
        optimizer.step()
        e_val = energy.item()
        energy_history.append(e_val)
        if i % 100 == 0 and logger:
            logger.info(f"Step {i}, Energy: {e_val:.6f}")
        if len(energy_history) >= optimizer_spec_obj.early_stop_window:
            recent = energy_history[-optimizer_spec_obj.early_stop_window:]
            if max(recent) - min(recent) < optimizer_spec_obj.early_stop_threshold:
                actual_steps = i + 1
                break

    final_energy = compute_energy_fn(params).item()
    return {
        "val_energy": final_energy,
        "exact_energy": exact_energy,
        "energy_error": abs(final_energy - exact_energy) if exact_energy is not None else None,
        "num_params": num_params,
        "training_seconds": time.time() - start_time,
        "actual_steps": actual_steps,
        "final_params": params.detach(),
        "energy_history": energy_history,
    }
