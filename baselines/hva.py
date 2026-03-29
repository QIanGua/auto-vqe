"""
Hamiltonian Variational Ansatz (HVA) / QAOA-style baseline.

This implements a simple TFIM-inspired structure:

  - Alternate e^{-i γ_l H_ZZ} layers with e^{-i β_l H_X} layers
  - Use one (γ_l, β_l) pair per layer l

For an environment with `n_qubits` qubits, the circuit for depth `p` is:

  for l in range(p):
      ZZ layer over nearest neighbours
      X  layer on every qubit

The exact physical calibration of γ/β is left for optimization.
"""

from __future__ import annotations

from typing import Any, Dict

import tensorcircuit as tc

from . import AnsatzSpec, QuantumEnvironment, _merge_config


def _default_config(env: QuantumEnvironment) -> Dict[str, Any]:
    # `layers` plays the role of the QAOA depth p.
    boundary = getattr(env, "boundary", "ring")
    return {
        "layers": 2,
        "use_ring": boundary == "ring",
    }


def build_ansatz(env: QuantumEnvironment, config: Dict[str, Any] | None = None) -> AnsatzSpec:
    """
    Build a Hamiltonian-Variational (HVA) ansatz.

    Parameters
    ----------
    env : QuantumEnvironment
        Used only for `n_qubits` and naming.
    config : dict | None
        Optional overrides. Recognized keys:
          - `layers`  (int): HVA depth p (default: 2)
          - `use_ring`(bool): include (N-1, 0) ZZ coupling (default: True)
    """
    cfg = _merge_config(_default_config(env), config)
    p = int(cfg.get("layers", 2))
    use_ring = bool(cfg.get("use_ring", True))

    n_qubits = env.n_qubits

    # We use 2 parameters per layer: one γ_l for ZZ terms, one β_l for X terms.
    num_params = 2 * p

    def create_circuit(params: Any):
        c = tc.Circuit(n_qubits)
        idx = 0

        for _layer in range(p):
            gamma = params[idx]; idx += 1
            beta = params[idx]; idx += 1

            # ZZ part: nearest neighbours (+ optional ring closure)
            for i in range(n_qubits - 1):
                c.rzz(i, i + 1, theta=2.0 * gamma)
            if use_ring and n_qubits > 2:
                c.rzz(n_qubits - 1, 0, theta=2.0 * gamma)

            # X part: transverse field
            for q in range(n_qubits):
                c.rx(q, theta=2.0 * beta)

        return c, idx

    return AnsatzSpec(
        name="hva",
        family="hva",
        env_name=env.name,
        n_qubits=n_qubits,
        create_circuit=create_circuit,
        num_params=num_params,
        config=cfg,
        metadata={
            "description": "Hamiltonian-Variational Ansatz / QAOA-style architecture",
            "boundary": "ring" if use_ring else "open",
        },
    )


__all__ = ["build_ansatz"]
