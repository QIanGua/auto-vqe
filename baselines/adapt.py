"""
ADAPT-VQE initialization helpers.

This module no longer pretends ADAPT-VQE is a static UCCSD baseline. Instead it
provides:

- an initial reference ansatz suitable for adaptive growth;
- a fermionic excitation pool for gradient-based operator selection.
"""

from __future__ import annotations

from typing import Any, Dict

from core.generator.adapt import build_fermionic_adapt_pool

from . import AnsatzSpec, QuantumEnvironment, _merge_config


def _default_config_for_env(env: QuantumEnvironment) -> Dict[str, Any]:
    n_qubits = getattr(env, "n_qubits", 4)
    hf_qubits = list(range(n_qubits // 2))
    return {
        "init_state": "hf" if getattr(env, "name", "").startswith("LiH") else "zero",
        "hf_qubits": hf_qubits,
        "occupied_orbitals": hf_qubits,
        "virtual_orbitals": [q for q in range(n_qubits) if q not in hf_qubits],
        "include_singles": True,
        "include_doubles": True,
    }


def build_ansatz(env: QuantumEnvironment, config: Dict[str, Any] | None = None) -> AnsatzSpec:
    final_cfg = _merge_config(_default_config_for_env(env), config)
    def create_circuit(_params: Any):
        import tensorcircuit as tc

        c = tc.Circuit(env.n_qubits)
        if final_cfg["init_state"] == "hf":
            for q in final_cfg.get("hf_qubits", []):
                c.x(q)
        return c, 0

    return AnsatzSpec(
        name="adapt_init",
        family="adapt",
        env_name=env.name,
        n_qubits=env.n_qubits,
        create_circuit=create_circuit,
        num_params=0,
        config={
            "init_state": final_cfg["init_state"],
            "hf_qubits": final_cfg.get("hf_qubits", []),
        },
        metadata={
            "description": "Initial reference state for ADAPT-VQE growth",
            "research_status": "search_initialized",
        },
    )


def build_operator_pool(env: QuantumEnvironment, config: Dict[str, Any] | None = None):
    final_cfg = _merge_config(_default_config_for_env(env), config)
    return build_fermionic_adapt_pool(env, final_cfg)
__all__ = ["build_ansatz", "build_operator_pool"]
