"""
Qubit-ADAPT-VQE initialization helpers.

This module provides the reference state and qubit-operator pool used by the
gradient-based Qubit-ADAPT-VQE search.
"""

from __future__ import annotations

from typing import Any, Dict

from core.generator.adapt import build_qubit_adapt_pool

from . import AnsatzSpec, QuantumEnvironment, _merge_config


def _default_config_for_env(env: QuantumEnvironment) -> Dict[str, Any]:
    n_qubits = getattr(env, "n_qubits", 4)
    return {
        "init_state": "hf" if getattr(env, "name", "").startswith("LiH") else "zero",
        "hf_qubits": list(range(n_qubits // 2)),
        "max_body": 2,
        "include_single_qubit": True,
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
        name="qubit_adapt_init",
        family="qubit_adapt",
        env_name=env.name,
        n_qubits=env.n_qubits,
        create_circuit=create_circuit,
        num_params=0,
        config={
            "init_state": final_cfg["init_state"],
            "hf_qubits": final_cfg.get("hf_qubits", []),
        },
        metadata={
            "description": "Initial reference state for Qubit-ADAPT-VQE growth",
            "research_status": "search_initialized",
        },
    )


def build_operator_pool(env: QuantumEnvironment, config: Dict[str, Any] | None = None):
    final_cfg = _merge_config(_default_config_for_env(env), config)
    return build_qubit_adapt_pool(
        env.n_qubits,
        max_body=int(final_cfg.get("max_body", 2)),
        include_single_qubit=bool(final_cfg.get("include_single_qubit", True)),
    )
__all__ = ["build_ansatz", "build_operator_pool"]
