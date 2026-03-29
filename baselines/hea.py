"""
Hardware-Efficient Ansatz (HEA) baseline.

This wraps the generic `core.representation.compiler.build_ansatz` helper and chooses
reasonable defaults per environment, while still allowing callers to override
any field via the `config` dict.

Interface
---------
    from baselines.hea import build_ansatz
    spec = build_ansatz(env, config)

where `spec` is an `AnsatzSpec` defined in `baselines.__init__`.
"""

from __future__ import annotations

from typing import Any, Dict

from core.representation.compiler import build_ansatz as cf_build_ansatz

from . import AnsatzSpec, QuantumEnvironment, _merge_config


def _default_config_for_env(env: QuantumEnvironment) -> Dict[str, Any]:
    """
    Provide conservative HEA defaults, loosely aligned with the hand-written
    baselines already used in TFIM / LiH experiments.
    """
    name = getattr(env, "name", "")

    # TFIM: 4 qubits, shallow RZZ + RY brick-wall structure (Phase-2 style)
    if name.startswith("TFIM"):
        return {
            "layers": 2,
            "single_qubit_gates": ["ry"],
            "two_qubit_gate": "rzz",
            "entanglement": "brick",
            "init_state": "zero",
            "param_strategy": "independent",
        }

    # LiH: use HF initialization with slightly richer single-qubit gate set
    if name.startswith("LiH"):
        n_qubits = getattr(env, "n_qubits", 4)
        # Simple HF occupation: first half of qubits set to |1>
        hf_qubits = list(range(n_qubits // 2))
        return {
            "init_state": "hf",
            "hf_qubits": hf_qubits,
            "layers": 2,
            "single_qubit_gates": ["ry", "rz"],
            "two_qubit_gate": "rzz",
            "entanglement": "linear",
            "param_strategy": "independent",
        }

    # Generic fallback HEA on unknown environments
    return {
        "layers": 2,
        "single_qubit_gates": ["ry"],
        "two_qubit_gate": "cnot",
        "entanglement": "linear",
        "init_state": "zero",
        "param_strategy": "independent",
    }


def build_ansatz(env: QuantumEnvironment, config: Dict[str, Any] | None = None) -> AnsatzSpec:
    """
    Build a hardware-efficient ansatz for a given environment.

    Parameters
    ----------
    env : QuantumEnvironment
        Provides `n_qubits` and a human-readable `name`.
    config : dict | None
        Optional overrides for the internal HEA configuration. Any keys in
        `config` replace the defaults from `_default_config_for_env`.
    """
    base_cfg = _default_config_for_env(env)
    final_cfg = _merge_config(base_cfg, config)

    create_circuit, num_params = cf_build_ansatz(final_cfg, env.n_qubits)

    return AnsatzSpec(
        name="hea",
        family="hea",
        env_name=env.name,
        n_qubits=env.n_qubits,
        create_circuit=create_circuit,
        num_params=num_params,
        config=final_cfg,
        metadata={
            "description": "Hardware-efficient ansatz built via core.representation.compiler",
        },
    )


__all__ = ["build_ansatz"]
