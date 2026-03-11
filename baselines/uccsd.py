"""
UCCSD-style baseline.

This is a *toy*, hardware-efficient proxy for chemistry-inspired UCCSD:

  - start from a Hartree–Fock (HF) reference state;
  - apply several layers of single-qubit rotations (RX/RY/RZ);
  - entangle qubits using a rich two-qubit gate `rxx_ryy_rzz` with a chosen
    entanglement topology.

It is intentionally implemented via `core.circuit_factory.build_ansatz` so that
it shares the same config schema as other HEA-style circuits while being
tagged distinctly as "uccsd" in logs and analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.circuit_factory import build_ansatz as cf_build_ansatz

from . import AnsatzSpec, QuantumEnvironment, _merge_config


def _default_config_for_env(env: QuantumEnvironment) -> Dict[str, Any]:
    n_qubits = getattr(env, "n_qubits", 4)

    # Simple HF occupation pattern: fill the first half of qubits.
    hf_qubits: List[int] = list(range(n_qubits // 2))

    return {
        "init_state": "hf",
        "hf_qubits": hf_qubits,
        "layers": 2,
        "single_qubit_gates": ["rx", "ry", "rz"],
        "two_qubit_gate": "rxx_ryy_rzz",
        "entanglement": "full",
        "param_strategy": "independent",
    }


def build_ansatz(env: QuantumEnvironment, config: Dict[str, Any] | None = None) -> AnsatzSpec:
    """
    Build a UCCSD-style ansatz using the generic circuit factory.

    Parameters
    ----------
    env : QuantumEnvironment
        Typically a chemistry system such as LiH.
    config : dict | None
        Optional overrides for the internal configuration. Any key here will
        replace the default value used to mimic a UCCSD-like structure.
    """
    base_cfg = _default_config_for_env(env)
    final_cfg = _merge_config(base_cfg, config)

    create_circuit, num_params = cf_build_ansatz(final_cfg, env.n_qubits)

    return AnsatzSpec(
        name="uccsd",
        family="uccsd",
        env_name=env.name,
        n_qubits=env.n_qubits,
        create_circuit=create_circuit,
        num_params=num_params,
        config=final_cfg,
        metadata={
            "description": "Toy UCCSD-like ansatz built on top of circuit_factory",
        },
    )


__all__ = ["build_ansatz"]

