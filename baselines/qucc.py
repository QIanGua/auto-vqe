"""
Qubit Unitary Coupled Cluster (QUCC) ansatz.

This implementation uses Qubit-UCC style excitations as defined in 
`chemistry.py` and `tensorcircuit`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from . import AnsatzSpec, QuantumEnvironment, _merge_config
from .uccsd import resolve_orbitals, enumerate_excitations


def _default_config_for_env(env: QuantumEnvironment) -> Dict[str, Any]:
    n_qubits = getattr(env, "n_qubits", 4)
    hf_qubits: List[int] = list(range(n_qubits // 2))

    return {
        "init_state": "hf",
        "hf_qubits": hf_qubits,
        "layers": 1,
        "include_singles": True,
        "include_doubles": True,
    }


def apply_qucc_single_excitation(c: Any, qubits: Sequence[int], theta: Any) -> None:
    """Implement the QUCC single excitation gate."""
    r, p = qubits
    c.rz(r, theta=np.pi / 2)
    c.ry(p, theta=-np.pi / 2)
    c.rz(p, theta=-np.pi / 2)

    c.cnot(r, p)

    c.ry(r, theta=theta / 2)
    c.rz(p, theta=-np.pi / 2)

    c.cnot(r, p)

    c.ry(r, theta=-theta / 2)
    c.h(p)

    c.cnot(r, p)


def apply_qucc_double_excitation(c: Any, qubits: Sequence[int], theta: Any) -> None:
    """Implement the QUCC double excitation gate."""
    # qubits: [s, r, q, p]
    s, r, q, p = qubits

    c.cnot(s, r)
    c.cnot(q, p)
    c.x(r)
    c.x(p)
    c.cnot(s, q)

    c.ry(s, theta=theta / 8)
    c.h(r)
    c.cnot(s, r)

    c.ry(s, theta=-theta / 8)
    c.h(p)
    c.cnot(s, p)

    c.ry(s, theta=theta / 8)
    c.cnot(s, r)

    c.ry(s, theta=-theta / 8)
    c.h(q)
    c.cnot(s, q)

    c.ry(s, theta=theta / 8)
    c.cnot(s, r)

    c.ry(s, theta=-theta / 8)
    c.cnot(s, p)

    c.ry(s, theta=theta / 8)
    c.h(p)
    c.cnot(s, r)

    c.ry(s, theta=-theta / 8)
    c.h(r)
    c.rz(q, theta=-np.pi / 2)
    c.cnot(s, q)

    c.rz(s, theta=np.pi / 2)
    c.rz(q, theta=-np.pi / 2)

    c.x(r)
    c.ry(q, theta=-np.pi / 2)
    c.x(p)

    c.cnot(s, r)
    c.cnot(q, p)


def build_ansatz(env: QuantumEnvironment, config: Dict[str, Any] | None = None) -> AnsatzSpec:
    final_cfg = _merge_config(_default_config_for_env(env), config)
    n_qubits = env.n_qubits
    hf_qubits = list(final_cfg.get("hf_qubits", []))
    layers = int(final_cfg.get("layers", 1))

    occupied, virtual = resolve_orbitals(final_cfg, n_qubits)
    excitations = enumerate_excitations(
        occupied=occupied,
        virtual=virtual,
        include_singles=bool(final_cfg.get("include_singles", True)),
        include_doubles=bool(final_cfg.get("include_doubles", True)),
    )
    
    num_params = layers * len(excitations)

    def create_circuit(params: Any) -> Tuple[Any, int]:
        import tensorcircuit as tc
        c = tc.Circuit(n_qubits)
        
        # Reference state
        for q in hf_qubits:
            c.x(q)
            
        param_idx = 0
        for _ in range(layers):
            for exc in excitations:
                theta = params[param_idx]
                if exc["kind"] == "single":
                    apply_qucc_single_excitation(c, [exc["occupied"][0], exc["virtual"][0]], theta)
                elif exc["kind"] == "double":
                    apply_qucc_double_excitation(c, exc["occupied"] + exc["virtual"], theta)
                param_idx += 1
                
        return c, param_idx

    return AnsatzSpec(
        name="qucc",
        family="uccsd",
        env_name=env.name,
        n_qubits=n_qubits,
        create_circuit=create_circuit,
        num_params=num_params,
        config=final_cfg,
        metadata={
            "description": "Qubit Unitary Coupled Cluster baseline",
            "excitation_count": len(excitations),
            "implementation": "qubit_gate_decomposition",
        },
    )


__all__ = ["build_ansatz", "apply_qucc_single_excitation", "apply_qucc_double_excitation"]
