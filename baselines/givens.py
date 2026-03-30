"""
Givens-rotation based UCC ansatz (Symmetry-Preserving).

This implementation uses the efficient 30-gate sequence for double excitations
and 6-gate sequence for single excitations, as found in the `chemistry.py`
reference. This is more hardware-efficient than standard Trotterized UCC
while maintaining particle number conservation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

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


def apply_double_excitation(c: Any, qubits: Sequence[int], theta: Any) -> None:
    """Implement the efficient Givens-style double excitation gate."""
    # qubits: [s, r, q, p] where (s, r) are occupied and (q, p) are virtual
    # Following chemistry.py logic: s=qubits[0], r=qubits[1], q=qubits[2], p=qubits[3]
    s, r, q, p = qubits
    
    c.cnot(q, p)
    c.cnot(s, q)
    c.h(p)
    c.h(s)
    c.cnot(q, p)
    c.cnot(s, r)
    c.ry(r, theta=theta / 8)
    c.ry(s, theta=-theta / 8)
    c.cnot(s, p)
    c.h(p)
    c.cnot(p, r)
    c.ry(r, theta=theta / 8)
    c.ry(s, theta=-theta / 8)
    c.cnot(q, r)
    c.cnot(q, s)
    c.ry(r, theta=-theta / 8)
    c.ry(s, theta=theta / 8)
    c.cnot(p, r)
    c.h(p)
    c.cnot(s, p)
    c.ry(r, theta=-theta / 8)
    c.ry(s, theta=theta / 8)
    c.cnot(s, r)
    c.cnot(q, s)
    c.h(s)
    c.h(p)
    c.cnot(s, q)
    c.cnot(q, p)


def apply_single_excitation(c: Any, qubits: Sequence[int], theta: Any) -> None:
    """Implement the efficient Givens-style single excitation gate."""
    # qubits: [occupied, virtual]
    o, v = qubits
    c.cnot(o, v)
    c.ry(o, theta=theta / 2)
    c.cnot(v, o)
    c.ry(o, theta=-theta / 2)
    c.cnot(v, o)
    c.cnot(o, v)


def build_ansatz(env: QuantumEnvironment, config: Dict[str, Any] | None = None) -> AnsatzSpec:
    """
    Build a Givens-rotation based UCC ansatz.
    """
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

    def create_circuit(params: Any) -> tuple[Any, int]:
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
                    apply_single_excitation(c, [exc["occupied"][0], exc["virtual"][0]], theta)
                elif exc["kind"] == "double":
                    # Givens double excitation: [s, r, q, p]
                    apply_double_excitation(c, exc["occupied"] + exc["virtual"], theta)
                param_idx += 1
                
        return c, param_idx

    return AnsatzSpec(
        name="givens",
        family="uccsd",
        env_name=env.name,
        n_qubits=n_qubits,
        create_circuit=create_circuit,
        num_params=num_params,
        config=final_cfg,
        metadata={
            "description": "Hardware-efficient Givens rotation UCC baseline",
            "excitation_count": len(excitations),
            "implementation": "manual_gate_decomposition",
        },
    )


__all__ = ["build_ansatz", "apply_single_excitation", "apply_double_excitation"]
