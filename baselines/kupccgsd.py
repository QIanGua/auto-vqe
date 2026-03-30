"""
k-UpCCGSD (k-layer Unitary Paired Coupled Cluster Generalized Single Double) ansatz.

This implementation uses generalized single excitations and paired double 
excitations, which are more expressive than standard UCCSD for strongly 
correlated systems.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from . import AnsatzSpec, QuantumEnvironment, _merge_config
from .uccsd import resolve_orbitals, _pauli_tensor


def _default_config_for_env(env: QuantumEnvironment) -> Dict[str, Any]:
    n_qubits = getattr(env, "n_qubits", 4)
    hf_qubits: List[int] = list(range(n_qubits // 2))

    return {
        "init_state": "hf",
        "hf_qubits": hf_qubits,
        "layers": 1,
        "delta_sz": 0,
    }


def get_generalized_singles(n_qubits: int, delta_sz: int = 0) -> List[Dict[str, Any]]:
    """Generate all generalized single excitations [r, p] with p != r."""
    sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(n_qubits)])
    singles = []
    for r in range(n_qubits):
        for p in range(n_qubits):
            if p != r and (sz[p] - sz[r]) == delta_sz:
                singles.append({
                    "kind": "generalized_single",
                    "qubits": sorted([r, p]),
                    "label": f"gs_{r}->{p}",
                })
    return singles


def get_paired_doubles(n_qubits: int) -> List[Dict[str, Any]]:
    """Generate paired double excitations [(r, r+1), (p, p+1)]."""
    doubles = []
    for r in range(0, n_qubits - 1, 2):
        for p in range(0, n_qubits - 1, 2):
            if p != r:
                doubles.append({
                    "kind": "paired_double",
                    "qubits": [r, r + 1, p, p + 1],
                    "label": f"pd_{r},{r+1}->{p},{p+1}",
                })
    return doubles


def _get_jw_excitations(n_qubits: int, delta_sz: int = 0) -> List[Dict[str, Any]]:
    # This is a bit complex as we need JW mapping for generic generalized excitations
    # For UpCCGSD, we often use the standard Pauli expansions.
    # To keep it simple and consistent with chemistry.py, we'll use a simplified 
    # model or reuse the openfermion logic if possible.
    # Here we'll manually define the JW terms for simplicity in this baseline.
    from openfermion import FermionOperator, jordan_wigner

    singles = get_generalized_singles(n_qubits, delta_sz)
    doubles = get_paired_doubles(n_qubits)
    
    all_excitations = []
    
    for s in singles:
        r, p = s["qubits"]
        op = FermionOperator(((p, 1), (r, 0)), 1.0) - FermionOperator(((r, 1), (p, 0)), 1.0)
        jw = jordan_wigner(op)
        terms = []
        for term_ops, coeff in jw.terms.items():
            if abs(coeff) < 1e-12: continue
            support = sorted([idx for idx, _ in term_ops])
            paulis = [dict(term_ops).get(i, "I") for i in support]
            terms.append({"support": support, "paulis": paulis, "coeff_imag": float(coeff.imag)})
        all_excitations.append({**s, "jw_terms": terms})
        
    for d in doubles:
        r1, r2, p1, p2 = d["qubits"]
        # Paired double: p1^ p2^ r2 r1
        op = FermionOperator(((p1, 1), (p2, 1), (r2, 0), (r1, 0)), 1.0) - \
             FermionOperator(((r1, 1), (r2, 1), (p2, 0), (p1, 0)), 1.0)
        jw = jordan_wigner(op)
        terms = []
        for term_ops, coeff in jw.terms.items():
            if abs(coeff) < 1e-12: continue
            support = sorted([idx for idx, _ in term_ops])
            paulis = [dict(term_ops).get(i, "I") for i in support]
            terms.append({"support": support, "paulis": paulis, "coeff_imag": float(coeff.imag)})
        all_excitations.append({**d, "jw_terms": terms})
        
    return all_excitations


def build_ansatz(env: QuantumEnvironment, config: Dict[str, Any] | None = None) -> AnsatzSpec:
    final_cfg = _merge_config(_default_config_for_env(env), config)
    n_qubits = env.n_qubits
    layers = int(final_cfg.get("layers", 1))
    delta_sz = int(final_cfg.get("delta_sz", 0))

    excitations = _get_jw_excitations(n_qubits, delta_sz)
    num_params = layers * len(excitations)

    # Pre-compile Pauli tensors for speed
    for exc in excitations:
        for term in exc["jw_terms"]:
            term["unitary"] = _pauli_tensor(term["paulis"])

    def create_circuit(params: Any) -> Tuple[Any, int]:
        import tensorcircuit as tc
        c = tc.Circuit(n_qubits)
        
        # HF Init if requested
        if final_cfg.get("init_state") == "hf":
            for q in final_cfg.get("hf_qubits", []):
                c.x(q)
                
        idx = 0
        for _ in range(layers):
            for exc in excitations:
                theta = params[idx]
                for term in exc["jw_terms"]:
                    c.exp1(
                        *term["support"],
                        unitary=term["unitary"],
                        theta=(-term["coeff_imag"]) * theta,
                    )
                idx += 1
        return c, idx

    return AnsatzSpec(
        name="kupccgsd",
        family="uccsd",
        env_name=env.name,
        n_qubits=n_qubits,
        create_circuit=create_circuit,
        num_params=num_params,
        config=final_cfg,
        metadata={
            "description": f"{layers}-layer UpCCGSD",
            "excitation_count": len(excitations),
            "implementation": "jw_trotter",
        },
    )


__all__ = ["build_ansatz"]
