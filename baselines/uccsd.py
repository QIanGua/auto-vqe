"""
Minimal UCCSD baseline.

This implementation is intentionally small, but it is no longer a HEA proxy.
It builds an explicit UCCSD-style ansatz from:

1. a Hartree-Fock reference state;
2. an excitation list of occupied -> virtual singles and occupied-pair ->
   virtual-pair doubles;
3. a first-order Trotterization of the Jordan-Wigner mapped anti-Hermitian
   excitation generators.

The result is still a lightweight baseline rather than a production chemistry
stack, but the circuit semantics now match the UCCSD label:

- one variational parameter per excitation per Trotter layer;
- particle-number-preserving excitation structure;
- explicit singles/doubles metadata for logging and downstream analysis.
"""

from __future__ import annotations

from itertools import combinations, product
from typing import Any, Dict, List, Sequence

import numpy as np

from . import AnsatzSpec, QuantumEnvironment, _merge_config
from core.model.schemas import OperatorPoolSpec, OperatorSpec


_PAULI_MATRICES: Dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=np.complex64),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex64),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex64),
}


def _default_config_for_env(env: QuantumEnvironment) -> Dict[str, Any]:
    n_qubits = getattr(env, "n_qubits", 4)
    hf_qubits: List[int] = list(range(n_qubits // 2))

    return {
        "init_state": "hf",
        "hf_qubits": hf_qubits,
        "layers": 1,
        "include_singles": True,
        "include_doubles": True,
        "mapping": "jordan_wigner",
        "trotter_order": 1,
    }


def _resolve_orbitals(config: Dict[str, Any], n_qubits: int) -> tuple[List[int], List[int]]:
    occupied = list(config.get("occupied_orbitals", config.get("hf_qubits", [])))
    if not occupied:
        occupied = list(range(n_qubits // 2))

    virtual = list(config.get("virtual_orbitals", [q for q in range(n_qubits) if q not in occupied]))

    if len(set(occupied)) != len(occupied):
        raise ValueError(f"Occupied orbitals must be unique, got {occupied}")
    if len(set(virtual)) != len(virtual):
        raise ValueError(f"Virtual orbitals must be unique, got {virtual}")
    if set(occupied) & set(virtual):
        raise ValueError(f"Occupied and virtual orbitals must be disjoint, got {occupied=} {virtual=}")
    if any(q < 0 or q >= n_qubits for q in occupied + virtual):
        raise ValueError(f"Orbital index out of range for {n_qubits} qubits: {occupied=} {virtual=}")

    return occupied, virtual


def _single_excitation(occupied: int, virtual: int) -> Dict[str, Any]:
    return {
        "kind": "single",
        "occupied": [occupied],
        "virtual": [virtual],
        "label": f"s_{occupied}->{virtual}",
    }


def _double_excitation(occupied_pair: Sequence[int], virtual_pair: Sequence[int]) -> Dict[str, Any]:
    occ = list(occupied_pair)
    virt = list(virtual_pair)
    return {
        "kind": "double",
        "occupied": occ,
        "virtual": virt,
        "label": f"d_{occ[0]}_{occ[1]}->{virt[0]}_{virt[1]}",
    }


def _enumerate_excitations(
    occupied: Sequence[int],
    virtual: Sequence[int],
    include_singles: bool,
    include_doubles: bool,
) -> List[Dict[str, Any]]:
    excitations: List[Dict[str, Any]] = []

    if include_singles:
        for occ, virt in product(occupied, virtual):
            excitations.append(_single_excitation(occ, virt))

    if include_doubles:
        for occ_pair in combinations(occupied, 2):
            for virt_pair in combinations(virtual, 2):
                excitations.append(_double_excitation(occ_pair, virt_pair))

    return excitations


def _build_fermion_generator(excitation: Dict[str, Any]) -> Any:
    from openfermion import FermionOperator

    kind = excitation["kind"]
    occ = excitation["occupied"]
    virt = excitation["virtual"]

    if kind == "single":
        forward = FermionOperator(((virt[0], 1), (occ[0], 0)), 1.0)
        backward = FermionOperator(((occ[0], 1), (virt[0], 0)), 1.0)
    elif kind == "double":
        forward = FermionOperator(
            ((virt[0], 1), (virt[1], 1), (occ[1], 0), (occ[0], 0)),
            1.0,
        )
        backward = FermionOperator(
            ((occ[0], 1), (occ[1], 1), (virt[1], 0), (virt[0], 0)),
            1.0,
        )
    else:
        raise ValueError(f"Unknown excitation kind: {kind}")

    return forward - backward


def _pauli_tensor(paulis: Sequence[str]) -> np.ndarray:
    matrix = np.array([[1.0 + 0.0j]], dtype=np.complex64)
    for pauli in paulis:
        matrix = np.kron(matrix, _PAULI_MATRICES[pauli])
    return matrix.reshape([2] * (2 * len(paulis)))


def _jw_terms_from_excitation(excitation: Dict[str, Any]) -> List[Dict[str, Any]]:
    from openfermion import jordan_wigner

    qubit_operator = jordan_wigner(_build_fermion_generator(excitation))
    jw_terms: List[Dict[str, Any]] = []

    for raw_term, coeff in qubit_operator.terms.items():
        coeff = complex(coeff)
        if abs(coeff) < 1e-12:
            continue
        if abs(coeff.real) > 1e-9:
            raise ValueError(
                f"Expected anti-Hermitian JW term with purely imaginary coefficient, got {coeff} for {raw_term}"
            )

        support = sorted(qubit for qubit, _ in raw_term)
        by_qubit = {qubit: pauli.upper() for qubit, pauli in raw_term}
        paulis = [by_qubit.get(qubit, "I") for qubit in support]

        jw_terms.append(
            {
                "support_qubits": support,
                "paulis": paulis,
                "coeff_imag": float(coeff.imag),
            }
        )

    if not jw_terms:
        raise ValueError(f"Excitation {excitation['label']} produced no JW terms")

    return jw_terms


def build_excitation_records(
    env: QuantumEnvironment,
    config: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    final_cfg = _merge_config(_default_config_for_env(env), config)
    occupied, virtual = _resolve_orbitals(final_cfg, env.n_qubits)
    excitations = _enumerate_excitations(
        occupied=occupied,
        virtual=virtual,
        include_singles=bool(final_cfg.get("include_singles", True)),
        include_doubles=bool(final_cfg.get("include_doubles", True)),
    )
    return [{**excitation, "jw_terms": _jw_terms_from_excitation(excitation)} for excitation in excitations]


def build_excitation_operator_pool(
    env: QuantumEnvironment,
    config: Dict[str, Any] | None = None,
) -> OperatorPoolSpec:
    excitation_records = build_excitation_records(env, config)
    operators: List[OperatorSpec] = []
    for excitation in excitation_records:
        support = sorted(
            {
                int(qubit)
                for term in excitation["jw_terms"]
                for qubit in term["support_qubits"]
            }
        )
        operators.append(
            OperatorSpec(
                name=excitation["label"],
                family="excitation",
                support_qubits=support,
                generator=excitation["label"],
                symmetry_tags=["particle_number_preserving"],
                metadata={
                    "kind": excitation["kind"],
                    "occupied": list(excitation["occupied"]),
                    "virtual": list(excitation["virtual"]),
                    "trotter_terms": excitation["jw_terms"],
                },
            )
        )

    return OperatorPoolSpec(
        name="fermionic_excitation_pool",
        operators=operators,
        metadata={
            "implementation": "jw_first_order_trotter",
            "source": "minimal_uccsd_excitation_pool",
        },
    )


def _build_minimal_uccsd_circuit(
    n_qubits: int,
    hf_qubits: Sequence[int],
    excitations: Sequence[Dict[str, Any]],
    layers: int,
) -> tuple[Any, int]:
    compiled_excitations: List[List[Dict[str, Any]]] = []
    for excitation in excitations:
        compiled_terms = []
        for term in excitation["jw_terms"]:
            compiled_terms.append(
                {
                    "support_qubits": term["support_qubits"],
                    "coeff_imag": term["coeff_imag"],
                    "unitary": _pauli_tensor(term["paulis"]),
                }
            )
        compiled_excitations.append(compiled_terms)

    num_params = layers * len(excitations)

    def create_circuit(params: Any) -> tuple[Any, int]:
        import tensorcircuit as tc

        c = tc.Circuit(n_qubits)
        for qubit in hf_qubits:
            c.x(qubit)

        idx = 0
        for _ in range(layers):
            for compiled_terms in compiled_excitations:
                theta = params[idx]
                for term in compiled_terms:
                    # exp1 applies exp(-i * phi * P); JW coefficients are i * a,
                    # so phi = -a * theta matches exp(theta * (i a P)).
                    c.exp1(
                        *term["support_qubits"],
                        unitary=term["unitary"],
                        theta=(-term["coeff_imag"]) * theta,
                    )
                idx += 1

        return c, idx

    return create_circuit, num_params


def build_ansatz(env: QuantumEnvironment, config: Dict[str, Any] | None = None) -> AnsatzSpec:
    """
    Build a minimal, explicit UCCSD ansatz.

    The implementation is intentionally conservative:
    - Hartree-Fock reference state from `hf_qubits`;
    - singles and doubles generated from occupied/virtual orbital partitions;
    - first-order Trotterization of Jordan-Wigner mapped generators.
    """
    final_cfg = _merge_config(_default_config_for_env(env), config)
    n_qubits = env.n_qubits
    hf_qubits = list(final_cfg.get("hf_qubits", []))
    layers = int(final_cfg.get("layers", 1))

    if final_cfg.get("mapping") != "jordan_wigner":
        raise ValueError("Minimal UCCSD baseline currently supports only jordan_wigner mapping")
    if int(final_cfg.get("trotter_order", 1)) != 1:
        raise ValueError("Minimal UCCSD baseline currently supports only first-order Trotterization")
    if layers <= 0:
        raise ValueError(f"layers must be positive, got {layers}")

    occupied, virtual = _resolve_orbitals(final_cfg, n_qubits)
    excitation_records = build_excitation_records(env, final_cfg)
    if not excitation_records:
        raise ValueError("Minimal UCCSD baseline generated zero excitations")

    create_circuit, num_params = _build_minimal_uccsd_circuit(
        n_qubits=n_qubits,
        hf_qubits=hf_qubits,
        excitations=excitation_records,
        layers=layers,
    )

    return AnsatzSpec(
        name="uccsd",
        family="uccsd",
        env_name=env.name,
        n_qubits=n_qubits,
        create_circuit=create_circuit,
        num_params=num_params,
        config={
            **final_cfg,
            "occupied_orbitals": occupied,
            "virtual_orbitals": virtual,
        },
        metadata={
            "description": "Minimal explicit UCCSD baseline with JW-mapped singles/doubles",
            "implementation": "jw_first_order_trotter",
            "excitation_count": len(excitation_records),
            "singles_count": sum(1 for exc in excitation_records if exc["kind"] == "single"),
            "doubles_count": sum(1 for exc in excitation_records if exc["kind"] == "double"),
            "excitations": excitation_records,
        },
    )


__all__ = ["build_ansatz", "build_excitation_records", "build_excitation_operator_pool"]
