from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core.molecular.schema import MolecularHamiltonianPoint, PauliTerm


@dataclass
class MolecularProblemSpec:
    system: str
    geometry: Sequence[Tuple[str, Tuple[float, float, float]]]
    basis: str
    multiplicity: int = 1
    charge: int = 0
    description: str = ""


@dataclass
class MolecularActiveSpaceSpec:
    occupied_indices: List[int]
    active_indices: List[int]
    mapping: str = "jordan_wigner"


def get_exact_from_qubit_op(qubit_op, n_qubits: int) -> float:
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.array([[1, 0], [0, 1]], dtype=complex)

    pauli_map = {"X": x, "Y": y, "Z": z}
    hamiltonian = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for term, coeff in qubit_op.terms.items():
        matrices = []
        term_map = {idx: pauli for idx, pauli in term}
        for qubit in range(n_qubits):
            matrices.append(pauli_map.get(term_map.get(qubit), identity))
        term_matrix = matrices[0]
        for matrix in matrices[1:]:
            term_matrix = np.kron(term_matrix, matrix)
        hamiltonian += coeff * term_matrix
    return float(np.min(np.linalg.eigvalsh(hamiltonian)).real)


def qubit_op_to_pauli_terms(qubit_op) -> List[PauliTerm]:
    paulis: List[PauliTerm] = []
    for term, coeff in qubit_op.terms.items():
        if not term:
            paulis.append(PauliTerm(float(coeff.real), []))
            continue
        ops = []
        for idx, pauli_char in term:
            op_type = pauli_char.lower()
            if op_type in ("x", "y", "z"):
                ops.append((op_type, int(idx)))
        if ops:
            paulis.append(PauliTerm(float(coeff.real), ops))
    return paulis


def build_molecular_point(
    molecule,
    qubit_hamiltonian,
    *,
    coordinates: Dict[str, float],
    metadata: Dict | None = None,
) -> MolecularHamiltonianPoint:
    from openfermion import count_qubits

    n_qubits = int(count_qubits(qubit_hamiltonian))
    return MolecularHamiltonianPoint(
        coordinates={key: float(value) for key, value in coordinates.items()},
        n_qubits=n_qubits,
        paulis=qubit_op_to_pauli_terms(qubit_hamiltonian),
        active_space_exact_energy=get_exact_from_qubit_op(qubit_hamiltonian, n_qubits),
        hf_energy=float(molecule.hf_energy) if molecule.hf_energy is not None else None,
        ccsd_energy=float(molecule.ccsd_energy) if molecule.ccsd_energy is not None else None,
        full_fci_energy=float(molecule.fci_energy) if molecule.fci_energy is not None else None,
        nuclear_repulsion=float(molecule.nuclear_repulsion) if molecule.nuclear_repulsion is not None else None,
        metadata=dict(metadata or {}),
    )
