from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from core.molecular.pauli import get_exact_from_paulis
from core.molecular.schema import MolecularHamiltonianData, MolecularHamiltonianPoint, PauliTerm


def _point_from_dict(data: Dict[str, Any]) -> MolecularHamiltonianPoint:
    coordinates = dict(data.get("coordinates", {}))
    if "R" in data and "R" not in coordinates:
        coordinates["R"] = float(data["R"])
    return MolecularHamiltonianPoint(
        coordinates={key: float(value) for key, value in coordinates.items()},
        n_qubits=int(data.get("n_qubits", 4)),
        paulis=[
            PauliTerm(
                coeff=float(term["coeff"]),
                ops=[(str(op_type), int(idx)) for op_type, idx in term.get("ops", [])],
            )
            for term in data.get("paulis", [])
        ],
        active_space_exact_energy=float(
            data["active_space_exact_energy"]
            if data.get("active_space_exact_energy") is not None
            else data.get("exact_energy")
        ),
        hf_energy=float(data["hf_energy"]) if data.get("hf_energy") is not None else None,
        ccsd_energy=float(data["ccsd_energy"]) if data.get("ccsd_energy") is not None else None,
        full_fci_energy=float(data["full_fci_energy"]) if data.get("full_fci_energy") is not None else (
            float(data["exact_energy"]) if data.get("exact_energy") is not None else None
        ),
        nuclear_repulsion=float(data["nuclear_repulsion"]) if data.get("nuclear_repulsion") is not None else None,
        metadata=dict(data.get("metadata", {})),
    )


def load_molecular_hamiltonian_data(path: str) -> Optional[MolecularHamiltonianData]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return MolecularHamiltonianData(
        system=str(payload["system"]),
        basis=str(payload.get("basis", "unknown")),
        mapping=str(payload.get("mapping", "unknown")),
        coordinate_axis=str(payload.get("metadata", {}).get("coordinate_axis", "R")),
        coordinate_unit=str(payload.get("coordinates_unit", payload.get("bond_lengths_unit", "Angstrom"))),
        points=[_point_from_dict(point) for point in payload.get("points", [])],
        metadata=dict(payload.get("metadata", {})),
    )


def find_point_for_coordinate(
    data: MolecularHamiltonianData,
    *,
    axis: str,
    value: float,
) -> Optional[MolecularHamiltonianPoint]:
    if not data.points:
        return None
    return min(data.points, key=lambda point: abs(float(point.coordinates.get(axis, float("inf"))) - value))


def select_lowest_exact_point(data: MolecularHamiltonianData) -> Optional[MolecularHamiltonianPoint]:
    if not data.points:
        return None
    return min(data.points, key=lambda point: point.active_space_exact_energy)


def point_to_pauli_list(point: MolecularHamiltonianPoint) -> List[Tuple[float, List[Tuple[str, int]]]]:
    return [
        (float(term.coeff), [(str(op).lower(), int(idx)) for op, idx in term.ops])
        for term in point.paulis
    ]
