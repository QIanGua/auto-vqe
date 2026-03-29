from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List

from core.molecular.schema import MolecularHamiltonianData, MolecularHamiltonianPoint, PauliTerm


def _point_to_dict(point: MolecularHamiltonianPoint) -> Dict[str, Any]:
    payload = asdict(point)
    if "R" in point.coordinates:
        payload["R"] = point.coordinates["R"]
    return payload


def save_molecular_hamiltonian_data(path: str, data: MolecularHamiltonianData) -> None:
    metadata = dict(data.metadata)
    metadata.setdefault("coordinate_axis", data.coordinate_axis)
    payload = {
        "system": data.system,
        "basis": data.basis,
        "mapping": data.mapping,
        "coordinates_unit": data.coordinate_unit,
        "bond_lengths_unit": data.coordinate_unit,
        "metadata": metadata,
        "points": [_point_to_dict(point) for point in data.points],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
