from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PauliTerm:
    coeff: float
    ops: List[Tuple[str, int]]


@dataclass
class MolecularHamiltonianPoint:
    coordinates: Dict[str, float]
    n_qubits: int
    paulis: List[PauliTerm]
    active_space_exact_energy: float
    hf_energy: Optional[float] = None
    ccsd_energy: Optional[float] = None
    full_fci_energy: Optional[float] = None
    nuclear_repulsion: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MolecularHamiltonianData:
    system: str
    basis: str
    mapping: str
    coordinate_axis: str
    coordinate_unit: str
    points: List[MolecularHamiltonianPoint]
    metadata: Dict[str, Any] = field(default_factory=dict)
