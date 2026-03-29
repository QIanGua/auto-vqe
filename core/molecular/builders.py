from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from core.molecular.generator import (
    MolecularActiveSpaceSpec,
    MolecularProblemSpec,
    build_molecular_point,
)
from core.molecular.schema import MolecularHamiltonianData, MolecularHamiltonianPoint


Geometry = Sequence[Tuple[str, Tuple[float, float, float]]]
CoordinateMap = Dict[str, float]
GeometryFactory = Callable[[CoordinateMap], Geometry]


@dataclass
class MolecularBuilderSpec:
    problem: MolecularProblemSpec
    active_space: MolecularActiveSpaceSpec
    coordinate_axis: str
    coordinate_unit: str
    coordinate_values: List[float]
    geometry_factory: GeometryFactory
    metadata: Dict[str, Any] = field(default_factory=dict)
    run_scf: bool = True
    run_mp2: bool = False
    run_cisd: bool = False
    run_ccsd: bool = False
    run_fci: bool = True


def make_diatomic_geometry_factory(
    atom_a: str,
    atom_b: str,
    *,
    axis: str = "z",
) -> GeometryFactory:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]

    def factory(coordinates: CoordinateMap) -> Geometry:
        distance = float(coordinates["R"])
        atom_b_position = [0.0, 0.0, 0.0]
        atom_b_position[axis_index] = distance
        return [
            (atom_a, (0.0, 0.0, 0.0)),
            (atom_b, tuple(atom_b_position)),
        ]

    return factory


def make_symmetric_triatomic_geometry_factory(
    center_atom: str,
    side_atom: str,
    *,
    axis: str = "z",
) -> GeometryFactory:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]

    def factory(coordinates: CoordinateMap) -> Geometry:
        distance = float(coordinates["R"])
        left_position = [0.0, 0.0, 0.0]
        right_position = [0.0, 0.0, 0.0]
        left_position[axis_index] = -distance
        right_position[axis_index] = distance
        return [
            (side_atom, tuple(left_position)),
            (center_atom, (0.0, 0.0, 0.0)),
            (side_atom, tuple(right_position)),
        ]

    return factory


def _build_data_metadata(spec: MolecularBuilderSpec) -> Dict[str, Any]:
    metadata = dict(spec.metadata)
    metadata.setdefault("description", spec.problem.description)
    metadata.setdefault("charge", spec.problem.charge)
    metadata.setdefault("multiplicity", spec.problem.multiplicity)
    metadata.setdefault(
        "active_space",
        {
            "occupied_indices": list(spec.active_space.occupied_indices),
            "active_indices": list(spec.active_space.active_indices),
        },
    )
    return metadata


def _build_point_metadata(
    spec: MolecularBuilderSpec,
    coordinates: CoordinateMap,
) -> Dict[str, Any]:
    return {
        "description": f"{spec.problem.system}_{spec.coordinate_axis}_{coordinates[spec.coordinate_axis]:.3f}",
        "occupied_indices": list(spec.active_space.occupied_indices),
        "active_indices": list(spec.active_space.active_indices),
    }


def build_scan_dataset(spec: MolecularBuilderSpec) -> MolecularHamiltonianData:
    try:
        from openfermion import jordan_wigner
        from openfermion.chem import MolecularData
        from openfermionpyscf import run_pyscf
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "This builder requires `openfermion`, `openfermionpyscf` and `pyscf`.\n"
            "Install them with:\n"
            "  pip install pyscf openfermion openfermionpyscf\n"
        ) from e

    points: List[MolecularHamiltonianPoint] = []
    for coordinate_value in spec.coordinate_values:
        coordinates = {spec.coordinate_axis: float(coordinate_value)}
        geometry = spec.geometry_factory(coordinates)
        molecule = MolecularData(
            geometry=geometry,
            basis=spec.problem.basis,
            multiplicity=spec.problem.multiplicity,
            charge=spec.problem.charge,
            description=f"{spec.problem.system}_{spec.coordinate_axis}_{coordinate_value:.3f}",
        )
        molecule = run_pyscf(
            molecule,
            run_scf=spec.run_scf,
            run_mp2=spec.run_mp2,
            run_cisd=spec.run_cisd,
            run_ccsd=spec.run_ccsd,
            run_fci=spec.run_fci,
        )
        if spec.run_fci and molecule.fci_energy is None:
            raise RuntimeError(
                f"FCI energy not computed for {spec.problem.system} at "
                f"{spec.coordinate_axis}={coordinate_value}"
            )

        molecular_h = molecule.get_molecular_hamiltonian(
            occupied_indices=spec.active_space.occupied_indices,
            active_indices=spec.active_space.active_indices,
        )
        qubit_h = jordan_wigner(molecular_h)
        points.append(
            build_molecular_point(
                molecule,
                qubit_h,
                coordinates=coordinates,
                metadata=_build_point_metadata(spec, coordinates),
            )
        )

    return MolecularHamiltonianData(
        system=spec.problem.system,
        basis=spec.problem.basis,
        mapping=spec.active_space.mapping,
        coordinate_axis=spec.coordinate_axis,
        coordinate_unit=spec.coordinate_unit,
        points=sorted(points, key=lambda point: point.coordinates[spec.coordinate_axis]),
        metadata=_build_data_metadata(spec),
    )


def clone_builder_with_coordinates(
    spec: MolecularBuilderSpec,
    coordinate_values: Sequence[float],
) -> MolecularBuilderSpec:
    return MolecularBuilderSpec(
        problem=spec.problem,
        active_space=spec.active_space,
        coordinate_axis=spec.coordinate_axis,
        coordinate_unit=spec.coordinate_unit,
        coordinate_values=[float(value) for value in coordinate_values],
        geometry_factory=spec.geometry_factory,
        metadata=dict(spec.metadata),
        run_scf=spec.run_scf,
        run_mp2=spec.run_mp2,
        run_cisd=spec.run_cisd,
        run_ccsd=spec.run_ccsd,
        run_fci=spec.run_fci,
    )


def builder_summary(spec: MolecularBuilderSpec) -> Dict[str, Any]:
    return {
        "system": spec.problem.system,
        "basis": spec.problem.basis,
        "coordinate_axis": spec.coordinate_axis,
        "coordinate_unit": spec.coordinate_unit,
        "n_points": len(spec.coordinate_values),
        "active_space": {
            "occupied_indices": list(spec.active_space.occupied_indices),
            "active_indices": list(spec.active_space.active_indices),
        },
    }
