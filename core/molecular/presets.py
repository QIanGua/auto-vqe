from __future__ import annotations

from core.molecular.builders import (
    MolecularBuilderSpec,
    make_diatomic_geometry_factory,
    make_symmetric_triatomic_geometry_factory,
)
from core.molecular.generator import MolecularActiveSpaceSpec, MolecularProblemSpec

DEFAULT_H2_BOND_LENGTHS_ANGSTROM = [0.3, 0.5, 0.7, 0.74, 0.9, 1.2, 1.6, 2.0]
DEFAULT_LIH_BOND_LENGTHS_ANGSTROM = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0]
DEFAULT_BEH2_BOND_LENGTHS_ANGSTROM = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]


def build_h2_builder() -> MolecularBuilderSpec:
    return MolecularBuilderSpec(
        problem=MolecularProblemSpec(
            system="H2",
            geometry=[
                ("H", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, DEFAULT_H2_BOND_LENGTHS_ANGSTROM[0])),
            ],
            basis="sto-3g",
            multiplicity=1,
            charge=0,
            description="H2 minimal-basis scan",
        ),
        active_space=MolecularActiveSpaceSpec(
            occupied_indices=[],
            active_indices=[0, 1],
            mapping="jordan_wigner",
        ),
        coordinate_axis="R",
        coordinate_unit="Angstrom",
        coordinate_values=list(DEFAULT_H2_BOND_LENGTHS_ANGSTROM),
        geometry_factory=make_diatomic_geometry_factory("H", "H"),
    )


def build_lih_builder() -> MolecularBuilderSpec:
    return MolecularBuilderSpec(
        problem=MolecularProblemSpec(
            system="LiH",
            geometry=[
                ("Li", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, DEFAULT_LIH_BOND_LENGTHS_ANGSTROM[0])),
            ],
            basis="sto-3g",
            multiplicity=1,
            charge=0,
            description="LiH 4-qubit active-space scan",
        ),
        active_space=MolecularActiveSpaceSpec(
            occupied_indices=[0],
            active_indices=[1, 2],
            mapping="jordan_wigner",
        ),
        coordinate_axis="R",
        coordinate_unit="Angstrom",
        coordinate_values=list(DEFAULT_LIH_BOND_LENGTHS_ANGSTROM),
        geometry_factory=make_diatomic_geometry_factory("Li", "H"),
    )


def build_beh2_builder() -> MolecularBuilderSpec:
    return MolecularBuilderSpec(
        problem=MolecularProblemSpec(
            system="BeH2",
            geometry=[
                ("H", (0.0, 0.0, -DEFAULT_BEH2_BOND_LENGTHS_ANGSTROM[0])),
                ("Be", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, DEFAULT_BEH2_BOND_LENGTHS_ANGSTROM[0])),
            ],
            basis="sto-3g",
            multiplicity=1,
            charge=0,
            description="Linear symmetric BeH2 active-space scan",
        ),
        active_space=MolecularActiveSpaceSpec(
            occupied_indices=[0],
            active_indices=[1, 2, 3],
            mapping="jordan_wigner",
        ),
        coordinate_axis="R",
        coordinate_unit="Angstrom",
        coordinate_values=list(DEFAULT_BEH2_BOND_LENGTHS_ANGSTROM),
        geometry_factory=make_symmetric_triatomic_geometry_factory("Be", "H"),
    )
