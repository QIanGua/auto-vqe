import json

import numpy as np

from core.molecular.builders import (
    MolecularBuilderSpec,
    builder_summary,
    clone_builder_with_coordinates,
    make_diatomic_geometry_factory,
    make_symmetric_triatomic_geometry_factory,
)
from core.molecular.generate import (
    default_output_path,
    format_registered_systems,
    generate_dataset,
    main as molecular_generate_main,
    parse_coordinate_values,
)
from core.molecular.loader import (
    find_point_for_coordinate,
    load_molecular_hamiltonian_data,
    point_to_pauli_list,
    select_lowest_exact_point,
)
from core.molecular.pauli import get_exact_from_paulis
from core.molecular.presets import (
    DEFAULT_LIH_BOND_LENGTHS_ANGSTROM,
    build_beh2_builder,
    build_h2_builder,
    build_lih_builder,
)
from core.molecular.registry import (
    get_molecular_builder,
    has_molecular_builder,
    list_molecular_builders,
    register_molecular_builder,
)
from core.molecular.generator import MolecularActiveSpaceSpec, MolecularProblemSpec
from core.molecular.schema import MolecularHamiltonianData, MolecularHamiltonianPoint, PauliTerm
from core.molecular.serialize import save_molecular_hamiltonian_data


def test_get_exact_from_paulis_single_qubit():
    paulis = [
        (-0.5, []),
        (0.25, [("z", 0)]),
    ]
    energy = get_exact_from_paulis(paulis, 1)
    assert np.isclose(energy, -0.75)


def test_loader_selects_closest_and_lowest_points(tmp_path):
    data = MolecularHamiltonianData(
        system="TestMol",
        basis="sto-3g",
        mapping="jordan_wigner",
        coordinate_axis="R",
        coordinate_unit="Angstrom",
        points=[
            MolecularHamiltonianPoint(
                coordinates={"R": 1.0},
                n_qubits=1,
                paulis=[PauliTerm(coeff=-0.5, ops=[]), PauliTerm(coeff=0.1, ops=[("z", 0)])],
                active_space_exact_energy=-0.6,
            ),
            MolecularHamiltonianPoint(
                coordinates={"R": 1.6},
                n_qubits=1,
                paulis=[PauliTerm(coeff=-0.5, ops=[]), PauliTerm(coeff=0.25, ops=[("z", 0)])],
                active_space_exact_energy=-0.75,
            ),
        ],
        metadata={"source": "unit-test"},
    )
    path = tmp_path / "molecular.json"
    save_molecular_hamiltonian_data(str(path), data)

    loaded = load_molecular_hamiltonian_data(str(path))
    assert loaded is not None
    assert loaded.coordinate_axis == "R"
    assert loaded.coordinate_unit == "Angstrom"

    closest = find_point_for_coordinate(loaded, axis="R", value=1.58)
    assert closest is not None
    assert np.isclose(closest.coordinates["R"], 1.6)

    lowest = select_lowest_exact_point(loaded)
    assert lowest is not None
    assert np.isclose(lowest.active_space_exact_energy, -0.75)

    paulis = point_to_pauli_list(lowest)
    assert paulis == [(-0.5, []), (0.25, [("z", 0)])]


def test_serializer_preserves_backward_compatible_R_field(tmp_path):
    data = MolecularHamiltonianData(
        system="TestMol",
        basis="sto-3g",
        mapping="jordan_wigner",
        coordinate_axis="R",
        coordinate_unit="Angstrom",
        points=[
            MolecularHamiltonianPoint(
                coordinates={"R": 1.6},
                n_qubits=1,
                paulis=[PauliTerm(coeff=-0.5, ops=[])],
                active_space_exact_energy=-0.5,
            )
        ],
    )
    path = tmp_path / "dataset.json"
    save_molecular_hamiltonian_data(str(path), data)

    payload = json.loads(path.read_text())
    assert payload["bond_lengths_unit"] == "Angstrom"
    assert payload["metadata"]["coordinate_axis"] == "R"
    assert np.isclose(payload["points"][0]["R"], 1.6)


def test_geometry_factories_support_diatomic_and_symmetric_triatomic():
    lih_geometry = make_diatomic_geometry_factory("Li", "H")({"R": 1.6})
    assert lih_geometry == [
        ("Li", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.6)),
    ]

    beh2_geometry = make_symmetric_triatomic_geometry_factory("Be", "H")({"R": 1.3})
    assert beh2_geometry == [
        ("H", (0.0, 0.0, -1.3)),
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.3)),
    ]


def test_builder_registry_and_coordinate_clone():
    name = "unit_test_h2_builder"
    spec = MolecularBuilderSpec(
        problem=MolecularProblemSpec(
            system="H2",
            geometry=[
                ("H", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, 0.74)),
            ],
            basis="sto-3g",
            description="H2 scan",
        ),
        active_space=MolecularActiveSpaceSpec(
            occupied_indices=[],
            active_indices=[0, 1],
            mapping="jordan_wigner",
        ),
        coordinate_axis="R",
        coordinate_unit="Angstrom",
        coordinate_values=[0.5, 0.7, 1.0],
        geometry_factory=make_diatomic_geometry_factory("H", "H"),
    )
    register_molecular_builder(name, spec)

    assert has_molecular_builder(name)
    assert name in list_molecular_builders()
    fetched = get_molecular_builder(name)
    assert fetched.problem.system == "H2"

    cloned = clone_builder_with_coordinates(fetched, [0.6, 0.8])
    summary = builder_summary(cloned)
    assert summary["system"] == "H2"
    assert summary["n_points"] == 2
    assert cloned.coordinate_values == [0.6, 0.8]


def test_builtin_presets_are_available_from_registry():
    assert {"beh2", "h2", "lih"}.issubset(set(list_molecular_builders()))

    h2 = get_molecular_builder("h2")
    lih = get_molecular_builder("lih")
    beh2 = get_molecular_builder("beh2")

    assert h2.problem.system == "H2"
    assert lih.problem.system == "LiH"
    assert beh2.problem.system == "BeH2"
    assert lih.coordinate_values == DEFAULT_LIH_BOND_LENGTHS_ANGSTROM

    assert build_h2_builder().problem.system == "H2"
    assert build_lih_builder().problem.system == "LiH"
    assert build_beh2_builder().problem.system == "BeH2"


def test_cli_helpers_parse_grid_and_format_defaults():
    assert parse_coordinate_values(None) is None
    assert parse_coordinate_values("1.0, 1.2,1.4") == [1.0, 1.2, 1.4]
    assert default_output_path("LiH").endswith("artifacts/molecular/lih_pyscf_data.json")
    assert format_registered_systems(["lih", "h2"]) == "h2, lih"


def test_generate_dataset_uses_registry_and_coordinate_override(monkeypatch):
    captured = {}

    def fake_build_scan_dataset(builder):
        captured["builder"] = builder
        return "dataset"

    monkeypatch.setattr("core.molecular.generate.build_scan_dataset", fake_build_scan_dataset)

    dataset = generate_dataset("LiH", coordinate_values=[1.1, 1.5])
    assert dataset == "dataset"
    assert captured["builder"].problem.system == "LiH"
    assert captured["builder"].coordinate_values == [1.1, 1.5]


def test_cli_list_mode_does_not_require_system(capsys):
    exit_code = molecular_generate_main(["--list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "lih" in captured.out
