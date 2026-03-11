"""
Use PySCF (+ OpenFermion) to generate LiH Hamiltonians and FCI energies
for a set of bond lengths, then save them into a JSON file which can be
consumed by `experiments.lih.env.LiHEnvironment`.

This script is *offline* with respect to the main VQE training loop:
the VQE code itself only needs the generated JSON and does not depend
on PySCF / OpenFermion at runtime.

Requirements (install in your Python environment before running):
    pip install pyscf openfermion openfermionpyscf

Usage (from project root):
    uv run python experiments/lih/pyscf_generate_lih_data.py
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

try:
    from .geom_grid import BOND_LENGTHS_ANGSTROM
except (ImportError, ValueError):
    from geom_grid import BOND_LENGTHS_ANGSTROM


@dataclass
class PauliTerm:
    coeff: float
    # ops is a list of [op_type, qubit_index], e.g. ["z", 0]
    ops: List[Tuple[str, int]]


@dataclass
class GeometryPoint:
    R: float  # bond length in Angstrom
    exact_energy: float  # FCI + nuclear repulsion, in Hartree
    n_qubits: int
    paulis: List[PauliTerm]


def _qubit_op_to_paulis(qubit_op) -> List[PauliTerm]:
    """
    Convert an OpenFermion QubitOperator into our (coeff, ops) format.
    """
    paulis: List[PauliTerm] = []
    for term, coeff in qubit_op.terms.items():
        if not term:
            paulis.append(PauliTerm(float(coeff.real), []))
            continue
        ops = []
        for idx, pauli_char in term:
            op_type = pauli_char.lower()  # 'X' -> 'x'
            if op_type not in ("x", "y", "z"):
                # Identity on a specific qubit is skipped; OpenFermion usually
                # encodes it by omitting that qubit from the term.
                continue
            ops.append((op_type, int(idx)))
        if ops:
            paulis.append(PauliTerm(float(coeff.real), ops))
    return paulis


def generate_lih_points() -> Dict[str, GeometryPoint]:
    """
    Run PySCF + OpenFermion for a grid of bond lengths.

    We deliberately construct a *4-qubit* effective Hamiltonian to match
    the existing LiH experiments in this repo:

      - Start from full LiH / STO-3G (12 spin-orbitals -> 12 qubits);
      - Freeze the lowest-energy core spatial orbital;
      - Keep a 2-orbital active space (2 spatial orbitals -> 4 spin-orbitals);
      - Map the resulting active-space Hamiltonian with Jordan–Wigner.

    This is a standard minimal active-space model for LiH often used as
    a 4-qubit benchmark. You can tweak `occupied_indices` /
    `active_indices` below if you want a different active space.

    Returns a dict keyed by a stringified bond length (e.g. "1.60")
    to make JSON serialization and lookup robust.
    """
    try:
        from openfermion import jordan_wigner, count_qubits
        from openfermion.chem import MolecularData
        from openfermionpyscf import run_pyscf
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "This script requires `openfermion`, `openfermionpyscf` and `pyscf`.\n"
            "Install them with:\n"
            "  pip install pyscf openfermion openfermionpyscf\n"
        ) from e

    basis = "sto-3g"
    multiplicity = 1
    charge = 0

    points: Dict[str, GeometryPoint] = {}

    for R in BOND_LENGTHS_ANGSTROM:
        geometry = [
            ("Li", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, float(R))),
        ]

        print(f"[PySCF] Running LiH at R = {R:.3f} Å ...")
        molecule = MolecularData(
            geometry=geometry,
            basis=basis,
            multiplicity=multiplicity,
            charge=charge,
            description=f"LiH_R_{R:.3f}",
        )

        molecule = run_pyscf(
            molecule,
            run_scf=True,
            run_mp2=False,
            run_cisd=False,
            run_ccsd=False,
            run_fci=True,
        )

        if molecule.fci_energy is None:
            raise RuntimeError(f"FCI energy not computed for R={R}")

        # NOTE:
        #   molecule.n_orbitals is the number of spatial orbitals.
        #   For LiH / STO-3G this is 6 (12 spin-orbitals).
        #
        # We build a minimal 2-orbital active space:
        #   - Freeze the lowest-energy spatial orbital (index 0);
        #   - Keep the next two spatial orbitals (indices 1, 2) active.
        #
        # This gives:
        #   2 active spatial orbitals -> 4 spin-orbitals -> 4 qubits
        #
        # The core / frozen energy contribution is folded into the
        # constant term of the molecular Hamiltonian.
        occupied_indices = [0]
        active_indices = [1, 2]

        molecular_h = molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices,
            active_indices=active_indices,
        )

        # Map the 4-qubit active-space Hamiltonian via Jordan–Wigner.
        qubit_h = jordan_wigner(molecular_h)

        # For consistency with earlier code, we store the *full* FCI
        # energy (electronic FCI + nuclear repulsion) as the reference.
        # In an exact calculation, diagonalizing `qubit_h` would recover
        # the same ground-state energy.
        exact_energy = float(molecule.fci_energy + molecule.nuclear_repulsion)

        pauli_terms = _qubit_op_to_paulis(qubit_h)

        # OpenFermion uses one qubit per spin-orbital; for our chosen
        # two-orbital active space this should be 4. `QubitOperator`
        # itself does not expose `n_qubits`, so we infer it via
        # `count_qubits`.
        n_qubits = int(count_qubits(qubit_h))

        key = f"{R:.3f}"
        points[key] = GeometryPoint(
            R=R,
            exact_energy=exact_energy,
            n_qubits=n_qubits,
            paulis=pauli_terms,
        )

    return points


def save_lih_data_json(path: str, points: Dict[str, GeometryPoint]) -> None:
    data = {
        "system": "LiH",
        "basis": "sto-3g",
        "mapping": "jordan_wigner",
        "bond_lengths_unit": "Angstrom",
        "points": [
            {
                "key": key,
                "R": pt.R,
                "exact_energy": pt.exact_energy,
                "n_qubits": pt.n_qubits,
                "paulis": [asdict(term) for term in pt.paulis],
            }
            for key, pt in sorted(points.items(), key=lambda kv: float(kv[0]))
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved LiH PySCF data to: {path}")


def main() -> None:
    here = os.path.dirname(__file__)
    out_path = os.path.join(here, "lih_pyscf_data.json")
    points = generate_lih_points()
    save_lih_data_json(out_path, points)


if __name__ == "__main__":
    main()
