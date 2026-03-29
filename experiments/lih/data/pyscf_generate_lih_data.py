"""
Use PySCF (+ OpenFermion) to generate LiH Hamiltonians and FCI energies
for a set of bond lengths, then save them into a JSON file which can be
consumed by `experiments.lih.env.LiHEnvironment`.

This script is offline with respect to the main VQE training loop:
the VQE code itself only needs the generated JSON and does not depend
on PySCF / OpenFermion at runtime.

Requirements (install in your Python environment before running):
    pip install pyscf openfermion openfermionpyscf

Usage (from project root):
    uv run python experiments/lih/data/pyscf_generate_lih_data.py
"""

from __future__ import annotations

import os

from core.molecular.generate import generate_dataset
from core.molecular.schema import MolecularHamiltonianData
from core.molecular.serialize import save_molecular_hamiltonian_data

try:
    from .geom_grid import BOND_LENGTHS_ANGSTROM
except (ImportError, ValueError):
    from experiments.lih.data.geom_grid import BOND_LENGTHS_ANGSTROM


def generate_lih_dataset() -> MolecularHamiltonianData:
    return generate_dataset("lih", coordinate_values=BOND_LENGTHS_ANGSTROM)


def save_lih_data_json(path: str, data: MolecularHamiltonianData) -> None:
    save_molecular_hamiltonian_data(path, data)
    print(f"Saved LiH PySCF data to: {path}")


def main() -> None:
    here = os.path.dirname(__file__)
    system_dir = os.path.dirname(here)
    out_path = os.path.join(system_dir, "data", "lih_pyscf_data.json")
    data = generate_lih_dataset()
    save_lih_data_json(out_path, data)


if __name__ == "__main__":
    main()
