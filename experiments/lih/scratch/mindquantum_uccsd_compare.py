#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy>=1.24,<2.0",
#   "scipy>=1.10,<2.0",
#   "pyscf>=2.12.1",
#   "openfermion>=1.7.1",
#   "openfermionpyscf>=0.5",
#   "mindquantum>=0.12,<0.13",
#   "mindspore",
# ]
# ///

"""
Reproduce the MindQuantum LiH UCCSD tutorial in a single uv-script file.

Usage:
  uv run experiments/lih/scratch/mindquantum_uccsd_compare.py --dist 1.5
  uv run experiments/lih/scratch/mindquantum_uccsd_compare.py --dist 1.6

By default this script follows the official MindQuantum tutorial:
  - LiH
  - STO-3G
  - full molecular space (no active-space reduction)
  - generate_uccsd(...)
  - BFGS on the MindQuantum gradient backend

The main purpose is to compare the official full-space result with this repo's
4-qubit active-space LiH baseline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from scipy.optimize import minimize

import mindspore as ms  # ty:ignore[unresolved-import]
from mindquantum.algorithm.nisq import generate_uccsd  # ty:ignore[unresolved-import]
from mindquantum.core.circuit import Circuit  # ty:ignore[unresolved-import]
from mindquantum.core.gates import X  # ty:ignore[unresolved-import]
from mindquantum.core.operators import Hamiltonian  # ty:ignore[unresolved-import]
from mindquantum.simulator import Simulator  # ty:ignore[unresolved-import]


ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")


@dataclass
class MQRunSummary:
    dist_angstrom: float
    basis: str
    n_qubits: int
    n_electrons: int
    hf_energy: float
    ccsd_energy: float
    fci_energy: float
    uccsd_vqe_energy: float
    energy_error_to_fci: float
    n_params: int
    data_file: str


def optimization_fun(params: np.ndarray, pqc, energy_list: list[float] | None = None):
    value, grad = pqc(params)
    value = float(np.real(value)[0, 0])
    grad = np.real(grad)[0, 0]
    if energy_list is not None:
        energy_list.append(value)
        if len(energy_list) % 5 == 0:
            print(f"Step: {len(energy_list):<3} energy: {value:.12f}")
    return value, grad


def run_mindquantum_uccsd(dist: float, data_dir: Path) -> MQRunSummary:
    geometry = [["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, float(dist)]]]
    basis = "sto3g"
    spin = 0

    molecule = MolecularData(
        geometry,
        basis,
        multiplicity=2 * spin + 1,
        data_directory=str(data_dir),
    )
    molecule = run_pyscf(
        molecule,
        run_scf=True,
        run_ccsd=True,
        run_fci=True,
    )
    molecule.save()

    ansatz_circuit, init_amplitudes, ansatz_parameter_names, hamiltonian_qubit_op, n_qubits, n_electrons = generate_uccsd(
        molecule.filename,
        threshold=-1,
    )

    hf_circuit = Circuit([X.on(i) for i in range(n_electrons)])
    total_circuit = hf_circuit + ansatz_circuit

    sim = Simulator("mqvector", total_circuit.n_qubits)
    molecule_pqc = sim.get_expectation_with_grad(Hamiltonian(hamiltonian_qubit_op), total_circuit)

    p0 = np.array(init_amplitudes, dtype=np.float64)
    energy_list: list[float] = []
    result = minimize(optimization_fun, p0, args=(molecule_pqc, energy_list), method="bfgs", jac=True)

    return MQRunSummary(
        dist_angstrom=float(dist),
        basis=basis,
        n_qubits=int(n_qubits),
        n_electrons=int(n_electrons),
        hf_energy=float(molecule.hf_energy),
        ccsd_energy=float(molecule.ccsd_energy),
        fci_energy=float(molecule.fci_energy),
        uccsd_vqe_energy=float(result.fun),
        energy_error_to_fci=float(result.fun - molecule.fci_energy),
        n_params=len(ansatz_parameter_names),
        data_file=molecule.filename,
    )


def load_repo_lih_reference() -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    data_path = repo_root / "experiments" / "lih" / "data" / "lih_pyscf_data.json"
    with data_path.open() as f:
        data = json.load(f)
    return data


def print_repo_reference_context() -> None:
    data = load_repo_lih_reference()
    pts = data["points"]
    active_16 = min(pts, key=lambda p: abs(float(p["R"]) - 1.6))
    print("\nRepo LiH reference context")
    print("-------------------------")
    print(f"Stored grid point nearest 1.6 A: R = {active_16['R']}")
    if "active_space_exact_energy" in active_16:
        print(f"Active-space exact       : {active_16['active_space_exact_energy']}")
        print(f"Full-space FCI           : {active_16['full_fci_energy']}")
        print(f"Nuclear repulsion        : {active_16['nuclear_repulsion']}")
    else:
        print(f"Legacy exact_energy field: {active_16['exact_energy']}")
    print("Repo baseline currently compares against the 4-qubit active-space exact energy, not full-space FCI.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MindQuantum UCCSD LiH and compare with the repo baseline.")
    parser.add_argument("--dist", type=float, default=1.5, help="Li-H bond length in Angstrom. Official tutorial uses 1.5.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "cache" / "mindquantum",
        help="Directory used by OpenFermion MolecularData.",
    )
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to write a JSON summary.")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)

    summary = run_mindquantum_uccsd(args.dist, args.data_dir)

    print("\nMindQuantum UCCSD summary")
    print("-------------------------")
    print(f"dist           : {summary.dist_angstrom}")
    print(f"basis          : {summary.basis}")
    print(f"n_qubits       : {summary.n_qubits}")
    print(f"n_electrons    : {summary.n_electrons}")
    print(f"HF             : {summary.hf_energy:.15f}")
    print(f"CCSD           : {summary.ccsd_energy:.15f}")
    print(f"FCI            : {summary.fci_energy:.15f}")
    print(f"UCCSD-VQE      : {summary.uccsd_vqe_energy:.15f}")
    print(f"UCCSD-FCI diff : {summary.energy_error_to_fci:.15e}")
    print(f"n_params       : {summary.n_params}")
    print(f"molecule file  : {summary.data_file}")

    print_repo_reference_context()

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w") as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"\nSaved summary to: {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
