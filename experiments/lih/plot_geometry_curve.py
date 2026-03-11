"""
Plot LiH geometry-scan energies and errors from TSV files.

This script reads the TSV produced by `run_geometry_scan` in
`experiments.lih.run` and focuses on:

  - Energy curves:      R vs E_VQE, E_active_exact
  - Ansatz error:       |E_VQE - E_active_exact|
  - Truncation error:   |E_active_exact - E_full_FCI|

`run_geometry_scan` writes them into `lih_geometry_curve_*.tsv` with
the following columns (new format):

  R_A, vqe_energy, exact_active, exact_fci, error_vs_fci,
  ansatz_error, truncation_error, num_params

Usage (from project root):
    uv run python experiments/lih/plot_geometry_curve.py
or
    uv run python experiments/lih/plot_geometry_curve.py path/to/lih_geometry_curve_2026....tsv

If no path is given, the script will automatically pick the latest
`lih_geometry_curve_*.tsv` in `experiments/lih/`.
"""

from __future__ import annotations

import glob
import math
import os
import sys
from typing import List, Tuple


def _find_latest_curve_tsv(base_dir: str) -> str | None:
    pattern = os.path.join(base_dir, "lih_geometry_curve_*.tsv")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def _load_curve(
    path: str,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Load geometry-scan TSV.

    Returns:
        Rs, vqe_energies, exact_active, ansatz_errors, truncation_errors
    """
    Rs: List[float] = []
    vqe: List[float] = []
    exact_active: List[float] = []
    ansatz_errors: List[float] = []
    trunc_errors: List[float] = []

    with open(path, "r") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")

            # New format (preferred):
            # R_A, vqe_energy, exact_active, exact_fci, error_vs_fci,
            # ansatz_error, truncation_error, num_params
            if len(parts) >= 7:
                R_A = float(parts[0])
                vqe_E = float(parts[1])
                active_E = float(parts[2])
                ansatz_err = float(parts[5])
                trunc_err = float(parts[6])
            elif len(parts) >= 6:
                # Transitional format (before exact_active was added):
                # R_A, vqe_energy, exact_fci, error_vs_fci,
                # ansatz_error, truncation_error, ...
                R_A = float(parts[0])
                vqe_E = float(parts[1])
                active_E = float("nan")
                ansatz_err = float(parts[4])
                trunc_err = float(parts[5])
            elif len(parts) >= 4:
                # Very old format:
                # R_A, vqe_energy, exact, error
                R_A = float(parts[0])
                vqe_E = float(parts[1])
                active_E = float("nan")
                ansatz_err = float(parts[3])
                trunc_err = 0.0
            else:
                continue

            Rs.append(R_A)
            vqe.append(vqe_E)
            exact_active.append(active_E)
            ansatz_errors.append(ansatz_err)
            trunc_errors.append(trunc_err)

    return Rs, vqe, exact_active, ansatz_errors, trunc_errors


def main(argv: list[str]) -> None:
    here = os.path.dirname(__file__)

    if len(argv) > 1:
        curve_path = argv[1]
        if not os.path.isabs(curve_path):
            curve_path = os.path.join(here, curve_path)
    else:
        curve_path = _find_latest_curve_tsv(here)

    if not curve_path or not os.path.exists(curve_path):
        print("No geometry-curve TSV found.")
        print("Run `run_geometry_scan()` first to generate a TSV file.")
        return

    print(f"Using curve file: {curve_path}")

    Rs, vqe, exact_active, ansatz_errors, trunc_errors = _load_curve(curve_path)

    if not Rs:
        print("Curve TSV appears to be empty or malformed.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fix for missing glyphs (like Å or minus signs) in certain fonts
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1) Energy vs R: VQE vs active-space exact
    valid_R = []
    valid_vqe = []
    valid_active = []
    for R, Ev, Ea in zip(Rs, vqe, exact_active):
        if Ea is None or (isinstance(Ea, float) and math.isnan(Ea)):
            continue
        valid_R.append(R)
        valid_vqe.append(Ev)
        valid_active.append(Ea)

    if valid_R:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(valid_R, valid_active, "o-", label="Active-space exact", color="#4CAF50")
        ax.plot(valid_R, valid_vqe, "s--", label="VQE", color="#2196F3")
        ax.set_xlabel(r"Li-H bond length R ($\AA$)")
        ax.set_ylabel("Energy (Hartree)")
        ax.set_title("LiH: VQE vs Active-Space Exact")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        energy_png = os.path.join(here, "lih_geometry_curve_energy_active.png")
        fig.savefig(energy_png, dpi=200)
        plt.close(fig)
        print(f"Saved energy curve (VQE vs active) to: {energy_png}")
    else:
        print("No valid active-space energies found; skipping energy plot.")

    # 2) Ansatz error vs R
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Rs, ansatz_errors, "o-", color="#2196F3")
    ax.set_xlabel(r"Li-H bond length R ($\AA$)")
    ax.set_ylabel("|E_VQE - E_active| (Hartree)")
    ax.set_title("LiH Ansatz Error vs Bond Length")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    ansatz_png = os.path.join(here, "lih_geometry_curve_ansatz_error.png")
    fig.savefig(ansatz_png, dpi=200)
    plt.close(fig)

    # 3) Truncation error vs R
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Rs, trunc_errors, "o-", color="#FF9800")
    ax.set_xlabel(r"Li-H bond length R ($\AA$)")
    ax.set_ylabel("|E_active - E_FCI| (Hartree)")
    ax.set_title("LiH Active-Space Truncation Error vs Bond Length")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    trunc_png = os.path.join(here, "lih_geometry_curve_truncation_error.png")
    fig.savefig(trunc_png, dpi=200)
    plt.close(fig)

    print(f"Saved ansatz-error curve      to: {ansatz_png}")
    print(f"Saved truncation-error curve  to: {trunc_png}")


if __name__ == "__main__":
    main(sys.argv)

