"""
Run a full LiH bond-length geometry scan with the current best Ansatz,
then immediately plot the resulting dissociation curve.

This is a thin wrapper around:
  - `run_geometry_scan` in `experiments.lih.run`
  - `main` in `experiments.lih.plot_geometry_curve`

Usage (from project root):
    uv run python experiments/lih/scan_and_plot.py

You can edit the default `trials_per_R`, `max_steps`, or `lr` below if
you want a finer or coarser scan without touching the core code.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> None:
    # Import inside main so that module-level side effects (like sys.path
    # tweaks in `run.py`) only happen when we actually execute the script.
    try:
        from .run import run_geometry_scan, ANSATZ_CONFIG
        from . import plot_geometry_curve
    except (ImportError, ValueError):
        from run import run_geometry_scan, ANSATZ_CONFIG
        import plot_geometry_curve

    # Simple, fixed hyperparameters; adjust here if needed.
    trials_per_R = 2
    max_steps = 800
    lr = 0.05

    print("=== LiH Geometry Scan + Plot ===")
    print("Using Ansatz config:")
    print(ANSATZ_CONFIG)
    print(f"\nScan settings: trials_per_R={trials_per_R}, max_steps={max_steps}, lr={lr}")

    # 1) Run the geometry scan; this will generate a TSV file and return scan_results.
    # We now need the exp_dir from the scan, but run_geometry_scan doesn't return it directly.
    # However, it will be the latest lih_geom_scan_* directory.
    scan_results = run_geometry_scan(
        trials_per_R=trials_per_R,
        max_steps=max_steps,
        lr=lr,
    )
    if not scan_results:
        print("Geometry scan produced no results; aborting plot.")
        return

    # 2) Plot the latest curve. 
    print("\nGeometry scan finished. Generating plots...")
    plot_geometry_curve.main(["plot_geometry_curve.py"])


if __name__ == "__main__":
    main(sys.argv)

