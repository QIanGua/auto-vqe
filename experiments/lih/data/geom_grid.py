"""
Bond-length grid (in Angstrom) for LiH geometry scan.

You can freely modify `BOND_LENGTHS_ANGSTROM` to change the scan range
or resolution. Both the PySCF data generation script and the LiH scan
runner import this list, so keeping it here in one place avoids drift.
"""

BOND_LENGTHS_ANGSTROM = [
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.5,
    3.0,
    4.0,
]

