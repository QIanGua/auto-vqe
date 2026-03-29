from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Sequence

# Allow `python core/molecular/generate.py ...` from the repo root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.molecular.builders import build_scan_dataset, clone_builder_with_coordinates
from core.molecular.registry import get_molecular_builder, list_molecular_builders
from core.molecular.serialize import save_molecular_hamiltonian_data


def parse_coordinate_values(raw: str | None) -> List[float] | None:
    if raw is None:
        return None
    chunks = [chunk.strip() for chunk in raw.split(",")]
    values = [chunk for chunk in chunks if chunk]
    if not values:
        raise ValueError("Coordinate list must not be empty.")
    return [float(value) for value in values]


def default_output_path(system: str) -> str:
    filename = f"{system.lower()}_pyscf_data.json"
    return os.path.join("artifacts", "molecular", filename)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate molecular Hamiltonian datasets from the shared core registry."
    )
    parser.add_argument(
        "--system",
        help="Registered molecular system name, for example: lih, h2, beh2.",
    )
    parser.add_argument(
        "--grid",
        help="Comma-separated coordinate values, for example: 1.0,1.2,1.4",
    )
    parser.add_argument(
        "--out",
        help="Output JSON path. Defaults to artifacts/molecular/<system>_pyscf_data.json",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered systems and exit.",
    )
    return parser


def resolve_output_path(system: str, out: str | None) -> str:
    return out or default_output_path(system)


def generate_dataset(
    system: str,
    *,
    coordinate_values: Sequence[float] | None = None,
):
    builder = get_molecular_builder(system)
    if coordinate_values is not None:
        builder = clone_builder_with_coordinates(builder, coordinate_values)
    return build_scan_dataset(builder)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def format_registered_systems(systems: Iterable[str]) -> str:
    return ", ".join(sorted(systems))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print(format_registered_systems(list_molecular_builders()))
        return 0

    if not args.system:
        parser.error("--system is required unless --list is used.")

    try:
        coordinate_values = parse_coordinate_values(args.grid)
    except ValueError as exc:
        parser.error(str(exc))

    out_path = resolve_output_path(args.system, args.out)
    data = generate_dataset(args.system, coordinate_values=coordinate_values)
    ensure_parent_dir(out_path)
    save_molecular_hamiltonian_data(out_path, data)
    print(f"Saved {args.system} dataset to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
