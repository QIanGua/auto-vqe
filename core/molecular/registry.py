from __future__ import annotations

from typing import Dict, List

from core.molecular.builders import MolecularBuilderSpec
from core.molecular.presets import build_beh2_builder, build_h2_builder, build_lih_builder


_MOLECULAR_BUILDERS: Dict[str, MolecularBuilderSpec] = {
    "beh2": build_beh2_builder(),
    "h2": build_h2_builder(),
    "lih": build_lih_builder(),
}


def register_molecular_builder(name: str, spec: MolecularBuilderSpec) -> None:
    _MOLECULAR_BUILDERS[str(name).lower()] = spec


def get_molecular_builder(name: str) -> MolecularBuilderSpec:
    key = str(name).lower()
    if key not in _MOLECULAR_BUILDERS:
        raise KeyError(f"Unknown molecular builder: {name}")
    return _MOLECULAR_BUILDERS[key]


def has_molecular_builder(name: str) -> bool:
    return str(name).lower() in _MOLECULAR_BUILDERS


def list_molecular_builders() -> List[str]:
    return sorted(_MOLECULAR_BUILDERS.keys())
