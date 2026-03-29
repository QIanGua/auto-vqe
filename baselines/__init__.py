"""
Baseline Zoo
------------

Baselines now share the same `AnsatzSpec` abstraction as the generator /
evaluator stack. This module keeps only light compatibility helpers and re-
exports the canonical model types from `core.model.schemas`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from core.foundation.base_env import QuantumEnvironment
from core.model.schemas import AnsatzSpec


CircuitBuilder = Callable[[Any], tuple[Any, int]]


def _merge_config(
    base: Mapping[str, Any],
    override: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """
    Shallow-merge `override` into `base`, returning a new dict.
    Lists / nested dicts are not merged deeply – they are replaced as units.
    """
    merged: Dict[str, Any] = dict(base)
    if override:
        for k, v in override.items():
            merged[k] = v
    return merged


__all__ = [
    "AnsatzSpec",
    "CircuitBuilder",
    "_merge_config",
    "QuantumEnvironment",
]
