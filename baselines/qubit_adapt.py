"""
Qubit-ADAPT-VQE style baseline (placeholder).

Similar to `baselines.adapt`, this module currently reuses the UCCSD-style
ansatz as a pragmatic stand-in, but tags the resulting specification as
belonging to the "qubit-adapt" family.

The unified interface:

    build_ansatz(env, config) -> AnsatzSpec

allows agents and experiment scripts to switch to a fully fledged
Qubit-ADAPT-VQE implementation in the future without touching call sites.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from . import AnsatzSpec, QuantumEnvironment
from .uccsd import build_ansatz as _uccsd_build_ansatz


def build_ansatz(env: QuantumEnvironment, config: Dict[str, Any] | None = None) -> AnsatzSpec:
    """
    Build a Qubit-ADAPT-style ansatz.

    Currently a thin wrapper on top of the UCCSD-style baseline with different
    `name` / `family` tags.
    """
    spec = _uccsd_build_ansatz(env, config)
    qubit_adapt_spec = replace(
        spec,
        name="qubit_adapt",
        family="qubit_adapt",
    )
    qubit_adapt_spec.metadata.setdefault(
        "description",
        "Qubit-ADAPT-style ansatz (currently UCCSD proxy)",
    )
    qubit_adapt_spec.metadata.setdefault(
        "notes",
        "TODO: replace with true Qubit-ADAPT-VQE builder.",
    )
    return qubit_adapt_spec


__all__ = ["build_ansatz"]

