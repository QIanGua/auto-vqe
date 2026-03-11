"""
ADAPT-VQE style baseline (placeholder).

In a full implementation, ADAPT-VQE would *adaptively* grow the ansatz from a
pool of operators selected by gradient information. For the purposes of this
repository, we start with a pragmatic compromise:

  - reuse the UCCSD-style configuration as the underlying circuit;
  - tag the resulting `AnsatzSpec` as belonging to the "adapt" family so it
    can be cleanly separated in ablation studies and papers;
  - keep the same `build_ansatz(env, config) -> AnsatzSpec` interface so that
    future work can swap in a true ADAPT builder without changing call sites.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from . import AnsatzSpec, QuantumEnvironment
from .uccsd import build_ansatz as _uccsd_build_ansatz


def build_ansatz(env: QuantumEnvironment, config: Dict[str, Any] | None = None) -> AnsatzSpec:
    """
    Build an ADAPT-style ansatz.

    Currently this is a thin wrapper over the UCCSD-style baseline, with a
    different `family` / `name` tag. This keeps the public interface stable
    while leaving room for a future, gradient-driven operator selection loop.
    """
    spec = _uccsd_build_ansatz(env, config)
    # Re-tag without mutating the original spec in case callers keep a copy.
    adapted = replace(
        spec,
        name="adapt",
        family="adapt",
    )
    adapted.metadata.setdefault("description", "ADAPT-style ansatz (currently UCCSD proxy)")
    adapted.metadata.setdefault("notes", "TODO: replace with true ADAPT-VQE builder.")
    return adapted


__all__ = ["build_ansatz"]

