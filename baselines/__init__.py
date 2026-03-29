"""
Baseline Zoo
------------

This package collects a set of *standard* VQE baselines that all expose the
same minimal interface:

    build_ansatz(env, config) -> AnsatzSpec

where:
  - `env`   is a `core.foundation.base_env.QuantumEnvironment` instance
  - `config` is a (possibly partial) configuration dict used to override
    sensible defaults for the given baseline family.

The returned `AnsatzSpec` is a lightweight description that can be used by
agents and experiment scripts to:
  - construct the actual quantum circuit (via `create_circuit`)
  - know the parameter count (`num_params`)
  - log a structured `ansatz_spec` dict into `results.jsonl`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping

from core.foundation.base_env import QuantumEnvironment


CircuitBuilder = Callable[[Any], tuple[Any, int]]


@dataclass
class AnsatzSpec:
    """
    Unified description of a baseline ansatz.

    Fields are intentionally simple so that `spec.to_logging_dict()` can be
    safely JSON-serialized and recorded in experiment logs.
    """

    # Human / paper-facing identifiers
    name: str                 # e.g. "hea", "uccsd", "hva"
    family: str               # high-level family, often same as name

    # Environment context
    env_name: str             # e.g. "TFIM", "LiH_R_1.6"
    n_qubits: int

    # Circuit constructor and parameter count
    create_circuit: CircuitBuilder
    num_params: int

    # Purely descriptive, JSON-serializable configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Optional free-form metadata (tags, notes, provenance)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_logging_dict(self) -> Dict[str, Any]:
        """
        Compress the ansatz description into a JSON-serializable dict that can
        be stored as `ansatz_spec` inside results.jsonl.

        By design this drops the non-serializable `create_circuit` callable and
        keeps only light-weight metadata.
        """
        return {
            "name": self.name,
            "family": self.family,
            "env_name": self.env_name,
            "n_qubits": self.n_qubits,
            "num_params": self.num_params,
            "config": self.config,
            "metadata": self.metadata,
        }


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
