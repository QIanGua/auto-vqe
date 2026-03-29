from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, Sequence

import torch

# Allow `python core/evaluator/render_report.py ...` from the repo root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.evaluator.report import generate_report, load_report_context
from core.evaluator.training import vqe_train


SYSTEM_MODULES = {
    "lih": "experiments.lih.run",
    "tfim": "experiments.tfim.run",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate markdown reports and visual assets for an existing experiment run."
    )
    parser.add_argument("--run-dir", required=True, help="Path to an experiment run directory containing run.json")
    parser.add_argument(
        "--recompute-if-missing",
        action="store_true",
        help="If report_context.json is missing, rerun optimization from the recorded config to recover final_params.",
    )
    parser.add_argument(
        "--markdown-only",
        action="store_true",
        help="Regenerate markdown only, without circuit/convergence assets.",
    )
    return parser


def load_run_record(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "run.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"run.json at {path} must contain a JSON object.")
    return payload


def load_manifest_for_system(system: str):
    key = str(system).lower()
    if key not in SYSTEM_MODULES:
        raise KeyError(f"Unsupported system '{system}'. Supported systems: {', '.join(sorted(SYSTEM_MODULES))}")
    module = importlib.import_module(SYSTEM_MODULES[key])
    return module.MANIFEST


def _results_from_record(record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(record.get("metrics", {}))
    exact_energy = record.get("measurement_spec", {}).get("exact_energy", metrics.get("exact_energy"))
    results = {
        "val_energy": metrics.get("val_energy"),
        "exact_energy": exact_energy,
        "energy_error": metrics.get("energy_error"),
        "num_params": metrics.get("num_params"),
        "runtime_sec": metrics.get("runtime_sec"),
        "training_seconds": metrics.get("runtime_sec"),
        "actual_steps": metrics.get("actual_steps"),
        "n_qubits": record.get("measurement_spec", {}).get("n_qubits"),
        "final_params": context.get("final_params"),
        "energy_history": context.get("energy_history", []),
    }
    return results


def _normalize_final_params(results: Dict[str, Any]) -> Dict[str, Any]:
    final_params = results.get("final_params")
    if final_params is not None and not isinstance(final_params, torch.Tensor):
        results["final_params"] = torch.tensor(final_params, dtype=torch.float32)
    return results


def recompute_results(record: Dict[str, Any], manifest, ansatz_spec: Dict[str, Any]) -> Dict[str, Any]:
    env = manifest.load_env()
    create_circuit_fn, num_params = manifest.build_circuit(ansatz_spec)

    def compute_energy_fn(params):
        circuit, _ = create_circuit_fn(params)
        return env.compute_energy(circuit)

    max_steps = record.get("metrics", {}).get("actual_steps") or manifest.run_max_steps
    return vqe_train(
        create_circuit_fn=create_circuit_fn,
        compute_energy_fn=compute_energy_fn,
        n_qubits=env.n_qubits,
        exact_energy=env.exact_energy,
        num_params=num_params,
        max_steps=int(max_steps),
        lr=manifest.run_lr,
    )


def regenerate_report(
    run_dir: str,
    *,
    recompute_if_missing: bool = False,
    markdown_only: bool = False,
) -> str:
    record = load_run_record(run_dir)
    manifest = load_manifest_for_system(record["system"])
    ansatz_spec = record.get("ansatz_spec")
    if not isinstance(ansatz_spec, dict):
        raise ValueError("run.json does not contain a usable ansatz_spec.")

    context = load_report_context(run_dir)
    results = _results_from_record(record, context)

    if results.get("final_params") is None:
        if not recompute_if_missing:
            raise FileNotFoundError(
                "report_context.json is missing or incomplete. "
                "Rerun with --recompute-if-missing to regenerate final_params."
            )
        results = recompute_results(record, manifest, ansatz_spec)

    results = _normalize_final_params(results)
    create_circuit_fn, _ = manifest.build_circuit(ansatz_spec)

    return generate_report(
        run_dir,
        record["exp_name"],
        results,
        create_circuit_fn,
        comment=record.get("comment", ""),
        ansatz_spec=ansatz_spec,
        decision=record.get("decision", "keep"),
        parent_experiment=record.get("parent_experiment"),
        change_summary=record.get("change_summary", ""),
        config_path=record.get("config_path_used"),
        render_markdown=True,
        render_assets=not markdown_only,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record_path = regenerate_report(
        args.run_dir,
        recompute_if_missing=args.recompute_if_missing,
        markdown_only=args.markdown_only,
    )
    print(f"Regenerated report artifacts for: {args.run_dir}")
    print(f"Updated run record: {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
