from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.shared import (
    BaselineSpec,
    ExperimentManifest,
    OrchestrationPhase,
    OrchestrationSpec,
    SearchSpec,
    load_best_config,
    run_baseline_experiment,
    run_config_experiment,
    run_orchestration_experiment,
    run_research_step,
    run_search_experiment,
)


SYSTEM_DIR = os.path.dirname(__file__)
RUNS_DIR = os.path.join(SYSTEM_DIR, "artifacts", "runs")
HF_QUBITS = [0, 1]
BASELINE_CONFIG = {
    "init_state": "hf",
    "hf_qubits": [0, 1],
    "occupied_orbitals": [0, 1],
    "virtual_orbitals": [2, 3],
    "layers": 1,
    "include_singles": True,
    "include_doubles": True,
    "mapping": "jordan_wigner",
    "trotter_order": 1,
}


def load_env():
    from experiments.lih.env import ENV

    return ENV


def build_circuit(config):
    from core.representation.compiler import build_ansatz

    final_config = dict(config)
    if final_config.get("init_state") == "hf" and "hf_qubits" not in final_config:
        final_config["hf_qubits"] = HF_QUBITS
    return build_ansatz(final_config, load_env().n_qubits)


def build_baseline(env, config):
    from baselines.uccsd import build_ansatz

    return build_ansatz(env, config or BASELINE_CONFIG)


MANIFEST = ExperimentManifest(
    name="lih",
    system_dir=SYSTEM_DIR,
    runs_dir=RUNS_DIR,
    fallback_config={
        "init_state": "hf",
        "hf_qubits": [0, 1],
        "layers": 2,
        "single_qubit_gates": ["ry", "rz"],
        "two_qubit_gate": "rzz",
        "entanglement": "linear",
    },
    config_priority=("presets/ga.json", "presets/multidim.json", "best_config.json"),
    run_result_label="LiH_Phase10",
    run_report_label="LiH_Phase10_Report",
    run_default_trials=2,
    run_seed_base=300,
    run_max_steps=800,
    run_lr=0.05,
    load_env=load_env,
    build_circuit=build_circuit,
    baseline=BaselineSpec(
        result_label="LiH_UCCSD_Baseline",
        report_label="LiH_UCCSD_Baseline_Report",
        run_slug="lih_baseline_uccsd",
        default_trials=5,
        max_steps=1500,
        lr=0.01,
        seed_base=50,
        builder=build_baseline,
        config=BASELINE_CONFIG,
    ),
    searches={
        "ga": SearchSpec(
            kind="ga",
            dimensions={
                "init_state": ["zero", "hf"],
                "layers": [2, 3, 4, 5],
                "single_qubit_gates": [["ry"], ["ry", "rz"], ["rx", "ry", "rz"]],
                "two_qubit_gate": ["cnot", "rzz", "rxx_ryy_rzz"],
                "entanglement": ["linear", "ring", "full"],
            },
            base_exp_name="LiH_GA_Search",
            run_slug="lih_ga_search",
            pop_size=10,
            generations=6,
            mutation_rate=0.4,
            elite_count=2,
            trials_per_config=2,
            max_steps=600,
            lr=0.05,
        ),
        "multidim": SearchSpec(
            kind="multidim",
            dimensions={
                "init_state": ["zero", "hf"],
                "layers": [1, 2, 3, 4],
                "single_qubit_gates": [["ry"], ["ry", "rz"]],
                "two_qubit_gate": ["cnot", "rzz"],
                "entanglement": ["linear", "ring"],
            },
            base_exp_name="LiH_MultiDim",
            run_slug="lih_multidim_search",
            trials_per_config=1,
            max_steps=600,
            lr=0.05,
        ),
    },
    orchestration=OrchestrationSpec(
        run_slug="lih_auto_search",
        max_runs=15,
        no_improvement_limit=3,
        failure_limit=3,
        phases=(
            OrchestrationPhase(
                kind="ga",
                dimensions={
                    "init_state": ["hf"],
                    "layers": [2, 3, 4],
                    "single_qubit_gates": [["ry"], ["ry", "rz"]],
                    "two_qubit_gate": ["cnot", "rzz"],
                    "entanglement": ["linear", "ring"],
                },
                base_exp_name="AutoSearch_LiH_Phase1_GA",
                pop_size=6,
                generations=2,
            ),
            OrchestrationPhase(
                kind="multidim",
                dimensions={
                    "init_state": ["hf"],
                    "layers": [2, 3],
                    "single_qubit_gates": [["ry", "rz"]],
                    "two_qubit_gate": ["rzz", "rxx_ryy_rzz"],
                    "entanglement": ["linear", "full"],
                },
                base_exp_name="AutoSearch_LiH_Phase2_Grid",
                trials_per_config=1,
                max_steps=600,
            ),
        ),
    ),
)


def get_default_ansatz_bundle():
    ansatz_config, config_path = load_best_config(MANIFEST)
    create_circuit, num_params = MANIFEST.build_circuit(ansatz_config)
    return ansatz_config, config_path, create_circuit, num_params


def run_geometry_scan(trials_per_R: int = 1, max_steps: int = 800, lr: float = 0.05):
    import torch

    from core.evaluator.api import prepare_experiment_dir
    from core.evaluator.logging_utils import log_results, setup_logger
    from core.evaluator.training import vqe_train
    from experiments.lih.data.geom_grid import BOND_LENGTHS_ANGSTROM
    from experiments.lih.env import LiHEnvironment

    default_ansatz_config, _, default_create_circuit, default_num_params = get_default_ansatz_bundle()
    exp_dir = prepare_experiment_dir(MANIFEST.runs_dir, "lih_geom_scan")

    log_path = os.path.join(exp_dir, "experiment.log")
    logger = setup_logger(log_path)
    logger.info("--- LiH Geometry Scan (structure transfer) ---")
    logger.info("Experiment Directory: %s", exp_dir)
    logger.info("Ansatz config: %s", default_ansatz_config)

    scan_results = []

    for R in BOND_LENGTHS_ANGSTROM:
        env = LiHEnvironment(R=R)
        exact_active = env.exact_energy
        full_fci = getattr(env, "full_fci_energy", exact_active)

        logger.info("\n=== Geometry R = %.3f Å ===", R)
        logger.info("Target energies: active_exact=%.8f, full_FCI=%.8f", exact_active, full_fci)

        best_results = None
        overall_best_energy = float("inf")

        def compute_energy_fn(params):
            c, _ = default_create_circuit(params)
            return env.compute_energy(c)

        for t in range(trials_per_R):
            logger.info("--- Trial %s/%s at R=%.3f Å ---", t + 1, trials_per_R, R)
            torch.manual_seed(300 + t)
            results = vqe_train(
                create_circuit_fn=default_create_circuit,
                compute_energy_fn=compute_energy_fn,
                n_qubits=env.n_qubits,
                exact_energy=exact_active,
                num_params=default_num_params,
                max_steps=max_steps,
                lr=lr,
                logger=logger,
            )
            if results["val_energy"] < overall_best_energy:
                overall_best_energy = results["val_energy"]
                best_results = results

        if best_results is None:
            logger.warning("No successful trials for R=%.3f Å, skipping.", R)
            continue

        best_results["exact_energy"] = exact_active
        best_results["energy_error"] = abs(best_results["val_energy"] - exact_active)
        best_results["full_fci_energy"] = full_fci
        best_results["error_vs_full"] = abs(best_results["val_energy"] - full_fci)
        best_results["truncation_error"] = abs(exact_active - full_fci)

        scan_results.append(
            {
                "R": R,
                "val_energy": float(best_results["val_energy"]),
                "exact_active": float(exact_active),
                "exact_fci": float(full_fci),
                "ansatz_error": float(best_results["energy_error"]),
                "error_vs_fci": float(best_results["error_vs_full"]),
                "truncation_error": float(best_results["truncation_error"]),
                "num_params": int(best_results["num_params"]),
            }
        )

        log_results(exp_dir, f"LiH_Geom_R_{R:.3f}", best_results, comment=f"LiH geometry scan, R={R:.3f} Å")

    curve_path = os.path.join(exp_dir, f"lih_geometry_curve_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
    with open(curve_path, "w", encoding="utf-8") as f:
        f.write("R_A\tvqe_energy\texact_active\texact_fci\terror_vs_fci\tansatz_error\ttruncation_error\tnum_params\n")
        for item in scan_results:
            f.write(
                f"{item['R']:.3f}\t{item['val_energy']:.8f}\t{item['exact_active']:.8f}\t{item['exact_fci']:.8f}\t"
                f"{item['error_vs_fci']:.3e}\t{item['ansatz_error']:.3e}\t{item['truncation_error']:.3e}\t{item['num_params']}\n"
            )

    logger.info("\nGeometry scan curve saved to: %s", curve_path)
    return scan_results


def _find_latest_curve_tsv(base_dir: str) -> str | None:
    files = glob.glob(os.path.join(base_dir, "**", "lih_geometry_curve_*.tsv"), recursive=True)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def _load_curve(path: str) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    rs: List[float] = []
    vqe: List[float] = []
    exact_active: List[float] = []
    ansatz_errors: List[float] = []
    trunc_errors: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                rs.append(float(parts[0]))
                vqe.append(float(parts[1]))
                exact_active.append(float(parts[2]))
                ansatz_errors.append(float(parts[5]))
                trunc_errors.append(float(parts[6]))
            elif len(parts) >= 4:
                rs.append(float(parts[0]))
                vqe.append(float(parts[1]))
                exact_active.append(float("nan"))
                ansatz_errors.append(float(parts[3]))
                trunc_errors.append(0.0)
    return rs, vqe, exact_active, ansatz_errors, trunc_errors


def plot_geometry_curve(curve_tsv: str | None = None, output_dir: str | None = None) -> None:
    system_dir = os.path.dirname(__file__)
    runs_dir = os.path.join(system_dir, "artifacts", "runs")

    curve_path = curve_tsv
    if curve_path:
        if not os.path.isabs(curve_path):
            curve_path = os.path.join(system_dir, curve_path)
    else:
        curve_path = _find_latest_curve_tsv(runs_dir)
    if not curve_path and output_dir:
        curve_path = _find_latest_curve_tsv(output_dir)
    if not curve_path or not os.path.exists(curve_path):
        print("No geometry-curve TSV found.")
        return

    save_dir = output_dir or os.path.dirname(curve_path)
    rs, vqe, exact_active, ansatz_errors, trunc_errors = _load_curve(curve_path)
    if not rs:
        print("Curve TSV appears to be empty or malformed.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    valid_r, valid_vqe, valid_active = [], [], []
    for r, ev, ea in zip(rs, vqe, exact_active):
        if ea is None or (isinstance(ea, float) and math.isnan(ea)):
            continue
        valid_r.append(r)
        valid_vqe.append(ev)
        valid_active.append(ea)

    if valid_r:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(valid_r, valid_active, "o-", label="Active-space exact", color="#4CAF50")
        ax.plot(valid_r, valid_vqe, "s--", label="VQE", color="#2196F3")
        ax.set_xlabel(r"Li-H bond length R ($\AA$)")
        ax.set_ylabel("Energy (Hartree)")
        ax.set_title("LiH: VQE vs Active-Space Exact")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "lih_geometry_curve_energy_active.png"), dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rs, ansatz_errors, "o-", color="#2196F3")
    ax.set_xlabel(r"Li-H bond length R ($\AA$)")
    ax.set_ylabel("|E_VQE - E_active| (Hartree)")
    ax.set_title("LiH Ansatz Error vs Bond Length")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "lih_geometry_curve_ansatz_error.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rs, trunc_errors, "o-", color="#FF9800")
    ax.set_xlabel(r"Li-H bond length R ($\AA$)")
    ax.set_ylabel("|E_active - E_FCI| (Hartree)")
    ax.set_title("LiH Active-Space Truncation Error vs Bond Length")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "lih_geometry_curve_truncation_error.png"), dpi=200)
    plt.close(fig)

    print(f"Using curve file: {curve_path}")
    print(f"Plotting to: {save_dir}")


@dataclass
class MQRunSummary:
    dist_angstrom: float
    basis: str
    n_qubits: int
    n_electrons: int
    hf_energy: float
    ccsd_energy: float
    fci_energy: float
    uccsd_vqe_energy: float
    energy_error_to_fci: float
    n_params: int
    data_file: str


def optimization_fun(params, pqc, energy_list: list[float] | None = None):
    import numpy as np

    value, grad = pqc(params)
    value = float(np.real(value)[0, 0])
    grad = np.real(grad)[0, 0]
    if energy_list is not None:
        energy_list.append(value)
        if len(energy_list) % 5 == 0:
            print(f"Step: {len(energy_list):<3} energy: {value:.12f}")
    return value, grad


def run_mindquantum_compare(dist: float, data_dir: Path, save_json: Path | None = None) -> int:
    import numpy as np
    from openfermion.chem import MolecularData
    from openfermionpyscf import run_pyscf
    from scipy.optimize import minimize

    import mindspore as ms  # ty:ignore[unresolved-import]
    from mindquantum.algorithm.nisq import generate_uccsd  # ty:ignore[unresolved-import]
    from mindquantum.core.circuit import Circuit  # ty:ignore[unresolved-import]
    from mindquantum.core.gates import X  # ty:ignore[unresolved-import]
    from mindquantum.core.operators import Hamiltonian  # ty:ignore[unresolved-import]
    from mindquantum.simulator import Simulator  # ty:ignore[unresolved-import]

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    geometry = [["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, float(dist)]]]
    molecule = MolecularData(geometry, "sto3g", multiplicity=1, data_directory=str(data_dir))
    molecule = run_pyscf(molecule, run_scf=True, run_ccsd=True, run_fci=True)
    molecule.save()

    ansatz_circuit, init_amplitudes, ansatz_parameter_names, hamiltonian_qubit_op, n_qubits, n_electrons = generate_uccsd(
        molecule.filename,
        threshold=-1,
    )
    total_circuit = Circuit([X.on(i) for i in range(n_electrons)]) + ansatz_circuit
    sim = Simulator("mqvector", total_circuit.n_qubits)
    molecule_pqc = sim.get_expectation_with_grad(Hamiltonian(hamiltonian_qubit_op), total_circuit)
    result = minimize(optimization_fun, np.array(init_amplitudes, dtype=np.float64), args=(molecule_pqc, []), method="bfgs", jac=True)

    summary = MQRunSummary(
        dist_angstrom=float(dist),
        basis="sto3g",
        n_qubits=int(n_qubits),
        n_electrons=int(n_electrons),
        hf_energy=float(molecule.hf_energy),
        ccsd_energy=float(molecule.ccsd_energy),
        fci_energy=float(molecule.fci_energy),
        uccsd_vqe_energy=float(result.fun),
        energy_error_to_fci=float(result.fun - molecule.fci_energy),
        n_params=len(ansatz_parameter_names),
        data_file=molecule.filename,
    )

    repo_data = json.loads((Path(__file__).resolve().parent / "data" / "lih_pyscf_data.json").read_text(encoding="utf-8"))
    active_16 = min(repo_data["points"], key=lambda p: abs(float(p["R"]) - 1.6))
    print("\nMindQuantum UCCSD summary")
    print("-------------------------")
    print(f"dist           : {summary.dist_angstrom}")
    print(f"basis          : {summary.basis}")
    print(f"n_qubits       : {summary.n_qubits}")
    print(f"n_electrons    : {summary.n_electrons}")
    print(f"HF             : {summary.hf_energy:.15f}")
    print(f"CCSD           : {summary.ccsd_energy:.15f}")
    print(f"FCI            : {summary.fci_energy:.15f}")
    print(f"UCCSD-VQE      : {summary.uccsd_vqe_energy:.15f}")
    print(f"UCCSD-FCI diff : {summary.energy_error_to_fci:.15e}")
    print(f"n_params       : {summary.n_params}")
    print(f"molecule file  : {summary.data_file}")
    print("\nRepo LiH reference context")
    print("-------------------------")
    print(f"Stored grid point nearest 1.6 A: R = {active_16['R']}")
    if "active_space_exact_energy" in active_16:
        print(f"Active-space exact       : {active_16['active_space_exact_energy']}")
        print(f"Full-space FCI           : {active_16['full_fci_energy']}")
        print(f"Nuclear repulsion        : {active_16['nuclear_repulsion']}")
    else:
        print(f"Legacy exact_energy field: {active_16['exact_energy']}")

    if save_json is not None:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        print(f"\nSaved summary to: {save_json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LiH experiments from one entrypoint.")
    parser.add_argument("--config", type=str, help="Path to explicit ansatz config JSON.")
    parser.add_argument("--trials", type=int, default=MANIFEST.run_default_trials, help="Number of trials with different seeds.")

    subparsers = parser.add_subparsers(dest="command")

    verify = subparsers.add_parser("verify", help="Verify the best-known or explicit config.")
    verify.add_argument("--config", type=str, help="Path to explicit ansatz config JSON.")
    verify.add_argument("--trials", type=int, default=MANIFEST.run_default_trials, help="Number of trials.")

    search = subparsers.add_parser("search", help="Run a structural search.")
    search.add_argument("strategy", choices=sorted(MANIFEST.searches.keys()))

    baseline = subparsers.add_parser("baseline", help="Run the baseline ansatz.")
    baseline.add_argument("--trials", type=int, default=MANIFEST.baseline.default_trials, help="Number of trials.")

    subparsers.add_parser("auto", help="Run the orchestration demo.")

    scan = subparsers.add_parser("scan", help="Run the LiH bond-length geometry scan.")
    scan.add_argument("--trials-per-R", type=int, default=1)
    scan.add_argument("--max-steps", type=int, default=800)
    scan.add_argument("--lr", type=float, default=0.05)

    plot = subparsers.add_parser("plot", help="Plot LiH geometry-scan outputs.")
    plot.add_argument("curve_tsv", nargs="?", help="Optional path to a `lih_geometry_curve_*.tsv` file.")
    plot.add_argument("--output-dir", type=str, help="Optional directory for plot outputs.")

    compare = subparsers.add_parser("compare", help="Compare against the MindQuantum LiH UCCSD workflow.")
    compare.add_argument("--dist", type=float, default=1.5)
    compare.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "cache" / "mindquantum")
    compare.add_argument("--save-json", type=Path, default=None)

    research = subparsers.add_parser("research-step", help="Run one research-loop search+verify iteration.")
    research.add_argument("--strategy", required=True, choices=sorted(MANIFEST.searches.keys()))
    research.add_argument("--verify-trials", type=int, default=2)

    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.command is None:
        return run_config_experiment(MANIFEST, trials=args.trials, explicit_config_path=args.config)
    if args.command == "verify":
        return run_config_experiment(MANIFEST, trials=args.trials, explicit_config_path=args.config)
    if args.command == "search":
        return run_search_experiment(MANIFEST, args.strategy)
    if args.command == "baseline":
        return run_baseline_experiment(MANIFEST, trials=args.trials)
    if args.command == "auto":
        return run_orchestration_experiment(MANIFEST)
    if args.command == "scan":
        return run_geometry_scan(trials_per_R=args.trials_per_R, max_steps=args.max_steps, lr=args.lr)
    if args.command == "plot":
        return plot_geometry_curve(curve_tsv=args.curve_tsv, output_dir=args.output_dir)
    if args.command == "compare":
        return run_mindquantum_compare(args.dist, args.data_dir, args.save_json)
    if args.command == "research-step":
        return run_research_step(MANIFEST, strategy_name=args.strategy, verify_trials=args.verify_trials)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
