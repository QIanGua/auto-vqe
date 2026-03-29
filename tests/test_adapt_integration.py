import json
from dataclasses import replace

from experiments.shared import (
    BaselineSpec,
    ExperimentManifest,
    OrchestrationSpec,
    SearchSpec,
    run_config_experiment,
    run_search_experiment,
)


class SimpleEnv:
    def __init__(self):
        self.n_qubits = 1
        self.exact_energy = -1.0
        self.name = "SimpleX"

    def compute_energy(self, circuit):
        import tensorcircuit as tc

        return -tc.backend.real(circuit.expectation([tc.gates.x(), [0]]))


def _manifest(tmp_path):
    return ExperimentManifest(
        name="simple",
        system_dir=str(tmp_path),
        runs_dir=str(tmp_path / "runs"),
        fallback_config={"layers": 1, "single_qubit_gates": ["ry"], "two_qubit_gate": "cnot", "entanglement": "linear"},
        config_priority=("best_config.json",),
        run_result_label="Simple",
        run_report_label="Simple_Report",
        run_default_trials=1,
        run_seed_base=0,
        run_max_steps=10,
        run_lr=0.1,
        load_env=lambda: SimpleEnv(),
        build_circuit=lambda config: __import__("core.representation.compiler", fromlist=["build_ansatz"]).build_ansatz(config, 1),
        baseline=BaselineSpec(
            result_label="Baseline",
            report_label="Baseline_Report",
            run_slug="baseline",
            default_trials=1,
            max_steps=10,
            lr=0.1,
            seed_base=0,
            builder=lambda env, cfg: None,
        ),
        searches={},
        orchestration=OrchestrationSpec(run_slug="auto", max_runs=1, no_improvement_limit=1),
    )


def test_run_config_experiment_supports_structured_ansatz_specs(tmp_path):
    manifest = _manifest(tmp_path)
    config_path = tmp_path / "structured.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "manual",
                "family": "blocks",
                "n_qubits": 1,
                "config": {},
                "blocks": [{"name": "ry", "family": "gate", "support_qubits": [0]}],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    result = run_config_experiment(manifest, trials=1, explicit_config_path=str(config_path))
    assert "val_energy" in result


def test_run_search_experiment_supports_adapt_variants(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    manifest = replace(
        manifest,
        searches={
            "adapt": SearchSpec(
                kind="adapt",
                dimensions={},
                base_exp_name="Adapt",
                run_slug="adapt",
                lr=0.1,
                max_steps=10,
                trials_per_config=1,
            ),
            "qubit_adapt": SearchSpec(
                kind="qubit_adapt",
                dimensions={},
                base_exp_name="QubitAdapt",
                run_slug="qubit_adapt",
                lr=0.1,
                max_steps=10,
                trials_per_config=1,
            ),
        },
    )

    monkeypatch.setattr(
        "core.generator.adapt.AdaptVQEStrategy.run",
        lambda self: {
            "best_results": {"val_energy": -1.0},
            "best_config": {"name": "adapt", "family": "adapt", "n_qubits": 1, "config": {}, "blocks": []},
            "ansatz_spec": {},
        },
    )

    adapt_result = run_search_experiment(manifest, "adapt")
    qubit_result = run_search_experiment(manifest, "qubit_adapt")

    assert adapt_result["best_config"]["family"] == "adapt"
    assert qubit_result["best_config"]["family"] == "adapt"
