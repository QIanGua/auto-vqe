import json
import os
import shutil
from dataclasses import replace

import pytest
import torch

from core.generator.ga import GASearchStrategy


class MockEnv:
    def __init__(self, n_qubits=2):
        self.n_qubits = n_qubits
        self.exact_energy = -1.0

    def compute_energy(self, circuit):
        return torch.tensor(-0.5, requires_grad=True)


def mock_make_circuit_fn(config):
    def create_circuit(params):
        class DummyCircuit:
            def __init__(self, n):
                self._nqubits = n

        return DummyCircuit(2), len(params)

    return create_circuit, 2


@pytest.fixture
def exp_dir():
    path = "./tmp_test_search_ga"
    os.makedirs(path, exist_ok=True)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture(autouse=True)
def mock_report_utils(monkeypatch):
    monkeypatch.setattr("core.generator.ga.generate_report", lambda *args, **kwargs: "/tmp/mock_report.png")


def test_ga_search_strategy(exp_dir):
    env = MockEnv()
    dimensions = {
        "n_layers": [1, 2],
        "rotation": ["rx", "ry"],
    }

    strategy = GASearchStrategy(
        env=env,
        make_circuit_fn=mock_make_circuit_fn,
        dimensions=dimensions,
        pop_size=4,
        generations=2,
        trials_per_config=1,
        max_steps=5,
        exp_dir=exp_dir,
    )

    result = strategy.run()
    assert "best_config" in result
    assert "best_results" in result
    assert result["best_results"]["val_energy"] == -0.5


def test_lih_ga_search_writes_plain_best_config(tmp_path, monkeypatch):
    from experiments.lih.run import MANIFEST
    from experiments.shared import run_search_experiment

    expected_config = {
        "init_state": "hf",
        "layers": 2,
        "single_qubit_gates": ["ry", "rz"],
        "two_qubit_gate": "rzz",
        "entanglement": "linear",
    }
    wrapped_result = {
        "best_config": expected_config,
        "best_results": {"val_energy": -1.23, "num_params": 6},
        "report_path": "/tmp/mock_report.md",
    }

    monkeypatch.setattr(
        "core.generator.ga.GASearchStrategy",
        lambda *args, **kwargs: type("DummyStrategy", (), {"run": lambda self: wrapped_result})(),
    )
    monkeypatch.setattr("core.evaluator.api.prepare_experiment_dir", lambda *args, **kwargs: str(tmp_path / "run"))
    temp_manifest = replace(MANIFEST, system_dir=str(tmp_path), runs_dir=str(tmp_path / "runs"))

    result = run_search_experiment(temp_manifest, "ga")

    assert result["best_config"] == expected_config
    assert result["best_results"] == wrapped_result["best_results"]
    assert result["best_config_path"] == str(tmp_path / "presets" / "ga.json")
    with open(tmp_path / "presets" / "ga.json", "r") as f:
        persisted = json.load(f)
    assert persisted == expected_config
