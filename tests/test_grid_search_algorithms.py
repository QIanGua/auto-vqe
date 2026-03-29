import os
import shutil

import pytest
import torch

from core.generator.grid import GridSearchStrategy, ansatz_search


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
    path = "./tmp_test_search_grid"
    os.makedirs(path, exist_ok=True)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture(autouse=True)
def mock_report_utils(monkeypatch):
    monkeypatch.setattr("core.generator.grid.generate_report", lambda *args, **kwargs: "/tmp/mock_report.png")


def test_grid_search_strategy(exp_dir):
    env = MockEnv()
    config_list = [
        {"n_layers": 1, "rotation": "rx"},
        {"n_layers": 2, "rotation": "ry"},
    ]

    strategy = GridSearchStrategy(
        env=env,
        make_create_circuit_fn=mock_make_circuit_fn,
        config_list=config_list,
        exp_dir=exp_dir,
        base_exp_name="GridTest",
        trials_per_config=1,
        max_steps=5,
    )

    result = strategy.run()
    assert "best_results" in result
    assert result["best_results"]["val_energy"] == -0.5


def test_grid_search_skips_failed_config_after_success(exp_dir, monkeypatch):
    env = MockEnv()
    config_list = [
        {"n_layers": 1, "rotation": "rx"},
        {"n_layers": 2, "rotation": "ry"},
    ]
    calls = {"count": 0}

    def fake_vqe_train(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "val_energy": -0.5,
                "num_params": 2,
                "final_params": [0.1, 0.2],
                "energy_history": [],
                "training_seconds": 0.0,
                "actual_steps": 1,
                "converged": False,
                "two_qubit_gate_count": 0,
            }
        raise RuntimeError("boom")

    monkeypatch.setattr("core.generator.grid.vqe_train", fake_vqe_train)
    monkeypatch.setattr("core.generator.grid.log_results", lambda *args, **kwargs: None)
    monkeypatch.setattr("core.generator.grid.generate_report", lambda *args, **kwargs: "/tmp/mock_report.png")

    result = ansatz_search(
        env=env,
        make_create_circuit_fn=mock_make_circuit_fn,
        config_list=config_list,
        exp_dir=exp_dir,
        base_exp_name="GridFailureRecovery",
        trials_per_config=1,
        max_steps=5,
    )

    assert result["best_config"] == config_list[0]
    assert result["best_results"]["val_energy"] == -0.5
