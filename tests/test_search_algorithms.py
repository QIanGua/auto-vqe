import pytest
import numpy as np
import torch
import os
import shutil
from core.search_algorithms import GASearchStrategy
from core.engine import GridSearchStrategy
from core.controller import SearchController

class MockEnv:
    def __init__(self, n_qubits=2):
        self.n_qubits = n_qubits
        self.exact_energy = -1.0
    def compute_energy(self, circuit):
        # Dummy energy that depends on parameters (mocking dependency)
        return torch.tensor(-0.5, requires_grad=True)

def mock_make_circuit_fn(config):
    def create_circuit(params):
        # Returns a dummy circuit object with necessary metadata
        class DummyCircuit:
            def __init__(self, n): self._nqubits = n
        return DummyCircuit(2), len(params)
    return create_circuit, 2 # num_params=2

@pytest.fixture
def exp_dir():
    path = "./tmp_test_search"
    os.makedirs(path, exist_ok=True)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)

@pytest.fixture(autouse=True)
def mock_engine_utils(monkeypatch):
    monkeypatch.setattr("core.engine.generate_report", lambda *args, **kwargs: "/tmp/mock_report.png")
    monkeypatch.setattr("core.search_algorithms.generate_report", lambda *args, **kwargs: "/tmp/mock_report.png")

def test_ga_search_strategy(exp_dir):
    env = MockEnv()
    dimensions = {
        "n_layers": [1, 2],
        "rotation": ["rx", "ry"]
    }
    
    strategy = GASearchStrategy(
        env=env,
        make_circuit_fn=mock_make_circuit_fn,
        dimensions=dimensions,
        pop_size=4,
        generations=2,
        trials_per_config=1,
        max_steps=5,
        exp_dir=exp_dir
    )
    
    result = strategy.run()
    assert "best_config" in result
    assert "best_results" in result
    assert result["best_results"]["val_energy"] == -0.5

def test_grid_search_strategy(exp_dir):
    env = MockEnv()
    config_list = [
        {"n_layers": 1, "rotation": "rx"},
        {"n_layers": 2, "rotation": "ry"}
    ]
    
    strategy = GridSearchStrategy(
        env=env,
        make_create_circuit_fn=mock_make_circuit_fn,
        config_list=config_list,
        exp_dir=exp_dir,
        base_exp_name="GridTest",
        trials_per_config=1,
        max_steps=5
    )
    
    result = strategy.run()
    assert "best_results" in result
    assert result["best_results"]["val_energy"] == -0.5
