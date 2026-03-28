import pytest
import os
import shutil
import torch
import numpy as np
from core.engine import generate_report, vqe_train, setup_logger
from core.schemas import AnsatzSpec
from baselines.uccsd import build_ansatz
from experiments.lih.env import ENV

def test_vqe_train_legacy_wrapper(tmp_path):
    # Test the new wrapper around optimize_parameters
    def create_circuit_fn(params):
        class Dummy:
            def __init__(self, n): self._nqubits = n
            def to_json(self): return []
        return Dummy(2), 2

    def compute_energy_fn(params):
        return torch.tensor(params[0]**2 + params[1]**2 - 1.0, requires_grad=True)

    res = vqe_train(
        create_circuit_fn=create_circuit_fn,
        compute_energy_fn=compute_energy_fn,
        n_qubits=2,
        num_params=2,
        max_steps=50, # More steps for convergence
        exact_energy=-1.0,
        lr=0.1 # Higher LR for fast mock convergence
    )
    assert "val_energy" in res
    assert "energy_error" in res
    assert isinstance(res["val_energy"], float)
    assert isinstance(res["final_params"], torch.Tensor)

def test_optimize_parameters_edge_cases():
    from core.engine import optimize_parameters
    # 1. Test missing both ansatz and create_circuit_fn
    with pytest.raises(ValueError, match="Either ansatz or create_circuit_fn must be provided"):
        optimize_parameters(env=None, ansatz=None, create_circuit_fn=None)
    
    # 2. Test default optimizer_spec=None
    from core.schemas import OptimizerSpec
    def mock_compute_energy(params): return torch.tensor(0.0, requires_grad=True)
    def mock_create(params): return None, 0
    res = optimize_parameters(env=None, compute_energy_fn=mock_compute_energy, create_circuit_fn=mock_create, optimizer_spec=None)
    assert res["actual_steps"] >= 0

def test_generate_report_minimal(tmp_path):
    # Test generate_report with minimal mocks
    exp_dir = str(tmp_path)
    
    def mock_create_circuit_fn(params):
        import tensorcircuit as tc
        c = tc.Circuit(2)
        # Use simple gates that always exist
        c.rz(0, theta=0.1)
        c.cx(0, 1)
        return c, 0

    results = {
        "val_energy": -0.99,
        "exact_energy": -1.0,
        "num_params": 0,
        "runtime_sec": 0.1,
        "training_seconds": 0.1, # Add it explicitly as well
        "actual_steps": 10,
        "final_params": np.array([]),
        "energy_history": [-0.5, -0.8, -0.9, -0.99]
    }
    
    ansatz_spec = {
        "name": "test",
        "family": "hea",
        "n_qubits": 2,
        "config": {}
    }
    
    report_path = generate_report(
        exp_dir, "TestReport", results, mock_create_circuit_fn,
        comment="Test", ansatz_spec=ansatz_spec
    )
    
    assert os.path.exists(report_path)
    assert os.path.exists(os.path.join(exp_dir, "results.jsonl"))


def test_generate_report_uccsd_uses_tensorcircuit_diagram_pipeline(tmp_path):
    exp_dir = str(tmp_path)
    spec = build_ansatz(
        ENV,
        {
            "init_state": "hf",
            "hf_qubits": [0, 1],
            "occupied_orbitals": [0, 1],
            "virtual_orbitals": [2, 3],
            "layers": 1,
            "include_singles": True,
            "include_doubles": True,
            "mapping": "jordan_wigner",
            "trotter_order": 1,
        },
    )

    results = {
        "val_energy": -7.86,
        "exact_energy": -7.862129,
        "energy_error": 0.002129,
        "num_params": spec.num_params,
        "runtime_sec": 0.1,
        "training_seconds": 0.1,
        "actual_steps": 5,
        "final_params": np.zeros(spec.num_params, dtype=np.float32),
        "energy_history": [-7.0, -7.5, -7.8, -7.85, -7.86],
        "n_qubits": ENV.n_qubits,
    }

    report_path = generate_report(
        exp_dir,
        "LiH_UCCSD_Test",
        results,
        spec.create_circuit,
        ansatz_spec=spec.to_logging_dict(),
    )

    assert os.path.exists(report_path)
    pngs = [name for name in os.listdir(exp_dir) if name.startswith("circuit_") and name.endswith(".png")]
    assert pngs
