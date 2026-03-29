import os

import numpy as np
import pytest

from baselines.uccsd import build_ansatz
from core.evaluator.report import generate_report
from experiments.lih.env import ENV


@pytest.mark.slow
def test_generate_report_minimal(tmp_path):
    exp_dir = str(tmp_path)

    def mock_create_circuit_fn(params):
        import tensorcircuit as tc

        c = tc.Circuit(2)
        c.rz(0, theta=0.1)
        c.cx(0, 1)
        return c, 0

    results = {
        "val_energy": -0.99,
        "exact_energy": -1.0,
        "num_params": 0,
        "runtime_sec": 0.1,
        "training_seconds": 0.1,
        "actual_steps": 10,
        "final_params": np.array([]),
        "energy_history": [-0.5, -0.8, -0.9, -0.99],
    }

    ansatz_spec = {
        "name": "test",
        "family": "hea",
        "n_qubits": 2,
        "config": {},
    }

    record_path = generate_report(
        exp_dir,
        "TestReport",
        results,
        mock_create_circuit_fn,
        comment="Test",
        ansatz_spec=ansatz_spec,
    )

    assert os.path.exists(record_path)
    assert os.path.exists(os.path.join(exp_dir, "run.json"))
    assert os.path.exists(os.path.join(exp_dir, "events.jsonl"))
    pngs = [name for name in os.listdir(exp_dir) if name.startswith("circuit_") and name.endswith(".png")]
    assert not pngs


@pytest.mark.slow
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

    record_path = generate_report(
        exp_dir,
        "LiH_UCCSD_Test",
        results,
        spec.create_circuit,
        ansatz_spec=spec.to_logging_dict(),
        render_markdown=True,
        render_assets=True,
    )

    assert os.path.exists(record_path)
    pngs = [name for name in os.listdir(exp_dir) if name.startswith("circuit_") and name.endswith(".png")]
    assert pngs
