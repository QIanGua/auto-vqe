import pytest
import torch

from core.evaluator.logging_utils import setup_logger
from core.evaluator.training import optimize_parameters, vqe_train


def test_vqe_train_legacy_wrapper(tmp_path):
    def create_circuit_fn(params):
        class Dummy:
            def __init__(self, n):
                self._nqubits = n

            def to_json(self):
                return []

        return Dummy(2), 2

    def compute_energy_fn(params):
        return params[0] ** 2 + params[1] ** 2 - 1.0

    res = vqe_train(
        create_circuit_fn=create_circuit_fn,
        compute_energy_fn=compute_energy_fn,
        n_qubits=2,
        num_params=2,
        max_steps=50,
        exact_energy=-1.0,
        lr=0.1,
    )
    assert "val_energy" in res
    assert "energy_error" in res
    assert isinstance(res["val_energy"], float)
    assert isinstance(res["final_params"], torch.Tensor)


def test_optimize_parameters_edge_cases():
    with pytest.raises(ValueError, match="Either ansatz or create_circuit_fn must be provided"):
        optimize_parameters(env=None, ansatz=None, create_circuit_fn=None)

    def mock_compute_energy(params):
        return torch.tensor(0.0, requires_grad=True)

    def mock_create(params):
        return None, 0

    res = optimize_parameters(
        env=None,
        compute_energy_fn=mock_compute_energy,
        create_circuit_fn=mock_create,
        optimizer_spec=None,
    )
    assert res["actual_steps"] >= 0


def test_setup_logger_creates_logger(tmp_path):
    log_path = tmp_path / "test.log"
    logger = setup_logger(str(log_path))
    logger.info("hello")
    assert log_path.exists()
