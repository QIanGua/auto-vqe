import pytest
import numpy as np
from core.model.schemas import AnsatzSpec, CandidateSpec, EvaluationSpec
from core.evaluator.api import evaluate_candidate, promote_candidate
from core.evaluator.training import optimize_parameters
from core.foundation.base_env import QuantumEnvironment

class MockEnv:
    def __init__(self, n_qubits=2):
        self.n_qubits = n_qubits
        self.exact_energy = -1.0
    def compute_energy(self, circuit):
        # Return an energy that depends on the parameters to avoid grad_fn error
        # In a real scenario, the circuit would be built from params.
        # Here we just assume the first 2 elements of some global or implicit state.
        # But wait, compute_energy takes a circuit, not params.
        # The create_circuit_fn inside optimize_parameters uses params to build circuit.
        # However, tc.Circuit doesn't store params.
        # So we mock a dependency by returning a tensor that we pretend is from circuit.
        import torch
        # Hack: return a value that looks like it depends on params if we could trace it.
        # For testing purposes, we can just return a value but ensure it's not a leaf if needed?
        # Actually, the error is because 'energy' returned by compute_energy doesn't have a grad_fn.
        # We need to make it depend on something that HAS a grad.
        # Since we don't have the params here, we can't easily do it.
        # Better: change the mock to return a value that uses a dummy variable with grad.
        return torch.tensor(-1.0, requires_grad=True)

def test_optimize_parameters():
    env = MockEnv()
    from core.model.schemas import OptimizerSpec
    opt_spec = OptimizerSpec(max_steps=10)

    def mock_create_circuit(params):
        return object(), len(params)

    def mock_compute_energy(params):
        import torch

        return params.sum() * 0 + torch.tensor(-1.0, requires_grad=True)

    results = optimize_parameters(
        env,
        ansatz=None,
        optimizer_spec=opt_spec,
        create_circuit_fn=mock_create_circuit,
        compute_energy_fn=mock_compute_energy,
        num_params=2,
    )
    assert "val_energy" in results
    assert results["actual_steps"] <= 10
    assert len(results["final_params"]) > 0

def test_evaluate_candidate(monkeypatch):
    env = MockEnv()
    ansatz = AnsatzSpec(name="test", n_qubits=2, config={"layers": 1})
    candidate = CandidateSpec(candidate_id="c1", ansatz=ansatz, proposed_by="test")
    eval_spec = EvaluationSpec(fidelity="quick", max_steps=5)

    monkeypatch.setattr(
        "core.evaluator.training.optimize_parameters",
        lambda **kwargs: {
            "val_energy": -1.0,
            "num_params": 2,
            "actual_steps": 3,
        },
    )
    monkeypatch.setattr(
        "core.representation.compiler.estimate_circuit_cost",
        lambda ansatz: {"two_qubit_gates": 0},
    )

    result = evaluate_candidate(env, candidate, eval_spec)
    assert result.candidate_id == "c1"
    assert result.fidelity == "quick"
    assert result.actual_steps <= 5

def test_promote_candidate():
    # Mock a previous result
    from core.model.schemas import EvaluationResult
    prev = EvaluationResult(
        candidate_id="c1", fidelity="quick", success=True, 
        val_energy=-0.8, num_params=2, two_qubit_gates=1, 
        runtime_sec=0.1, actual_steps=5
    )
    
    next_spec = promote_candidate(prev, "medium")
    assert next_spec.fidelity == "medium"
    assert next_spec.max_steps > 5
