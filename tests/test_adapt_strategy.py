import pytest
import numpy as np
from core.model.schemas import (
    AnsatzSpec, OperatorSpec, OperatorPoolSpec, 
    StrategyCheckpoint, EvaluationResult
)
from core.generator.adapt import AdaptVQEStrategy

class MockEnv:
    def __init__(self):
        self.n_qubits = 2

def test_adapt_vqe_propose():
    env = MockEnv()
    op_pool = OperatorPoolSpec(
        name="pauli_pool",
        operators=[
            OperatorSpec(name="ry", family="gate", support_qubits=[0]),
            OperatorSpec(name="rz", family="gate", support_qubits=[0])
        ]
    )
    strategy = AdaptVQEStrategy(env, op_pool)
    state = strategy.initialize()
    
    candidates = strategy.propose(state, budget=2)
    assert len(candidates) == 2
    assert candidates[0].ansatz.blocks[0].name == "ry"
    assert candidates[1].ansatz.blocks[0].name == "rz"

def test_adapt_vqe_update():
    env = MockEnv()
    op_pool = OperatorPoolSpec(name="pool", operators=[])
    strategy = AdaptVQEStrategy(env, op_pool)
    state = strategy.initialize()
    
    res = EvaluationResult(
        candidate_id="c1", fidelity="quick", success=True, 
        val_energy=-0.5, num_params=1, two_qubit_gates=0, 
        runtime_sec=0.1, actual_steps=5
    )
    
    new_state = strategy.update(state, [res])
    assert new_state.step_count == 1
    assert new_state.best_score == -0.5
    assert new_state.best_candidate_id == "c1"
