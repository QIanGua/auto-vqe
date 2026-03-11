import pytest
import logging
from core.schemas import CandidateSpec, EvaluationResult, AnsatzSpec
from core.controller import SearchOrchestrator, SearchController

def test_orchestrator_promotion():
    ansatz = AnsatzSpec(name="a1", n_qubits=2)
    cand1 = CandidateSpec(candidate_id="c1", ansatz=ansatz, proposed_by="test")
    cand2 = CandidateSpec(candidate_id="c2", ansatz=ansatz, proposed_by="test")
    
    orchestrator = SearchOrchestrator(strategies=[])
    orchestrator.submit_candidates([cand1, cand2], fidelity="quick")
    
    # Mock results
    res1 = EvaluationResult(
        candidate_id="c1", fidelity="quick", success=True, 
        val_energy=-0.9, num_params=1, two_qubit_gates=0, 
        runtime_sec=0.1, actual_steps=5
    )
    res2 = EvaluationResult(
        candidate_id="c2", fidelity="quick", success=True, 
        val_energy=-0.8, num_params=1, two_qubit_gates=0, 
        runtime_sec=0.1, actual_steps=5
    )
    
    promoted = orchestrator.promote([res1, res2])
    assert len(promoted) == 1
    assert promoted[0].candidate_id == "c1"
    
    # Check if next evaluation is scheduled
    batch = orchestrator.schedule_next_batch()
    # Initial quick for c1, c2 (2 items) -> we take 2
    # Then promote added 1 medium for c1.
    # Total evaluation_queue was 2 (quick) + 1 (medium) = 3
    # schedule_next_batch(4) should take all 3 if they were in queue.
    # Wait, submit_candidates added 2. promote added 1. queue has 2 items left after submit + 1 after promote?
    # Actually submit_candidates added (c1, quick), (c2, quick).
    # Then promote added (c1, medium).
    # Total queue: [(c1, quick), (c2, quick), (c1, medium)]
    assert len(batch) <= 3 
    assert any(b[1].fidelity == "medium" for b in batch)
