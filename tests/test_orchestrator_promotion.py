from core.model.schemas import CandidateSpec, EvaluationResult, AnsatzSpec
from core.orchestration.controller import SearchOrchestrator

def test_orchestrator_promotion():
    ansatz = AnsatzSpec(name="a1", n_qubits=2)
    cand1 = CandidateSpec(candidate_id="c1", ansatz=ansatz, proposed_by="test")
    cand2 = CandidateSpec(candidate_id="c2", ansatz=ansatz, proposed_by="test")
    
    orchestrator = SearchOrchestrator(generators=[])
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
    
    promotions = orchestrator.promote([res1, res2])
    assert len(promotions) == 1
    assert promotions[0].candidate.candidate_id == "c1"
    assert promotions[0].next_evaluation.fidelity == "medium"

    # `promote()` should no longer mutate the queue by itself.
    initial_batch = orchestrator.schedule_next_batch()
    assert len(initial_batch) == 2
    assert all(item[1].fidelity == "quick" for item in initial_batch)

    scheduled = orchestrator.enqueue_promotions(promotions)
    assert len(scheduled) == 1
    assert scheduled[0][0].candidate_id == "c1"
    assert scheduled[0][1].fidelity == "medium"

    promoted_batch = orchestrator.schedule_next_batch()
    assert len(promoted_batch) == 1
    assert promoted_batch[0][0].candidate_id == "c1"
    assert promoted_batch[0][1].fidelity == "medium"
