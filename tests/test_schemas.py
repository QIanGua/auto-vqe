import pytest
from pydantic import ValidationError
from core.schemas import (
    AnsatzSpec, OptimizerSpec, SchedulerSpec, SearchSpaceSpec,
    BlockSpec, OperatorSpec, OperatorPoolSpec, StructureEdit,
    WarmStartPlan, EvaluationSpec, EvaluationResult, CandidateSpec,
    StrategyCheckpoint, HardwareConstraintSpec
)

def test_ansatz_spec_valid():
    config = {
        "layers": 2,
        "single_qubit_gates": ["ry"],
        "two_qubit_gate": "cnot",
        "entanglement": "linear"
    }
    spec = AnsatzSpec(name="test_vqe", n_qubits=4, config=config)
    assert spec.name == "test_vqe"
    assert spec.n_qubits == 4
    assert spec.config["layers"] == 2

def test_ansatz_spec_invalid_qubits():
    config = {"layers": 2}
    with pytest.raises(ValidationError):
        AnsatzSpec(name="test", n_qubits=0, config=config)
    with pytest.raises(ValidationError):
        AnsatzSpec(name="test", n_qubits=-1, config=config)

def test_optimizer_spec_defaults():
    spec = OptimizerSpec()
    assert spec.method == "Adam"
    assert spec.lr == 0.01
    assert spec.max_steps == 500
    assert spec.scheduler is not None
    assert spec.scheduler.type == "ReduceLROnPlateau"

def test_optimizer_spec_custom():
    spec = OptimizerSpec(lr=0.05, max_steps=1000, tol=1e-10)
    assert spec.lr == 0.05
    assert spec.max_steps == 1000
    assert spec.tol == 1e-10

def test_optimizer_spec_invalid():
    with pytest.raises(ValidationError):
        OptimizerSpec(lr=0)
    with pytest.raises(ValidationError):
        OptimizerSpec(max_steps=0)

def test_scheduler_spec():
    spec = SchedulerSpec(patience=10, factor=0.1)
    assert spec.patience == 10
    assert spec.factor == 0.1

def test_search_space_spec_defaults():
    spec = SearchSpaceSpec()
    assert 1 in spec.layers
    assert "linear" in spec.entanglement
    assert ["ry"] in spec.single_qubit_gates_options

def test_search_space_spec_custom():
    spec = SearchSpaceSpec(layers=[10, 20], entanglement=["full"])
    assert spec.layers == [10, 20]
    assert spec.entanglement == ["full"]

def test_block_spec():
    block = BlockSpec(
        name="test_block",
        family="hea",
        params_per_repeat=4
    )
    assert block.name == "test_block"
    assert block.repetitions == 1

def test_operator_spec():
    op = OperatorSpec(
        name="test_op",
        family="pauli",
        support_qubits=[0, 1],
        generator="Z0Z1"
    )
    assert op.name == "test_op"
    assert op.hardware_legal is True

def test_structure_edit():
    edit = StructureEdit(
        edit_type="append_block",
        payload={"block_name": "b1"}
    )
    assert edit.edit_type == "append_block"

def test_evaluation_spec():
    spec = EvaluationSpec(fidelity="quick", max_steps=20)
    assert spec.fidelity == "quick"
    assert spec.max_steps == 20

def test_ansatz_with_blocks():
    block = BlockSpec(name="b1", family="hea", params_per_repeat=2)
    ansatz = AnsatzSpec(name="a1", n_qubits=2, blocks=[block])
    assert len(ansatz.blocks) == 1
    assert ansatz.blocks[0].name == "b1"

def test_candidate_spec():
    ansatz = AnsatzSpec(name="a1", n_qubits=2)
    candidate = CandidateSpec(candidate_id="c1", ansatz=ansatz, proposed_by="tester")
    assert candidate.candidate_id == "c1"
    assert candidate.proposed_by == "tester"
