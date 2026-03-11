import pytest
import numpy as np
from core.schemas import AnsatzSpec, BlockSpec, OperatorSpec, StructureEdit
from core.circuit_factory import (
    apply_structure_edit, build_circuit_from_ansatz, estimate_circuit_cost
)

def test_apply_structure_edit():
    ansatz = AnsatzSpec(name="test", n_qubits=2)
    
    # Append block
    edit = StructureEdit(
        edit_type="append_block",
        payload={"block": {"name": "b1", "family": "hea", "params_per_repeat": 2}}
    )
    ansatz = apply_structure_edit(ansatz, edit)
    assert len(ansatz.blocks) == 1
    assert ansatz.blocks[0].name == "b1"
    
    # Append operator
    edit = StructureEdit(
        edit_type="append_operator",
        payload={"operator": {"name": "ry", "family": "gate", "support_qubits": [0]}}
    )
    ansatz = apply_structure_edit(ansatz, edit)
    assert len(ansatz.blocks) == 2
    assert ansatz.blocks[1].name == "ry"

def test_build_circuit_from_blocks():
    block = BlockSpec(
        name="b1", family="hea", params_per_repeat=2,
        metadata={"single_qubit_gates": ["ry"], "entanglement": "linear"}
    )
    ansatz = AnsatzSpec(name="test", n_qubits=2, blocks=[block])
    
    create_fn, num_params = build_circuit_from_ansatz(ansatz)
    assert num_params == 2 # 2 qubits * 1 (ry)
    
    params = np.array([0.1, 0.2])
    c, idx = create_fn(params)
    assert idx == 2
    assert c._nqubits == 2

def test_estimate_circuit_cost():
    block = BlockSpec(
        name="b1", family="hea", params_per_repeat=2,
        metadata={"single_qubit_gates": ["ry"], "entanglement": "linear"}
    )
    ansatz = AnsatzSpec(name="test", n_qubits=2, blocks=[block])
    
    cost = estimate_circuit_cost(ansatz)
    assert cost["num_params"] == 2
    assert "two_qubit_gates" in cost
    assert cost["depth"] >= 0
