import pytest
import numpy as np
import torch
from core.model.schemas import AnsatzSpec, BlockSpec, OperatorSpec, StructureEdit
from core.representation.compiler import build_circuit_from_ansatz, estimate_circuit_cost
from core.representation.compiler import build_ansatz
from core.representation.edits import apply_structure_edit

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

    # Replace block
    edit = StructureEdit(
        edit_type="replace_block",
        payload={"index": 0, "block": {"name": "b2", "family": "hea", "params_per_repeat": 4}}
    )
    ansatz = apply_structure_edit(ansatz, edit)
    assert ansatz.blocks[0].name == "b2"

    # Remove block
    edit = StructureEdit(edit_type="remove_block", payload={"index": 1})
    ansatz = apply_structure_edit(ansatz, edit)
    assert len(ansatz.blocks) == 1
    assert ansatz.blocks[0].name == "b2"

    # Expand qubit subset
    edit = StructureEdit(edit_type="expand_qubit_subset", payload={"n_qubits": 4})
    ansatz = apply_structure_edit(ansatz, edit)
    assert ansatz.n_qubits == 4

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

def test_build_circuit_hea_cz():
    block = BlockSpec(
        name="b1", family="hea", repetitions=1, params_per_repeat=2,
        metadata={"single_qubit_gates": ["ry"], "two_qubit_gate": "cz", "entanglement": "linear"}
    )
    ansatz = AnsatzSpec(name="test", n_qubits=2, blocks=[block])
    create_fn, num_params = build_circuit_from_ansatz(ansatz)
    params = np.array([0.1, 0.2])
    c, _ = create_fn(params)
    # No direct way to check gate type easily in tc without deeper inspection, 
    # but ensuring it runs is good.
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

def test_estimate_circuit_cost_advanced_gates():
    # Test with other gates
    ansatz_ext = AnsatzSpec(
        name="ext_gates",
        n_qubits=2,
        blocks=[
            OperatorSpec(name="rxx", family="gate", support_qubits=[0, 1]),
            OperatorSpec(name="ryy", family="gate", support_qubits=[0, 1]),
            OperatorSpec(name="rzz", family="gate", support_qubits=[0, 1]),
        ]
    )
    cost_ext = estimate_circuit_cost(ansatz_ext)
    assert cost_ext["num_params"] == 3
    assert cost_ext["two_qubit_gates"] == 3
    assert cost_ext["depth"] >= 0


def test_build_ansatz_accepts_torch_params_with_grad():
    config = {
        "layers": 1,
        "single_qubit_gates": ["rx", "ry"],
        "two_qubit_gate": "cnot",
        "entanglement": "linear",
    }

    create_fn, num_params = build_ansatz(config, 2)
    params = torch.randn(num_params, requires_grad=True)

    circuit, used_params = create_fn(params)

    assert circuit._nqubits == 2
    assert used_params == num_params


def test_build_circuit_supports_pauli_exp_operator():
    ansatz = AnsatzSpec(
        name="pauli_exp_test",
        n_qubits=2,
        blocks=[
            OperatorSpec(
                name="Y0_X1",
                family="pauli_exp",
                support_qubits=[0, 1],
                generator="Y0 X1",
                metadata={"paulis": ["Y", "X"]},
            )
        ],
    )
    create_fn, num_params = build_circuit_from_ansatz(ansatz)
    assert num_params == 1
    c, used = create_fn(np.array([0.1], dtype=np.float32))
    assert c._nqubits == 2
    assert used == 1


def test_build_circuit_supports_excitation_operator():
    ansatz = AnsatzSpec(
        name="excitation_test",
        n_qubits=4,
        config={"init_state": "hf", "hf_qubits": [0, 1]},
        blocks=[
            OperatorSpec(
                name="s_0->2",
                family="excitation",
                support_qubits=[0, 1, 2],
                generator="s_0->2",
                metadata={
                    "trotter_terms": [
                        {"support_qubits": [0, 1, 2], "paulis": ["Y", "Z", "X"], "coeff_imag": 0.5},
                        {"support_qubits": [0, 1, 2], "paulis": ["X", "Z", "Y"], "coeff_imag": -0.5},
                    ]
                },
            )
        ],
    )
    create_fn, num_params = build_circuit_from_ansatz(ansatz)
    assert num_params == 1
    c, used = create_fn(np.array([0.2], dtype=np.float32))
    assert c._nqubits == 4
    assert used == 1
