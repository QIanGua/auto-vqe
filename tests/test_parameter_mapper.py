import pytest
import numpy as np
from core.schemas import AnsatzSpec, BlockSpec, OperatorSpec
from core.parameter_mapper import ParameterMapper

def test_parameter_mapper_identity():
    mapper = ParameterMapper()
    block = BlockSpec(name="b1", family="hea", params_per_repeat=2)
    ansatz = AnsatzSpec(name="a1", n_qubits=2, blocks=[block])
    
    old_params = np.array([0.1, 0.2])
    plan = mapper.build_plan(ansatz, ansatz, old_params)
    
    assert plan.old_param_count == 2
    assert plan.new_param_count == 2
    assert len(plan.reused_indices) == 2
    
    new_params = mapper.apply_plan(plan, old_params)
    assert np.allclose(new_params, old_params)

def test_parameter_mapper_append():
    mapper = ParameterMapper()
    b1 = BlockSpec(name="b1", family="hea", params_per_repeat=2)
    b2 = BlockSpec(name="b2", family="hea", params_per_repeat=1)
    
    old_ansatz = AnsatzSpec(name="a1", n_qubits=2, blocks=[b1])
    new_ansatz = AnsatzSpec(name="a2", n_qubits=2, blocks=[b1, b2])
    
    old_params = np.array([0.1, 0.2])
    plan = mapper.build_plan(old_ansatz, new_ansatz, old_params, init_strategy="zeros")
    
    assert plan.new_param_count == 3
    assert (0, 0) in plan.reused_indices
    assert (1, 1) in plan.reused_indices
    assert 2 in plan.initialized_indices
    
    new_params = mapper.apply_plan(plan, old_params)
    assert new_params[0] == 0.1
    assert new_params[1] == 0.2
    assert new_params[2] == 0.0

def test_parameter_mapper_mismatch():
    mapper = ParameterMapper()
    b1 = BlockSpec(name="b1", family="hea", params_per_repeat=2)
    b2 = BlockSpec(name="b2", family="hea", params_per_repeat=3) # Changed
    
    old_ansatz = AnsatzSpec(name="a1", n_qubits=2, blocks=[b1])
    new_ansatz = AnsatzSpec(name="a2", n_qubits=2, blocks=[b2])
    
    old_params = np.array([0.1, 0.2])
    plan = mapper.build_plan(old_ansatz, new_ansatz, old_params)
    
    # Should not reuse if first block changed size
    assert len(plan.reused_indices) == 0
    assert plan.new_param_count == 3
    assert len(plan.initialized_indices) == 3
