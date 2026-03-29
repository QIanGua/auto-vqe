from core.model.schemas import AnsatzSpec, BlockSpec, OperatorSpec, StructureEdit


def apply_structure_edit(ansatz: AnsatzSpec, edit: StructureEdit) -> AnsatzSpec:
    new_ansatz = ansatz.model_copy(deep=True)

    if edit.edit_type == "append_block":
        block_data = edit.payload.get("block")
        if block_data is None:
            return new_ansatz
        if isinstance(block_data, dict):
            block = BlockSpec(**block_data) if "params_per_repeat" in block_data else OperatorSpec(**block_data)
        else:
            block = block_data
        new_ansatz.blocks.append(block)

    elif edit.edit_type == "append_operator":
        op_data = edit.payload.get("operator")
        if op_data is None:
            return new_ansatz
        op = OperatorSpec(**op_data) if isinstance(op_data, dict) else op_data
        new_ansatz.blocks.append(op)

    elif edit.edit_type == "remove_block":
        idx = edit.payload.get("index")
        if isinstance(idx, int) and 0 <= idx < len(new_ansatz.blocks):
            new_ansatz.blocks.pop(idx)

    elif edit.edit_type == "replace_block":
        idx = edit.payload.get("index")
        block_data = edit.payload.get("block")
        if isinstance(idx, int) and 0 <= idx < len(new_ansatz.blocks):
            if block_data is None:
                return new_ansatz
            if isinstance(block_data, dict):
                block = BlockSpec(**block_data) if "params_per_repeat" in block_data else OperatorSpec(**block_data)
            else:
                block = block_data
            new_ansatz.blocks[idx] = block

    elif edit.edit_type == "expand_qubit_subset":
        new_n = edit.payload.get("n_qubits")
        if new_n and new_n > new_ansatz.n_qubits:
            new_ansatz.n_qubits = new_n

    new_ansatz.growth_history.append(edit)
    return new_ansatz
