import itertools
from functools import lru_cache
from typing import Any, Callable, Dict, Tuple

import numpy as np

from core.model.schemas import AnsatzSpec, BlockSpec, HardwareConstraintSpec, OperatorSpec


@lru_cache(maxsize=1)
def _tensorcircuit():
    import tensorcircuit as tc

    return tc


def _linear_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _ring_pairs(n: int) -> list[tuple[int, int]]:
    return _linear_pairs(n) + [(n - 1, 0)]


def _brick_pairs(n: int, layer_idx: int) -> list[tuple[int, int]]:
    offset = layer_idx % 2
    return [(i, i + 1) for i in range(offset, n - 1, 2)]


def _full_pairs(n: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(n), 2))


def get_pairs(entanglement: str, n_qubits: int, layer_idx: int = 0) -> list[tuple[int, int]]:
    if entanglement == "linear":
        return _linear_pairs(n_qubits)
    if entanglement == "ring":
        return _ring_pairs(n_qubits)
    if entanglement == "brick":
        return _brick_pairs(n_qubits, layer_idx)
    if entanglement == "full":
        return _full_pairs(n_qubits)
    raise ValueError(f"Unknown entanglement topology: {entanglement}")


def _count_two_qubit_params(gate: str) -> int:
    if gate in ("cnot", "cz"):
        return 0
    if gate == "rzz":
        return 1
    if gate == "rxx_ryy_rzz":
        return 3
    raise ValueError(f"Unknown two-qubit gate: {gate}")


def count_params(config: dict, n_qubits: int) -> int:
    layers = config.get("layers", 1)
    sq_gates = config.get("single_qubit_gates", ["ry"])
    tq_gate = config.get("two_qubit_gate", "cnot")
    entanglement = config.get("entanglement", "linear")
    param_strategy = config.get("param_strategy", "independent")

    sq_per_layer = len(sq_gates) * n_qubits
    tq_param_per_pair = _count_two_qubit_params(tq_gate)

    if entanglement == "brick":
        total_tq = 0
        for l in range(layers):
            total_tq += len(_brick_pairs(n_qubits, l)) * tq_param_per_pair
    else:
        pairs = get_pairs(entanglement, n_qubits)
        total_tq = layers * len(pairs) * tq_param_per_pair

    total_sq = layers * sq_per_layer

    if param_strategy == "translational":
        return layers * (len(sq_gates) + tq_param_per_pair)
    if param_strategy == "tied":
        if entanglement == "brick":
            pairs_0 = _brick_pairs(n_qubits, 0)
            one_layer = sq_per_layer + len(pairs_0) * tq_param_per_pair
        else:
            pairs = get_pairs(entanglement, n_qubits)
            one_layer = sq_per_layer + len(pairs) * tq_param_per_pair
        return one_layer
    return total_sq + total_tq


def build_ansatz(config: dict, n_qubits: int) -> tuple[Callable, int]:
    layers = config.get("layers", 1)
    sq_gates = config.get("single_qubit_gates", ["ry"])
    tq_gate = config.get("two_qubit_gate", "cnot")
    entanglement = config.get("entanglement", "linear")
    init_state = config.get("init_state", "zero")
    hf_qubits = config.get("hf_qubits", [])
    param_strategy = config.get("param_strategy", "independent")
    num_params = count_params(config, n_qubits)

    valid_sq = {"rx", "ry", "rz"}
    for g in sq_gates:
        if g not in valid_sq:
            raise ValueError(f"Unknown single-qubit gate: {g}. Valid: {valid_sq}")

    valid_tq = {"cnot", "cz", "rzz", "rxx_ryy_rzz"}
    if tq_gate not in valid_tq:
        raise ValueError(f"Unknown two-qubit gate: {tq_gate}. Valid: {valid_tq}")

    use_mps = config.get("use_mps", False)

    def create_circuit(params):
        tc = _tensorcircuit()
        c = tc.MPSCircuit(n_qubits) if use_mps else tc.Circuit(n_qubits)
        if init_state == "hadamard":
            for i in range(n_qubits):
                c.h(i)
        elif init_state == "hf":
            for q in hf_qubits:
                c.x(q)

        idx = 0
        for layer in range(layers):
            if param_strategy == "tied":
                idx = 0
            if param_strategy == "translational":
                layer_start_idx = idx

            for g_idx, g in enumerate(sq_gates):
                if param_strategy == "translational":
                    shared_theta = params[layer_start_idx + g_idx]
                for i in range(n_qubits):
                    gate_fn = getattr(c, g)
                    if param_strategy == "translational":
                        gate_fn(i, theta=shared_theta)
                    else:
                        gate_fn(i, theta=params[idx])
                        idx += 1

            if param_strategy == "translational":
                idx = layer_start_idx + len(sq_gates)

            pairs = get_pairs(entanglement, n_qubits, layer_idx=layer)
            for i, j in pairs:
                tq_params_used = 0
                if tq_gate == "cnot":
                    c.cnot(i, j)
                elif tq_gate == "cz":
                    c.cz(i, j)
                elif tq_gate == "rzz":
                    theta = params[idx] if param_strategy == "translational" else params[idx + tq_params_used]
                    c.rzz(i, j, theta=theta)
                    tq_params_used += 1
                elif tq_gate == "rxx_ryy_rzz":
                    theta_x = params[idx] if param_strategy == "translational" else params[idx + tq_params_used]
                    c.rxx(i, j, theta=theta_x)
                    tq_params_used += 1
                    theta_y = params[idx + 1] if param_strategy == "translational" else params[idx + tq_params_used]
                    c.ryy(i, j, theta=theta_y)
                    tq_params_used += 1
                    theta_z = params[idx + 2] if param_strategy == "translational" else params[idx + tq_params_used]
                    c.rzz(i, j, theta=theta_z)
                    tq_params_used += 1
                if param_strategy != "translational":
                    idx += tq_params_used

            if param_strategy == "translational":
                idx += _count_two_qubit_params(tq_gate)

        actual_idx = num_params if param_strategy == "tied" else idx
        return c, actual_idx

    return create_circuit, num_params


def _apply_operator(c: Any, op: OperatorSpec, params: Any, idx: int) -> int:
    tc = _tensorcircuit()
    if hasattr(tc.Circuit, op.name):
        gate_fn = getattr(c, op.name)
        gate_fn(*op.support_qubits, theta=params[idx])
        idx += 1
    return idx


def _apply_block(c: Any, block: BlockSpec, params: Any, idx: int) -> int:
    n_qubits = c._nqubits
    qubits = block.qubit_subset or list(range(n_qubits))
    if block.family == "hea":
        sq_gates = block.metadata.get("single_qubit_gates", ["ry"])
        tq_gate = block.metadata.get("two_qubit_gate", "cnot")
        entanglement = block.metadata.get("entanglement", "linear")
        for _ in range(block.repetitions):
            for q in qubits:
                for g in sq_gates:
                    getattr(c, g)(q, theta=params[idx])
                    idx += 1

            pairs = get_pairs(entanglement, len(qubits))
            abs_pairs = [(qubits[p1], qubits[p2]) for p1, p2 in pairs]
            for i, j in abs_pairs:
                if tq_gate == "cnot":
                    c.cnot(i, j)
                elif tq_gate == "cz":
                    c.cz(i, j)
                elif tq_gate == "rzz":
                    c.rzz(i, j, theta=params[idx])
                    idx += 1
    return idx


def build_circuit_from_ansatz(ansatz: AnsatzSpec) -> Tuple[Callable, int]:
    n_qubits = ansatz.n_qubits
    total_params = 0
    if ansatz.config:
        total_params += count_params(ansatz.config, n_qubits)
    for item in ansatz.blocks:
        if isinstance(item, BlockSpec):
            total_params += item.params_per_repeat * item.repetitions
        elif isinstance(item, OperatorSpec):
            total_params += 1

    use_mps = ansatz.config.get("use_mps", False) if ansatz.config else False

    def create_circuit(params):
        tc = _tensorcircuit()
        c = tc.MPSCircuit(n_qubits) if use_mps else tc.Circuit(n_qubits)
        idx = 0
        init_state = ansatz.config.get("init_state", "zero")
        if init_state == "hadamard":
            for i in range(n_qubits):
                c.h(i)
        elif init_state == "hf":
            for q in ansatz.config.get("hf_qubits", []):
                c.x(q)

        for item in ansatz.blocks:
            if isinstance(item, BlockSpec):
                idx = _apply_block(c, item, params, idx)
            elif isinstance(item, OperatorSpec):
                idx = _apply_operator(c, item, params, idx)
        return c, idx

    return create_circuit, total_params


def estimate_circuit_cost(
    ansatz: AnsatzSpec,
    hardware: HardwareConstraintSpec | None = None,
) -> Dict[str, float]:
    create_fn, num_params = build_circuit_from_ansatz(ansatz)
    c, _ = create_fn(np.zeros(num_params))
    return {
        "num_params": float(num_params),
        "two_qubit_gates": float(len([g for g in c.to_qir() if len(g["index"]) == 2])),
        "depth": 0.0,
        "hardware_legal": True if hardware is None else True,
    }
