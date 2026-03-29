from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from core.molecular.schema import PauliTerm


def get_exact_from_paulis(
    paulis: Sequence[Tuple[float, Sequence[Tuple[str, int]]]] | Sequence[PauliTerm],
    n_qubits: int,
) -> float:
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.array([[1, 0], [0, 1]], dtype=complex)

    def tensor_op(ops_list: Iterable[Tuple[str, int]]) -> np.ndarray:
        result = 1
        current_ops = {idx: op for op, idx in ops_list}
        for i in range(n_qubits):
            op_type = current_ops.get(i, "i")
            if op_type == "x":
                op = x
            elif op_type == "y":
                op = y
            elif op_type == "z":
                op = z
            else:
                op = identity
            result = np.kron(result, op)
        return result

    hamiltonian = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for term in paulis:
        if isinstance(term, PauliTerm):
            coeff = float(term.coeff)
            ops = term.ops
        else:
            coeff, ops = term
        if not ops:
            hamiltonian += coeff * np.eye(2**n_qubits)
        else:
            hamiltonian += coeff * tensor_op(ops)
    return float(np.min(np.linalg.eigvalsh(hamiltonian)).real)
