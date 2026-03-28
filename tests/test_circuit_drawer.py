import os

from core.circuit_drawer import compact_gate_columns, render_circuit_diagram


def test_compact_gate_columns_reuses_free_columns():
    circuit = [
        {"name": "x", "qubits": [0]},
        {"name": "x", "qubits": [1]},
        {"name": "cx", "qubits": [0, 1]},
        {"name": "rz", "qubits": [2]},
    ]

    placed = compact_gate_columns(circuit)
    assert placed[0].column == 0
    assert placed[1].column == 0
    assert placed[2].column == 1
    assert placed[3].column == 0


def test_render_circuit_diagram_writes_png(tmp_path):
    output_path = tmp_path / "circuit.png"
    circuit = [
        {"name": "x", "qubits": [0]},
        {"name": "h", "qubits": [1]},
        {"name": "exp1", "qubits": [0, 1, 2]},
        {"name": "rz", "qubits": [2]},
    ]

    render_circuit_diagram(circuit, str(output_path), title="Test Circuit", dpi=120)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
