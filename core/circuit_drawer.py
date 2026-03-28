from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import numpy as np


@dataclass
class PlacedGate:
    name: str
    qubits: list[int]
    column: int
    kind: str = "gate"
    label: str | None = None
    theta: float | None = None
    group: int | None = None


def load_circuit_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Circuit JSON must be a list, got {type(data).__name__}")
    return data


def infer_n_qubits(circuit_data: Sequence[dict[str, Any]]) -> int:
    max_qubit = -1
    for gate in circuit_data:
        for qubit in gate.get("qubits", []):
            max_qubit = max(max_qubit, int(qubit))
    return max_qubit + 1 if max_qubit >= 0 else 1


def compact_gate_columns(circuit_data: Sequence[dict[str, Any]]) -> list[PlacedGate]:
    column_usage: list[set[int]] = []
    placed: list[PlacedGate] = []

    for gate in circuit_data:
        qubits = sorted(int(q) for q in gate.get("qubits", []))
        if not qubits:
            continue

        support = set(range(min(qubits), max(qubits) + 1))
        explicit_column = gate.get("column")
        if explicit_column is not None:
            column = int(explicit_column)
            while len(column_usage) <= column:
                column_usage.append(set())
            column_usage[column].update(support)
        else:
            column = 0
            while column < len(column_usage) and column_usage[column] & support:
                column += 1
            if column == len(column_usage):
                column_usage.append(set())
            column_usage[column].update(support)
        placed.append(
            PlacedGate(
                name=str(gate.get("name", "?")),
                qubits=qubits,
                column=column,
                kind=str(gate.get("kind", "gate")),
                label=gate.get("label"),
                theta=gate.get("theta"),
                group=gate.get("group"),
            )
        )

    return placed


def _gate_fill(name: str) -> str:
    lname = name.lower()
    if lname == "ry":
        return "#A61B5B"
    if lname == "rz":
        return "#3FA9F5"
    if lname in {"x", "y", "z"}:
        return "#D7E5F7"
    if lname in {"h", "s", "sdg", "t"}:
        return "#F3F5F7"
    if lname.startswith("r"):
        return "#D9F5EC"
    if lname in {"cx", "cnot", "cy", "cz"}:
        return "#E6E0FF"
    if lname.startswith("exp"):
        return "#FFE9C7"
    if lname in {"measure", "reset"}:
        return "#FFD9D6"
    return "#F1F3F5"


def _format_theta(theta: float | None) -> str:
    if theta is None:
        return ""
    abs_theta = abs(theta)
    if abs_theta >= 1:
        return f"{theta:.2f}"
    if abs_theta >= 1e-2:
        return f"{theta:.3f}"
    return f"{theta:.2e}"


def _pretty_gate_label(name: str, default: str | None = None) -> str:
    if default:
        return default
    lname = name.lower()
    mapping = {
        "ry": r"$\mathrm{R_Y}$",
        "rz": r"$\mathrm{R_Z}$",
        "rx": r"$\mathrm{R_X}$",
        "h": "H",
        "s": "S",
        "sdg": "Sd",
        "x": "X",
        "y": "Y",
        "z": "Z",
    }
    return mapping.get(lname, name.upper())


def _candidate_pauli_tensor(paulis: Sequence[str]) -> np.ndarray:
    mats = {
        "I": np.eye(2, dtype=np.complex64),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex64),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex64),
    }
    out = np.array([[1.0 + 0.0j]], dtype=np.complex64)
    for p in paulis:
        out = np.kron(out, mats[p])
    return out


def _json_complex_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim >= 1 and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    if arr.ndim >= 1 and arr.shape[0] == 2:
        return arr[0] + 1j * arr[1]
    return arr.astype(np.complex64)


def _extract_pauli_string(gate: dict[str, Any]) -> list[str] | None:
    params = gate.get("parameters")
    qubits = gate.get("qubits", [])
    if gate.get("name") != "exp1" or not isinstance(params, dict):
        return None

    n = len(qubits)
    if n == 0:
        return None

    matrix_raw = gate.get("matrix")
    if matrix_raw is None:
        return None
    unitary = _json_complex_array(matrix_raw).reshape(2**n, 2**n)
    theta_raw = params.get("theta")
    if theta_raw is None:
        return None
    theta = float(np.asarray(theta_raw, dtype=np.float32).reshape(-1)[0])
    if abs(np.sin(theta)) < 1e-12:
        return None
    # exp1 gate matrix is exp(-i theta P), so recover P from the gate itself.
    unitary = (np.cos(theta) * np.eye(2**n) - unitary) / (1j * np.sin(theta))
    alphabet = ("I", "X", "Y", "Z")
    best: list[str] | None = None
    for code in np.ndindex(*(4 for _ in range(n))):
        paulis = [alphabet[i] for i in code]
        candidate = _candidate_pauli_tensor(paulis).reshape(2**n, 2**n)
        if np.allclose(unitary, candidate, atol=1e-5):
            best = paulis
            break
        if np.allclose(unitary, -candidate, atol=1e-5):
            best = paulis
            break
    return best


def expand_circuit_for_display(circuit_data: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    column_cursor = 0
    group_cursor = 0

    def emit_stage(stage_gates: Sequence[dict[str, Any]], advance: bool = True) -> None:
        nonlocal column_cursor
        if not stage_gates:
            return
        for stage_gate in stage_gates:
            expanded.append({**stage_gate, "column": column_cursor, "group": group_cursor})
        if advance:
            column_cursor += 1

    for gate in circuit_data:
        name = str(gate.get("name", "?")).lower()
        qubits = [int(q) for q in gate.get("qubits", [])]

        if name != "exp1":
            emit_stage([dict(gate)], advance=True)
            column_cursor += 1
            group_cursor += 1
            continue

        paulis = _extract_pauli_string(gate)
        params = gate.get("parameters", {})
        theta_raw = params.get("theta")
        theta = None
        if theta_raw is not None:
            theta_arr = np.asarray(theta_raw, dtype=np.float32)
            theta = float(theta_arr.reshape(-1)[0])

        if not paulis or len(paulis) != len(qubits):
            expanded.append(dict(gate))
            continue

        active_pairs = [(q, p) for q, p in zip(qubits, paulis) if p != "I"]
        if len(active_pairs) <= 1:
            q = active_pairs[0][0] if active_pairs else qubits[0]
            emit_stage(
                [
                    {
                        "name": "rz",
                        "qubits": [q],
                        "kind": "rotation",
                        "label": None,
                        "theta": 2 * theta if theta is not None else None,
                    }
                ]
            )
            column_cursor += 1
            group_cursor += 1
            continue

        active_qubits = [q for q, _ in active_pairs]

        stage = []
        for q, p in active_pairs:
            if p == "X":
                stage.append({"name": "h", "qubits": [q], "kind": "basis", "label": "H"})
            elif p == "Y":
                stage.append({"name": "sdg", "qubits": [q], "kind": "basis", "label": "Sd"})
        emit_stage(stage)

        stage = []
        for q, p in active_pairs:
            if p == "Y":
                stage.append({"name": "h", "qubits": [q], "kind": "basis", "label": "H"})
        emit_stage(stage)

        for ctrl, tgt in zip(active_qubits[:-1], active_qubits[1:]):
            emit_stage([{"name": "cx", "qubits": [ctrl, tgt], "kind": "cx", "label": "CX"}])

        emit_stage(
            [
                {
                    "name": "rz",
                    "qubits": [active_qubits[-1]],
                    "kind": "rotation",
                    "label": None,
                    "theta": 2 * theta if theta is not None else None,
                }
            ]
        )

        for ctrl, tgt in reversed(list(zip(active_qubits[:-1], active_qubits[1:]))):
            emit_stage([{"name": "cx", "qubits": [ctrl, tgt], "kind": "cx", "label": "CX"}])

        stage = []
        for q, p in reversed(active_pairs):
            if p == "X":
                stage.append({"name": "h", "qubits": [q], "kind": "basis", "label": "H"})
            elif p == "Y":
                stage.append({"name": "h", "qubits": [q], "kind": "basis", "label": "H"})
        emit_stage(stage)

        stage = []
        for q, p in reversed(active_pairs):
            if p == "Y":
                stage.append({"name": "s", "qubits": [q], "kind": "basis", "label": "S"})
        emit_stage(stage)

        column_cursor += 1
        group_cursor += 1

    return expanded


def _draw_gate(
    ax: Any,
    gate: PlacedGate,
    x: float,
    y_positions: Sequence[float],
    box_w: float,
    simplify: bool = False,
) -> None:
    from matplotlib.patches import FancyBboxPatch, Circle

    name = gate.name
    qubits = gate.qubits
    ys = [y_positions[q] for q in qubits]
    y_min = min(ys)
    y_max = max(ys)
    fill = _gate_fill(name)
    label = _pretty_gate_label(name, gate.label)
    is_rotation = gate.kind == "rotation"
    is_basis = gate.kind == "basis" or name.lower() in {"h", "s", "sdg"}
    gate_box_w = max(box_w if is_rotation else 0.52, min(1.0, 0.16 * len(str(label)) + 0.14))
    weak_basis = simplify and is_basis

    if gate.kind == "cx" and len(qubits) == 2:
        ctrl_y = y_positions[qubits[0]]
        tgt_y = y_positions[qubits[1]]
        ax.plot([x, x], [min(ctrl_y, tgt_y), max(ctrl_y, tgt_y)], color="#0F3FB2", linewidth=2.0, zorder=2)
        ax.add_patch(Circle((x, ctrl_y), radius=0.06, facecolor="#0F3FB2", edgecolor="none", zorder=4))
        ax.add_patch(Circle((x, tgt_y), radius=0.16, facecolor="#0F3FB2", edgecolor="#0F3FB2", zorder=3))
        ax.plot([x - 0.09, x + 0.09], [tgt_y, tgt_y], color="white", linewidth=2.0, zorder=4)
        ax.plot([x, x], [tgt_y - 0.09, tgt_y + 0.09], color="white", linewidth=2.0, zorder=4)
        return

    if len(qubits) == 1:
        y = ys[0]
        patch = FancyBboxPatch(
            (x - gate_box_w / 2, y - 0.28),
            gate_box_w * (0.86 if weak_basis else 1.0),
            0.64 if is_rotation else (0.34 if weak_basis else 0.46),
            boxstyle="round,pad=0.02,rounding_size=0.01" if is_rotation else "round,pad=0.02,rounding_size=0.04",
            linewidth=0.7 if weak_basis else 1.0,
            edgecolor="#AEB8C5" if weak_basis else "#4A5568",
            facecolor="#F8FAFC" if weak_basis else fill,
            zorder=3,
        )
        ax.add_patch(patch)
        text_color = "white" if gate.name.lower() == "ry" else "#111827"
        if is_rotation:
            theta_text = _format_theta(gate.theta)
            ax.text(x, y + 0.10, label, ha="center", va="center", fontsize=10.5, color=text_color, zorder=4)
            if theta_text:
                ax.text(x, y - 0.12, theta_text, ha="center", va="center", fontsize=7.8, color=text_color, zorder=4)
        else:
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=6.8 if weak_basis else (7.8 if is_basis else 8.4),
                color="#94A3B8" if weak_basis else "#1F2937",
                zorder=4,
            )
        return

    ax.plot([x, x], [y_min, y_max], color="#6B7280", linewidth=1.3, zorder=2)

    gate_h = max(0.58, (y_max - y_min) + 0.42)
    patch = FancyBboxPatch(
        (x - gate_box_w / 2, (y_min + y_max) / 2 - gate_h / 2),
        gate_box_w,
        gate_h,
        boxstyle="round,pad=0.03,rounding_size=0.09",
        linewidth=1.2,
        edgecolor="#4A5568",
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(patch)

    if len(qubits) >= 4 and len(str(label)) > 5:
        label = f"{label}\n{len(qubits)}q"
    ax.text(
        x,
        (y_min + y_max) / 2,
        label,
        ha="center",
        va="center",
        fontsize=7.4,
        color="#1F2937",
        zorder=4,
    )

    for y in ys:
        ax.add_patch(Circle((x, y), radius=0.045, facecolor="#4A5568", edgecolor="none", zorder=4))


def render_circuit_diagram(
    circuit_data: Sequence[dict[str, Any]],
    output_path: str,
    title: str | None = None,
    dpi: int = 220,
    max_columns_per_panel: int = 16,
    simplify: bool = False,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_data = expand_circuit_for_display(circuit_data)
    n_qubits = infer_n_qubits(display_data)
    placed = compact_gate_columns(display_data)
    n_columns = max((gate.column for gate in placed), default=-1) + 1
    panel_count = max(1, math.ceil(max(1, n_columns) / max_columns_per_panel))

    left_pad = 1.1
    right_pad = 0.8
    column_pitch = 1.18
    box_w = 0.82

    panel_columns = min(max_columns_per_panel, max(1, n_columns))
    width = max(11.5, left_pad + right_pad + panel_columns * column_pitch)
    panel_height = max(3.0, 1.02 * n_qubits + 1.15)
    title_extra = 0.55 if title else 0.0
    height = panel_count * panel_height + title_extra

    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FCFCFD")

    panel_gap = 0.65
    panel_span = panel_height + panel_gap
    base_y_positions = [n_qubits - 1 - i for i in range(n_qubits)]
    x_end = left_pad + panel_columns * column_pitch

    for panel_idx in range(panel_count):
        panel_top_offset = (panel_count - 1 - panel_idx) * panel_span
        y_positions = [y + panel_top_offset for y in base_y_positions]

        for qubit, y in enumerate(y_positions):
            ax.plot([left_pad - 0.3, x_end], [y, y], color="#111111", linewidth=1.6, zorder=1)
            ax.text(
                left_pad - 0.45,
                y,
                rf"$q_{qubit}$",
                ha="right",
                va="center",
                fontsize=17,
                color="#111111",
            )

        start_col = panel_idx * max_columns_per_panel
        end_col = start_col + max_columns_per_panel
        panel_gates = [gate for gate in placed if start_col <= gate.column < end_col]

        first_gate_by_group: dict[int, int] = {}
        for gate in panel_gates:
            if gate.group is None:
                continue
            first_gate_by_group.setdefault(gate.group, gate.column)

        for _, gate_col in sorted(first_gate_by_group.items(), key=lambda item: item[1]):
            local_col = gate_col - start_col
            if local_col <= 0:
                continue
            x_sep = left_pad + local_col * column_pitch - column_pitch * 0.58
            ax.plot(
                [x_sep, x_sep],
                [min(y_positions) - 0.18, max(y_positions) + 0.18],
                color="#E5E7EB",
                linewidth=0.9,
                zorder=0,
            )

        for gate in panel_gates:
            x = left_pad + (gate.column - start_col) * column_pitch
            _draw_gate(ax, gate, x, y_positions, box_w, simplify=simplify)

    if title:
        ax.text(
            left_pad - 0.3,
            panel_count * panel_span - panel_gap + 0.18,
            title,
            ha="left",
            va="bottom",
            fontsize=13,
            color="#0F172A",
            weight="semibold",
        )

    ax.set_xlim(0, x_end + 0.2)
    ax.set_ylim(-0.6, panel_count * panel_span - panel_gap + 0.8)
    ax.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def render_circuit_json_file(
    input_path: str,
    output_path: str | None = None,
    title: str | None = None,
    dpi: int = 220,
    simplify: bool = False,
) -> str:
    circuit_data = load_circuit_json(input_path)
    if output_path is None:
        stem, _ = os.path.splitext(input_path)
        output_path = stem + ".png"
    return render_circuit_diagram(
        circuit_data,
        output_path=output_path,
        title=title,
        dpi=dpi,
        simplify=simplify,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a circuit JSON file into a clean PNG diagram.")
    parser.add_argument("input", help="Path to a circuit_*.json file")
    parser.add_argument("output", nargs="?", help="Output PNG path; defaults to alongside the input")
    parser.add_argument("--title", default=None, help="Optional diagram title")
    parser.add_argument("--dpi", type=int, default=220, help="PNG DPI")
    parser.add_argument("--simplify", action="store_true", help="De-emphasize auxiliary basis-change gates like H/S/Sd")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_path = render_circuit_json_file(
        args.input,
        output_path=args.output,
        title=args.title,
        dpi=args.dpi,
        simplify=args.simplify,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
