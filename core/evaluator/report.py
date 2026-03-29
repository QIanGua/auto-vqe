import datetime
import hashlib
import json
import os
import platform
import subprocess
import uuid
from typing import Any, Dict, Optional

import torch

from core.evaluator.logging_utils import append_event_jsonl
from core.representation.compiler import _brick_pairs, get_pairs


def _json_ready_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_ready_value(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            return _json_ready_value(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_ready_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready_value(item) for item in value]
    return value


def _infer_system_from_exp_dir(exp_dir: str) -> str:
    parts = os.path.abspath(exp_dir).split(os.sep)
    if "experiments" in parts:
        idx = parts.index("experiments")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def _infer_system_dir(exp_dir: str) -> Optional[str]:
    parts = os.path.abspath(exp_dir).split(os.sep)
    if "experiments" not in parts:
        return None
    idx = parts.index("experiments")
    if idx + 1 >= len(parts):
        return None
    return os.sep.join(parts[: idx + 2])


def _find_git_root(start_dir: str) -> Optional[str]:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _get_git_info(repo_root: Optional[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"commit": None, "dirty": False, "diff_hash": None, "diff": None}
    if repo_root is None:
        return info
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        info["commit"] = sha

        diff_text = subprocess.check_output(
            ["git", "-C", repo_root, "diff"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
        if diff_text.strip():
            info["dirty"] = True
            info["diff"] = diff_text
            info["diff_hash"] = hashlib.md5(diff_text.encode()).hexdigest()
        else:
            info["diff"] = ""
    except Exception:
        pass
    return info


def _save_git_diff_artifact(exp_dir: str, diff_text: str) -> str:
    audit_dir = os.path.join(exp_dir, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    patch_path = os.path.join(audit_dir, "git_diff.patch")
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(diff_text)
    return os.path.join("audit", "git_diff.patch")


def _save_report_context(exp_dir: str, results: Dict[str, Any]) -> str:
    context_path = os.path.join(exp_dir, "report_context.json")
    payload = {
        "final_params": _json_ready_value(results.get("final_params")),
        "energy_history": _json_ready_value(results.get("energy_history", [])),
    }
    with open(context_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return context_path


def load_report_context(exp_dir: str) -> Dict[str, Any]:
    context_path = os.path.join(exp_dir, "report_context.json")
    if not os.path.exists(context_path):
        return {}
    with open(context_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _get_runtime_env() -> Dict[str, Any]:
    import tensorcircuit as tc

    return {
        "python_version": platform.python_version(),
        "os": platform.system(),
        "libraries": {
            "tensorcircuit": tc.__version__ if hasattr(tc, "__version__") else "unknown",
            "torch": torch.__version__,
        },
    }


def _estimate_two_qubit_gates(
    ansatz_spec: Optional[Dict[str, Any]],
    n_qubits: Optional[int],
) -> Optional[int]:
    if ansatz_spec is None or n_qubits is None:
        return None

    cfg = ansatz_spec["config"] if isinstance(ansatz_spec.get("config"), dict) else ansatz_spec
    layers = int(cfg.get("layers", 1))
    entanglement = cfg.get("entanglement", "linear")

    if entanglement == "brick":
        return sum(len(_brick_pairs(n_qubits, layer_idx)) for layer_idx in range(layers))
    return layers * len(get_pairs(entanglement, n_qubits))


def _save_tensorcircuit_circuit_png(circuit: Any, output_path: str) -> None:
    from core.rendering.circuit_drawer import render_circuit_diagram

    raw_data = circuit.to_json()
    circuit_data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    render_circuit_diagram(circuit_data, output_path=output_path, simplify=True)


def generate_report(
    exp_dir,
    exp_name,
    results,
    create_circuit_fn,
    comment: str = "",
    ansatz_spec: Optional[Dict[str, Any]] = None,
    decision: str = "keep",
    parent_experiment: Optional[str] = None,
    change_summary: str = "",
    config_path: Optional[str] = None,
    render_markdown: bool = False,
    render_assets: bool = False,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_record_path = os.path.join(exp_dir, "run.json")
    report_path = os.path.join(exp_dir, f"report_{timestamp}.md")
    config_snapshot_path = os.path.join(exp_dir, "config_snapshot.json")
    circuit_img_path = os.path.join(exp_dir, f"circuit_{timestamp}.png")
    convergence_img_path = os.path.join(exp_dir, f"convergence_{timestamp}.png")
    circuit_json_path = os.path.join(exp_dir, f"circuit_{timestamp}.json")

    has_image = False
    has_convergence = False

    if render_assets:
        try:
            circuit, _ = create_circuit_fn(results["final_params"])
            _save_tensorcircuit_circuit_png(circuit, circuit_img_path)
            has_image = True
        except Exception as e:
            print(f"Image generation failed: {e}")

        energy_history = results.get("energy_history", [])
        if energy_history:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(energy_history, linewidth=0.8, color="#2196F3")
                ax.axhline(
                    y=results["exact_energy"],
                    color="#F44336",
                    linestyle="--",
                    linewidth=1,
                    label=f"Exact: {results['exact_energy']:.6f}",
                )
                ax.set_xlabel("Optimization Step")
                ax.set_ylabel("Energy")
                ax.set_title(f"{exp_name} Convergence")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(convergence_img_path, dpi=200)
                plt.close(fig)
                has_convergence = True
            except Exception:
                pass

        try:
            circuit, _ = create_circuit_fn(results["final_params"])
            raw_data = circuit.to_json()
            circuit_data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            with open(circuit_json_path, "w", encoding="utf-8") as f:
                json.dump(circuit_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    system = _infer_system_from_exp_dir(exp_dir)
    system_dir = _infer_system_dir(exp_dir)
    git_root = _find_git_root(exp_dir)
    git_info = _get_git_info(git_root)
    runtime_env = _get_runtime_env()

    if git_info.get("diff"):
        diff_path = _save_git_diff_artifact(exp_dir, git_info["diff"])
        git_info["diff_path"] = diff_path
        del git_info["diff"]

    accuracy_status = "探索中"
    if results.get("energy_error") is not None:
        if results["energy_error"] < 0.0016:
            accuracy_status = "成功 (达到化学精度)"
        if results["energy_error"] < 1e-5:
            accuracy_status = "完美收敛 (高精度)"

    metrics: Dict[str, Any] = {
        "val_energy": results.get("val_energy"),
        "exact_energy": results.get("exact_energy"),
        "energy_error": results.get("energy_error"),
        "num_params": results.get("num_params"),
        "depth": None,
        "two_qubit_gates": _estimate_two_qubit_gates(ansatz_spec, results.get("n_qubits")),
        "runtime_sec": results.get("training_seconds", results.get("runtime_sec")),
        "actual_steps": results.get("actual_steps"),
    }
    artifact_paths: Dict[str, Any] = {
        "run_json": run_record_path,
        "report_md": report_path if render_markdown else None,
        "circuit_png": circuit_img_path if has_image else None,
        "convergence_png": convergence_img_path if has_convergence else None,
        "circuit_json": circuit_json_path if render_assets else None,
        "report_context": _save_report_context(exp_dir, results),
    }

    if ansatz_spec is not None:
        with open(config_snapshot_path, "w", encoding="utf-8") as f:
            json.dump(ansatz_spec, f, ensure_ascii=False, indent=2)
        artifact_paths["config_snapshot"] = config_snapshot_path

    experiment_record: Dict[str, Any] = {
        "schema_version": "1.2",
        "experiment_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system": system,
        "exp_name": exp_name,
        "seed": results.get("seed"),
        "n_qubits": results.get("n_qubits"),
        "ansatz_spec": ansatz_spec,
        "optimizer_spec": results.get("optimizer_spec"),
        "measurement_spec": {
            "observable": "Hamiltonian",
            "n_qubits": results.get("n_qubits"),
            "exact_energy": results.get("exact_energy"),
        },
        "metrics": metrics,
        "decision": decision,
        "parent_experiment": parent_experiment,
        "change_summary": change_summary,
        "comment": comment,
        "config_path_used": config_path,
        "git_info": {
            "commit": git_info["commit"],
            "dirty": git_info["dirty"],
            "diff_hash": git_info["diff_hash"],
        },
        "runtime_env": runtime_env,
        "artifact_paths": artifact_paths,
        "status": accuracy_status,
        "run_dir": exp_dir,
    }

    if git_info.get("diff_path"):
        experiment_record["git_info"]["diff_path"] = git_info["diff_path"]

    with open(run_record_path, "w", encoding="utf-8") as f:
        json.dump(experiment_record, f, ensure_ascii=False, indent=2)

    append_event_jsonl(
        exp_dir,
        {
            "schema_version": "1.2",
            "kind": "final_record",
            "exp_name": exp_name,
            "run_json": run_record_path,
            "metrics": metrics,
            "config_path_used": config_path,
        },
    )

    if system_dir is not None:
        index_path = os.path.join(system_dir, "artifacts", "index.jsonl")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(experiment_record, ensure_ascii=False) + "\n")

    if render_markdown:
        actual_steps = results.get("actual_steps", "N/A")
        img_embed = (
            f"![Circuit Diagram]({os.path.basename(circuit_img_path)})\n"
            if has_image
            else "*默认未生成线路图；如需可视化请显式开启 render_assets。*\n"
        )
        convergence_embed = (
            f"![Convergence Curve]({os.path.basename(convergence_img_path)})\n"
            if has_convergence
            else ""
        )
        report_content = f"""# 实验结题报告: {exp_name}
- **日期**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **后端**: TensorCircuit (PyTorch)

## 一、 核心指标
| 指标 | 数值 |
| :--- | :--- |
| **最终能量** | {results['val_energy']:.8f} |
| **精确能量** | {results.get('exact_energy') if results.get('exact_energy') is not None else 'N/A'} |
| **能量误差** | {results.get('energy_error') if results.get('energy_error') is not None else 'N/A'} |
| **参数量** | {results['num_params']} |
| **实际步数** | {actual_steps} |
| **耗时** | {results.get('training_seconds', results.get('runtime_sec', 'N/A'))} s |
| **状态** | {accuracy_status} |

## 二、 收敛曲线
{convergence_embed}
## 三、 线路可视化图示
{img_embed}

## 四、 结果分析
{comment if comment else "自动生成的分析报告：实验已完成。"}

## 五、 审计信息
- **配置路径**: `{config_path or "N/A"}`
- **代码版本**: `{(git_info.get("commit") or "unknown")[:8] if git_info.get("commit") else "unknown"}`
- **环境指纹**: `Python {runtime_env.get("python_version", "N/A")}`
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

    return run_record_path
