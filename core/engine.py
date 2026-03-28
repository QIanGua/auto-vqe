import torch
import time
import os
import datetime
import logging
import warnings
import contextlib
import io
import uuid
import json
import subprocess
import numpy as np
from typing import Any, Dict, Optional, Literal, Callable

# Import tensorcircuit once, but silence its optional-backend warnings and
# Python SyntaxWarning noise that may appear on import under newer Python.
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import tensorcircuit as tc

# Some tensorcircuit submodules trigger SyntaxWarning under Python 3.12;
# they are harmless for our usage, so we ignore them to keep logs clean.
warnings.filterwarnings("ignore", category=SyntaxWarning, module="tensorcircuit.*")

tc.set_backend("pytorch")


from contextlib import contextmanager
from core.schemas import (
    OptimizerSpec, AnsatzSpec, CandidateSpec, EvaluationSpec, 
    EvaluationResult, WarmStartPlan, StructureEdit
)
import numpy as np


# ---------------------------------------------------------------------------
# Helpers for structured experiment logging
# ---------------------------------------------------------------------------

def _infer_system_from_exp_dir(exp_dir: str) -> str:
    """
    Try to infer the physical / problem system name from the experiment path.
    For this repo we assume pattern: .../experiments/<system>/...
    """
    parts = os.path.abspath(exp_dir).split(os.sep)
    if "experiments" in parts:
        idx = parts.index("experiments")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def _find_git_root(start_dir: str) -> Optional[str]:
    """
    Walk upwards from start_dir until we find a .git directory, or return None.
    """
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _get_git_info(repo_root: Optional[str]) -> Dict[str, Any]:
    """
    Capture git commit SHA and dirty status/diff.
    """
    info: Dict[str, Any] = {"commit": None, "dirty": False, "diff_hash": None, "diff": None}
    if repo_root is None:
        return info
    try:
        # Get current commit SHA
        sha = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        info["commit"] = sha

        # Get diff
        out = subprocess.check_output(
            ["git", "-C", repo_root, "diff"],
            stderr=subprocess.DEVNULL,
        )
        diff_text = out.decode("utf-8", errors="replace")
        if diff_text.strip():
            info["dirty"] = True
            info["diff"] = diff_text
            # Store a short hash of the diff to avoid massive redundancy in some analyses
            import hashlib
            info["diff_hash"] = hashlib.md5(diff_text.encode()).hexdigest()
        else:
            info["diff"] = ""
    except Exception:
        pass
    return info


def _save_git_diff_artifact(exp_dir: str, diff_text: str) -> str:
    """
    Save git diff to a separate patch file and return the relative path.
    """
    audit_dir = os.path.join(exp_dir, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    patch_name = "git_diff.patch"
    patch_path = os.path.join(audit_dir, patch_name)
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(diff_text)
    return os.path.join("audit", patch_name)


def _get_runtime_env() -> Dict[str, Any]:
    """
    Capture Python and library versions.
    """
    import platform
    env = {
        "python_version": platform.python_version(),
        "os": platform.system(),
        "libraries": {
            "tensorcircuit": tc.__version__ if hasattr(tc, "__version__") else "unknown",
            "torch": torch.__version__,
        }
    }
    return env


def prepare_experiment_dir(base_dir: str, exp_name: str) -> str:
    """
    为实验创建一个独立的、带时间戳的文件夹。
    命名规范: YYYYMMDD_HHMMSS_exp_name
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{exp_name}"
    session_root = os.environ.get("AGENT_VQE_SESSION_DIR")
    if session_root:
        exp_base_dir = session_root
    else:
        exp_base_dir = base_dir
    exp_path = os.path.join(exp_base_dir, folder_name)
    os.makedirs(exp_path, exist_ok=True)
    return exp_path


def _append_experiment_jsonl(exp_dir: str, record: Dict[str, Any]) -> None:
    """
    Append a single structured experiment record into results.jsonl
    under the given experiment directory. Keeps TSV logs alongside
    a machine-readable event stream.
    """
    path = os.path.join(exp_dir, "results.jsonl")
    # Use utf-8 so that comments / summaries can contain non-ASCII if needed.
    with open(path, "a", encoding="utf-8") as f:
        # Update schema version to 1.2 reflecting diff_path
        if "schema_version" not in record:
            record["schema_version"] = "1.2"
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def _estimate_two_qubit_gates(
    ansatz_spec: Optional[Dict[str, Any]],
    n_qubits: Optional[int],
) -> Optional[int]:
    """
    Roughly estimate the number of two-qubit gates from the ansatz spec
    and qubit count. This uses the same topology helpers as circuit_factory
    so it stays consistent with how circuits are actually built.

    支持两种 ansatz_spec 形式：
      1) 旧版: 直接是 config dict，包含 "layers" / "entanglement"；
      2) 标准化版: `AnsatzSpec.to_logging_dict()` 的结果，其中实际
         config 嵌套在 ansatz_spec["config"] 下。
    """
    if ansatz_spec is None or n_qubits is None:
        return None

    # 兼容旧版与新版结构：优先从 "config" 中取实际 ansatz 配置。
    cfg: Dict[str, Any]
    if "config" in ansatz_spec and isinstance(ansatz_spec["config"], dict):
        cfg = ansatz_spec["config"]  # 来自 AnsatzSpec.to_logging_dict()
    else:
        cfg = ansatz_spec

    try:
        # Local import to avoid any potential import-order surprises.
        from core.circuit_factory import get_pairs  # type: ignore
    except Exception:
        return None

    layers = int(cfg.get("layers", 1))
    entanglement = cfg.get("entanglement", "linear")

    total = 0
    if entanglement == "brick":
        # brick has layer-dependent pairs
        from core.circuit_factory import _brick_pairs  # type: ignore

        for l in range(layers):
            total += len(_brick_pairs(n_qubits, l))
    else:
        pairs = get_pairs(entanglement, n_qubits)
        total = layers * len(pairs)
    return total


def _save_tensorcircuit_circuit_png(circuit: Any, output_path: str) -> None:
    """
    Save a tensorcircuit circuit to PNG using the repo's lightweight drawer.
    """
    import json

    from core.circuit_drawer import render_circuit_diagram

    raw_data = circuit.to_json()
    if isinstance(raw_data, str):
        circuit_data = json.loads(raw_data)
    else:
        circuit_data = raw_data

    render_circuit_diagram(circuit_data, output_path=output_path, simplify=True)

@contextmanager
def experiment_guard(run_py_path, logger=None):
    """
    实验守卫：自动备份 run.py，若实验过程抛出异常则自动回退。
    用于保护 Agent 对 create_circuit 的修改不会因崩溃而丢失原始代码。
    """
    backup = open(run_py_path, encoding="utf-8").read()
    try:
        yield
    except Exception as e:
        with open(run_py_path, "w", encoding="utf-8") as f:
            f.write(backup)
        msg = f"Experiment failed, reverted {run_py_path}. Error: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        raise

def setup_logger(log_path):
    """
    配置标准的日志系统，确保实时落盘且不为空
    """
    logger = logging.getLogger("VQE_Engine")
    logger.setLevel(logging.INFO)
    
    # 清除旧的 handler
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 文件 Handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    
    # 控制台 Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def optimize_parameters(
    env: Any,
    ansatz: Optional[AnsatzSpec] = None,
    optimizer_spec: Optional[OptimizerSpec] = None,
    init_params: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None,
    seed: Optional[int] = None,
    # 额外参数支持 legacy vqe_train
    create_circuit_fn: Optional[Callable] = None,
    compute_energy_fn: Optional[Callable] = None,
    num_params: Optional[int] = None,
) -> Dict[str, Any]:
    """
    针对给定的 AnsatzSpec 或自定义电路工厂进行参数优化。
    """
    if optimizer_spec is None:
        optimizer_spec = OptimizerSpec()

    if create_circuit_fn is None:
        if ansatz is None:
            raise ValueError("Either ansatz or create_circuit_fn must be provided.")
        from core.circuit_factory import build_circuit_from_ansatz
        create_circuit_fn, num_params = build_circuit_from_ansatz(ansatz)
    
    if compute_energy_fn is None:
        def compute_energy_fn(params):
            c, _ = create_circuit_fn(params)
            return env.compute_energy(c)
    
    if num_params is None:
        # 尝试推导或者使用 0
        num_params = 0
    
    # 转换为 torch 初始参数
    if init_params is not None:
        params_tensor = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)
    else:
        if seed is not None:
            torch.manual_seed(seed)
        params_tensor = torch.randn(num_params, requires_grad=True)

    max_steps = optimizer_spec.max_steps
    lr = optimizer_spec.lr
    early_stop_window = optimizer_spec.early_stop_window
    early_stop_threshold = optimizer_spec.early_stop_threshold
    grad_clip_norm = optimizer_spec.grad_clip_norm

    if optimizer_spec.method == "Adam":
        optimizer = torch.optim.Adam([params_tensor], lr=lr)
    else:
        optimizer = torch.optim.Adam([params_tensor], lr=lr)

    sched_spec = optimizer_spec.scheduler
    if sched_spec and sched_spec.type == "ReduceLROnPlateau":
        mode: Any = sched_spec.mode
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, patience=sched_spec.patience, 
            factor=sched_spec.factor, min_lr=sched_spec.min_lr
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)

    start_time = time.time()
    energy_history = []
    actual_steps = max_steps

    for i in range(max_steps):
        optimizer.zero_grad()
        energy = compute_energy_fn(params_tensor)
        energy.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([params_tensor], max_norm=grad_clip_norm)
        optimizer.step()

        e_val = energy.item()
        energy_history.append(e_val)
        scheduler.step(e_val)

        if i % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            msg = f"Step {i}, Energy: {e_val:.6f}, LR: {current_lr:.2e}"
            if logger: logger.info(msg)

        if len(energy_history) >= early_stop_window:
            recent = energy_history[-early_stop_window:]
            if max(recent) - min(recent) < early_stop_threshold:
                actual_steps = i + 1
                break

    end_time = time.time()
    final_energy = compute_energy_fn(params_tensor).item()
    
    return {
        "val_energy": final_energy,
        "exact_energy": env.exact_energy if hasattr(env, "exact_energy") else None,
        "energy_error": abs(final_energy - env.exact_energy) if hasattr(env, "exact_energy") else None,
        "num_params": num_params,
        "runtime_sec": end_time - start_time,
        "actual_steps": actual_steps,
        "final_params": params_tensor.detach().numpy(),
        "energy_history": energy_history,
    }

def evaluate_candidate(
    env: Any,
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
    warm_start: Optional[WarmStartPlan] = None,
    logger: Optional[logging.Logger] = None,
) -> EvaluationResult:
    """
    多保真评估包装器。
    """
    start_time = time.time()
    
    # 1. 准备初始参数 (Warm-start)
    init_params = None
    if warm_start and candidate.warm_start_from:
        # 这里需要从存储或上下文获取旧参数，MVP 暂假设外部已处理并注入 init_params
        pass
    
    # 2. 映射 EvaluationSpec 到 OptimizerSpec
    opt_spec = OptimizerSpec(
        method=evaluation.optimizer_name,
        max_steps=evaluation.max_steps,
        early_stop_threshold=1e-8 if evaluation.enable_early_stop else 1e-12
    )
    
    # 3. 执行优化
    # TODO: 处理 n_seeds
    results = optimize_parameters(
        env=env,
        ansatz=candidate.ansatz,
        optimizer_spec=opt_spec,
        init_params=init_params,
        logger=logger
    )
    
    # 4. 构建结果
    from core.circuit_factory import estimate_circuit_cost
    cost = estimate_circuit_cost(candidate.ansatz)
    
    return EvaluationResult(
        candidate_id=candidate.candidate_id,
        fidelity=evaluation.fidelity,
        success=True,
        val_energy=results["val_energy"],
        energy_error=abs(results["val_energy"] - env.exact_energy) if hasattr(env, "exact_energy") else None,
        num_params=results["num_params"],
        two_qubit_gates=int(cost["two_qubit_gates"]),
        runtime_sec=time.time() - start_time,
        actual_steps=results["actual_steps"],
        artifacts={}
    )

def promote_candidate(
    previous: EvaluationResult,
    next_fidelity: Literal["medium", "full"],
) -> EvaluationSpec:
    """
    根据上一阶段评估结果，生成下一阶段的评估配置。
    """
    if next_fidelity == "medium":
        return EvaluationSpec(fidelity="medium", max_steps=150, n_seeds=2)
    else:
        return EvaluationSpec(fidelity="full", max_steps=500, n_seeds=3)

def vqe_train(
    create_circuit_fn,
    compute_energy_fn,
    n_qubits,
    exact_energy=None,
    num_params=0,
    max_steps=1000,
    lr=0.01,
    logger=None,
    seed=None,
    early_stop_window=50,
    early_stop_threshold=1e-8,
    grad_clip_norm=1.0,
    optimizer_spec_obj: Optional["OptimizerSpec"] = None,
):
    """
    Legacy vqe_train 兼容层。
    """
    if optimizer_spec_obj is None:
        optimizer_spec_obj = OptimizerSpec(
            lr=lr, max_steps=max_steps, 
            early_stop_window=early_stop_window, 
            early_stop_threshold=early_stop_threshold, 
            grad_clip_norm=grad_clip_norm
        )
    # 或者让 optimize_parameters 接受 create_circuit_fn / compute_energy_fn。
    
    # --- 原始逻辑开始 ---
    if seed is not None:
        torch.manual_seed(seed)

    params = torch.randn(num_params, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=optimizer_spec_obj.lr)
    
    start_time = time.time()
    energy_history = []
    actual_steps = optimizer_spec_obj.max_steps

    for i in range(optimizer_spec_obj.max_steps):
        optimizer.zero_grad()
        energy = compute_energy_fn(params)
        energy.backward()
        if optimizer_spec_obj.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([params], max_norm=optimizer_spec_obj.grad_clip_norm)
        optimizer.step()
        e_val = energy.item()
        energy_history.append(e_val)
        if i % 100 == 0:
            if logger: logger.info(f"Step {i}, Energy: {e_val:.6f}")
        if len(energy_history) >= optimizer_spec_obj.early_stop_window:
            recent = energy_history[-optimizer_spec_obj.early_stop_window:]
            if max(recent) - min(recent) < optimizer_spec_obj.early_stop_threshold:
                actual_steps = i + 1
                break
    
    final_energy = compute_energy_fn(params).item()
    return {
        "val_energy": final_energy,
        "exact_energy": exact_energy,
        "energy_error": abs(final_energy - exact_energy) if exact_energy is not None else None,
        "num_params": num_params,
        "training_seconds": time.time() - start_time,
        "actual_steps": actual_steps,
        "final_params": params.detach(),
        "energy_history": energy_history,
    }

def print_results(results, logger=None):
    output = "\n--- VQE Results ---\n"
    skip_keys = {"final_params", "energy_history"}
    for k, v in results.items():
        if k not in skip_keys:
            output += f"{k:<18}: {v}\n"
    output += "-------------------\n"
    if logger: logger.info(output)
    else: print(output)

def log_results(exp_dir, exp_name, results, comment="", global_dir=None):
    """
    记录实验结果。
    同时在实验目录下记录 results.tsv，如果提供 global_dir，则在全局目录也记录一份。
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = "timestamp\texp_name\tval_energy\tenergy_error\tnum_params\tactual_steps\ttraining_sec\tcomment\n"
    actual_steps = results.get('actual_steps', 'N/A')
    line = f"{ts}\t{exp_name}\t{results['val_energy']:.6f}\t{results['energy_error']:.6f}\t{results['num_params']}\t{actual_steps}\t{results['training_seconds']}\t{comment}\n"

    # 1. 记录到当前实验目录
    log_path = os.path.join(exp_dir, "results.tsv")
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a") as f:
        if not file_exists:
            f.write(header)
        f.write(line)

    # 2. 记录到全局汇总目录 (索引) - 可选
    if global_dir and global_dir != exp_dir:
        global_path = os.path.join(global_dir, "results.tsv")
        global_exists = os.path.isfile(global_path)
        with open(global_path, "a") as f:
            if not global_exists:
                f.write(header)
            f.write(line)

def summarize_config(config):
    """
    将 ansatz 配置压缩成一行易读字符串，便于写入日志/tsv。
    """
    if isinstance(config, dict):
        parts = [f"{k}={v}" for k, v in sorted(config.items())]
        return ", ".join(parts)
    return str(config)

from core.strategy_base import SearchStrategy

class GridSearchStrategy(SearchStrategy):
    """
    网格搜索策略。封装已有的 ansatz_search 函数。
    """
    def __init__(
        self,
        env,
        make_create_circuit_fn,
        config_list,
        exp_dir,
        base_exp_name,
        lr=0.01,
        trials_per_config=3,
        max_steps=1000,
        sub_dir: str | None = None,
        logger: logging.Logger | None = None,
        controller: Optional["SearchController"] = None,
    ):
        super().__init__(env, controller, logger, name=base_exp_name)
        self.make_create_circuit_fn = make_create_circuit_fn
        self.config_list = config_list
        self.exp_dir = exp_dir
        self.base_exp_name = base_exp_name
        self.lr = lr
        self.trials_per_config = trials_per_config
        self.max_steps = max_steps
        self.sub_dir = sub_dir

    def run(self) -> dict:
        return ansatz_search(
            env=self.env,
            make_create_circuit_fn=self.make_create_circuit_fn,
            config_list=self.config_list,
            exp_dir=self.exp_dir,
            base_exp_name=self.base_exp_name,
            lr=self.lr,
            trials_per_config=self.trials_per_config,
            max_steps=self.max_steps,
            sub_dir=self.sub_dir,
            logger=self.logger,
            controller=self.controller,
        )
from core.controller import SearchController

def ansatz_search(
    env,
    make_create_circuit_fn,
    config_list,
    exp_dir,
    base_exp_name,
    lr=0.01,
    trials_per_config=3,
    max_steps=1000,
    sub_dir: str | None = None,
    logger: logging.Logger | None = None,
    controller: Optional[SearchController] = None,
):
    """
    在给定的 ansatz 配置列表上进行自动搜索。

    - env: QuantumEnvironment 子类实例，提供 n_qubits, exact_energy, compute_energy(c)。
    - make_create_circuit_fn(config) -> (create_circuit_fn, num_params)  或
      返回一个带 `create_circuit` / `num_params` / `to_logging_dict()` 的
      对象（例如 Baseline Zoo 中的 `AnsatzSpec`）。
    - config_list: 离散的 ansatz 配置列表（dict 或任意可打印对象）。
    - controller: 可选的 SearchController 实例，用于管理预算和停止规则。

    Ranking 规则：
      1. 先按 val_energy 从小到大排序；
      2. 若能量差异在 1e-4 以内，则按 num_params 从少到多排序（奥卡姆剃刀）。
    """
    if sub_dir:
        exp_dir = os.path.join(exp_dir, sub_dir)
        os.makedirs(exp_dir, exist_ok=True)

    if logger is None:
        # 默认在搜索级别只用一个简单的 logger
        log_path = os.path.join(exp_dir, f"{base_exp_name}_search_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logger = setup_logger(log_path)

    if controller is None:
        controller = SearchController(logger=logger)

    logger.info(f"=== Ansatz Search: {base_exp_name} ===")
    logger.info(f"Target Energy: {env.exact_energy:.6f}")

    best_overall: Optional[Dict[str, Any]] = None
    best_overall_config: Any = None
    best_overall_spec: Optional[Dict[str, Any]] = None

    for idx, config in enumerate(config_list):
        if not controller.should_continue():
            logger.info(f"Search interrupted by controller: {controller.stop_reason}")
            break

        cfg_str = summarize_config(config)
        logger.info(f"\n--- Config {idx+1}/{len(config_list)}: {cfg_str} ---")

        ansatz_obj = make_create_circuit_fn(config)
        # 支持两种返回形式：旧版 (create_circuit_fn, num_params) 或
        # Baseline Zoo 风格的 AnsatzSpec。
        if isinstance(ansatz_obj, tuple) and len(ansatz_obj) == 2:
            create_circuit_fn, num_params = ansatz_obj
            # 为了在 results.jsonl 中统一，构造一份轻量级 ansatz_spec 字典。
            if isinstance(config, dict):
                cfg_dict: Dict[str, Any] = dict(config)
            else:
                cfg_dict = {"raw_config": str(config)}
            ansatz_spec_dict = {
                "name": base_exp_name,
                "family": "multidim",
                "env_name": getattr(env, "name", "unknown"),
                "n_qubits": getattr(env, "n_qubits", None),
                "num_params": num_params,
                "config": cfg_dict,
                "metadata": {
                    "search": "multidim",
                },
            }
        else:
            create_circuit_fn = getattr(ansatz_obj, "create_circuit")
            num_params = int(getattr(ansatz_obj, "num_params"))
            if hasattr(ansatz_obj, "to_logging_dict"):
                ansatz_spec_dict = ansatz_obj.to_logging_dict()  # type: ignore[attr-defined]
            else:
                if isinstance(config, dict):
                    cfg_dict = dict(config)
                else:
                    cfg_dict = {"raw_config": str(config)}
                ansatz_spec_dict = {
                    "name": base_exp_name,
                    "family": "multidim",
                    "env_name": getattr(env, "name", "unknown"),
                    "n_qubits": getattr(env, "n_qubits", None),
                    "num_params": num_params,
                    "config": cfg_dict,
                    "metadata": {
                        "search": "multidim",
                    },
                }

        def compute_energy_fn(params):
            c, _ = create_circuit_fn(params)
            return env.compute_energy(c)

        best_for_cfg = None
        
        # Internal failure counter for this specific proposal
        proposal_failures = 0

        for t in range(trials_per_config):
            if not controller.should_continue():
                break

            seed = 1000 + idx * 100 + t
            logger.info(f"  Trial {t+1}/{trials_per_config}, seed={seed}")

            try:
                # 训练内环不直接使用 exact_energy，只返回能量本身
                results = vqe_train(
                    create_circuit_fn=create_circuit_fn,
                    compute_energy_fn=compute_energy_fn,
                    n_qubits=env.n_qubits,
                    exact_energy=None,
                    num_params=num_params,
                    max_steps=max_steps,
                    lr=lr,
                    logger=logger,
                    seed=seed,
                )
                
                controller.report_result(results)

                if (best_for_cfg is None) or (results["val_energy"] < best_for_cfg["val_energy"]):
                    best_for_cfg = results

            except Exception as e:
                proposal_failures += 1
                logger.error(f"  Trial {t+1} failed: {e}")
                controller.report_result({}, is_failure=True)
                if proposal_failures >= 3:
                    logger.warning(f"Too many failures for this proposal ({proposal_failures}), skipping.")
                    break

        if best_for_cfg is not None:
            # 在配置级别（外层）再使用 exact_energy 做评估，而不是在训练内环中
            best_for_cfg["exact_energy"] = env.exact_energy
            best_for_cfg["energy_error"] = abs(
                best_for_cfg["val_energy"] - env.exact_energy
            )
            comment = f"config: {cfg_str}"
            log_results(exp_dir, f"{base_exp_name}_cfg{idx}", best_for_cfg, comment=comment)
            logger.info(f"Best for config {idx+1}: val_energy={best_for_cfg['val_energy']:.6f}, num_params={best_for_cfg['num_params']}")
        else:
            logger.warning(f"No successful trials for config {idx+1}")

        def is_better(new, old):
            if new is None:
                return False
            if old is None:
                return True
            # 先看能量
            if new["val_energy"] + 1e-4 < old["val_energy"]:
                return True
            if abs(new["val_energy"] - old["val_energy"]) <= 1e-4:
                # 能量相近时，参数更少者优先
                return new["num_params"] < old["num_params"]
            return False

        if is_better(best_for_cfg, best_overall):
            best_overall = best_for_cfg
            best_overall_config = config
            best_overall_spec = ansatz_spec_dict

    logger.info("\n=== Ansatz Search Final Best ===")
    if best_overall_config and best_overall is not None and best_overall_spec is not None:
        logger.info(f"Best config: {summarize_config(best_overall_config)}")
        print_results(best_overall, logger=logger)
        
        # 重新构造 create_circuit_fn（开销极小），确保与 best_overall_spec 对齐。
        best_ansatz_obj = make_create_circuit_fn(best_overall_config)
        if isinstance(best_ansatz_obj, tuple) and len(best_ansatz_obj) == 2:
            best_create_fn = best_ansatz_obj[0]
        else:
            best_create_fn = getattr(best_ansatz_obj, "create_circuit")

        report_path = generate_report(
            exp_dir,
            f"{base_exp_name}_Best_Report",
            best_overall,
            best_create_fn,
            comment=f"Best config: {summarize_config(best_overall_config)}",
            # 使用与 Baseline Zoo 对齐的 ansatz_spec 字典写入 results.jsonl
            ansatz_spec=best_overall_spec,
        )
        logger.info(f"Report generated at: {report_path}")

        # Explicitly save the best config for MultiDim search
        config_path = os.path.join(exp_dir, "best_config_multidim.json")
        try:
            with open(config_path, "w") as f:
                json.dump(best_overall_config, f, indent=4)
            logger.info(f"Best MultiDim config saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save best config: {e}")

        return {
            "best_config": best_overall_config,
            "best_results": best_overall,
            "report_path": report_path,
            "ansatz_spec": best_overall_spec,
        }
    else:
        logger.error("No valid results found in search.")
        return {}

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
):
    """
    自动生成实验报告：Markdown 分析 + 线路图像可视化
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"report_{timestamp}.md"
    report_path = os.path.join(exp_dir, report_name)
    
    # 1. 尝试生成真实的线路图片 (PNG)
    circuit_img_name = f"circuit_{timestamp}.png"
    circuit_img_path = os.path.join(exp_dir, circuit_img_name)
    has_image = False
    
    try:
        c, _ = create_circuit_fn(results['final_params'])
        _save_tensorcircuit_circuit_png(c, circuit_img_path)
        has_image = True
    except Exception as e:
        print(f"Image generation failed: {e}")

    # 1.5 绘制收敛曲线
    convergence_img_name = f"convergence_{timestamp}.png"
    convergence_img_path = os.path.join(exp_dir, convergence_img_name)
    has_convergence = False
    energy_history = results.get("energy_history", [])
    if energy_history:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(energy_history, linewidth=0.8, color='#2196F3')
            ax.axhline(y=results['exact_energy'], color='#F44336', linestyle='--', linewidth=1, label=f"Exact: {results['exact_energy']:.6f}")
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

    # 2. 保存完整线路 JSON（仅用于后续分析，不在报告中展示）
    circuit_json_path = os.path.join(exp_dir, f"circuit_{timestamp}.json")
    try:
        c, _ = create_circuit_fn(results["final_params"])
        raw_data = c.to_json()
        import json

        if isinstance(raw_data, str):
            circuit_data = json.loads(raw_data)
        else:
            circuit_data = raw_data

        with open(circuit_json_path, "w") as f:
            f.write(json.dumps(circuit_data, indent=2))
    except Exception:
        # JSON 导出失败不会影响报告生成
        pass

    # 2.5 获取审计信息
    system = _infer_system_from_exp_dir(exp_dir)
    git_root = _find_git_root(exp_dir)
    git_info = _get_git_info(git_root)
    runtime_env = _get_runtime_env()

    # 2.6 Optimize git_diff in record
    if git_info.get("diff"):
        diff_path = _save_git_diff_artifact(exp_dir, git_info["diff"])
        git_info["diff_path"] = diff_path
        # Remove full diff from memory to keep jsonl small
        del git_info["diff"]

    # 3. 生成报告文本
    accuracy_status = "探索中"
    if results.get('energy_error') is not None:
        if results['energy_error'] < 0.0016:
            accuracy_status = "成功 (达到化学精度)"
        if results['energy_error'] < 1e-5:
            accuracy_status = "完美收敛 (高精度)"

    img_embed = f"![Circuit Diagram]({circuit_img_name})\n" if has_image else "*线路图生成暂不可用，请查看 JSON 结构数据。*\n"

    convergence_embed = f"![Convergence Curve]({convergence_img_name})\n" if has_convergence else ""
    actual_steps = results.get('actual_steps', 'N/A')

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
{comment if comment else "自动生成的分析报告：实验已完成，收敛曲线正常。"}

### 精度评价
{'当前结果已进入化学精度范围。' if (results.get('energy_error') or 1.0) < 0.0016 else '当前结果尚未达到化学精度，建议压榨参数或增加深度。'}

## 五、 审计信息
- **配置路径**: `{config_path or "N/A"}`
- **代码版本**: `{(git_info.get("commit") or "unknown")[:8] if git_info.get("commit") else "unknown"}`
- **环境指纹**: `Python {runtime_env.get("python_version", "N/A")}`

---
*完整实验数据（包括线路 JSON 与图像）已保存至目录下。*
"""
    with open(report_path, "w") as f:
        f.write(report_content)

    # 4. 追加结构化实验记录到 results.jsonl
    try:

        metrics: Dict[str, Any] = {
            "val_energy": results.get("val_energy"),
            "exact_energy": results.get("exact_energy"),
            "energy_error": results.get("energy_error"),
            "num_params": results.get("num_params"),
            "depth": None,  # 可在后续版本用真实线路深度替换
            "two_qubit_gates": _estimate_two_qubit_gates(
                ansatz_spec, results.get("n_qubits")
            ),
            "runtime_sec": results.get("training_seconds"),
            "actual_steps": results.get("actual_steps"),
        }

        artifact_paths: Dict[str, Any] = {
            "report_md": report_path,
            "circuit_png": circuit_img_path if has_image else None,
            "convergence_png": convergence_img_path if has_convergence else None,
            "circuit_json": circuit_json_path,
        }

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
            "git_diff": git_info["diff"],
            "runtime_env": runtime_env,
            "artifact_paths": artifact_paths,
        }

        _append_experiment_jsonl(exp_dir, experiment_record)
    except Exception as e:
        # 结构化记录永远不应阻塞正常报告生成流程
        print(f"Failed to log structured experiment: {e}")
        pass

    return report_path
