import os
import time
from typing import Any, Optional, Literal

from core.model.schemas import EvaluationSpec, EvaluationResult, OptimizerSpec, CandidateSpec, WarmStartPlan


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


def evaluate_candidate(
    env: Any,
    candidate: CandidateSpec,
    evaluation: EvaluationSpec,
    warm_start: Optional[WarmStartPlan] = None,
    logger: Optional[Any] = None,
) -> EvaluationResult:
    """
    多保真评估包装器。
    """
    start_time = time.time()

    init_params = None
    if warm_start and candidate.warm_start_from:
        pass

    opt_spec = OptimizerSpec(
        method=evaluation.optimizer_name,
        max_steps=evaluation.max_steps,
        early_stop_threshold=1e-8 if evaluation.enable_early_stop else 1e-12,
    )

    from core.evaluator.training import optimize_parameters
    from core.representation.compiler import estimate_circuit_cost

    results = optimize_parameters(
        env=env,
        ansatz=candidate.ansatz,
        optimizer_spec=opt_spec,
        init_params=init_params,
        logger=logger,
    )
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
        artifacts={},
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
    return EvaluationSpec(fidelity="full", max_steps=500, n_seeds=3)
