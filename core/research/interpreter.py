from typing import Any, Dict, List, Tuple

from core.model.research_schemas import (
    ActionSpec,
    DecisionRecord,
    FailureType,
    HypothesisSpec,
    ResearchMemory,
    RunBundle,
)


class ResultInterpreter:
    """Convert raw run metrics into research-level decisions."""

    def __init__(self, session_jsonl_path: str | None = None):
        self.session_jsonl_path = session_jsonl_path

    def interpret(
        self,
        *,
        iteration: int,
        memory: ResearchMemory,
        hypothesis: HypothesisSpec,
        action: ActionSpec,
        metrics: Dict[str, Any],
    ) -> DecisionRecord:
        return self.interpret_run(
            iteration=iteration,
            memory=memory,
            hypothesis=hypothesis,
            run=RunBundle(
                action=action,
                metrics=metrics,
                target_candidate_id=action.target_candidate_id,
                selected_candidate_id=metrics.get("selected_candidate_id"),
                selected_config_path=metrics.get("selected_config_path"),
            ),
        )

    def interpret_run(
        self,
        *,
        iteration: int,
        memory: ResearchMemory,
        hypothesis: HypothesisSpec,
        run: RunBundle,
    ) -> DecisionRecord:
        action = run.action
        metrics = run.metrics
        if not run.success:
            summary, failure_type, evidence_against, followup_actions, failure_signals = self._interpret_failed_run(run)
            return DecisionRecord(
                decision_id=f"{action.action_id}-decision",
                iteration=iteration,
                hypothesis_id=hypothesis.hypothesis_id,
                action_id=action.action_id,
                decision="refine",
                summary=summary,
                evidence_against=evidence_against,
                confidence=0.9,
                failure_type=failure_type,
                failure_signals=failure_signals,
                selected_candidate_id=run.selected_candidate_id or metrics.get("selected_candidate_id"),
                selected_config_path=run.selected_config_path or metrics.get("selected_config_path"),
                followup_actions=followup_actions,
            )

        current_err = metrics.get("energy_error", float("inf"))
        current_params = metrics.get("num_params", 999)
        best_err = memory.best_energy_error if memory.best_energy_error is not None else float("inf")
        best_params = memory.best_num_params if memory.best_num_params is not None else 999

        is_better_energy = isinstance(current_err, (int, float)) and current_err < best_err * 0.95
        is_better_efficiency = (
            isinstance(current_err, (int, float))
            and current_err < best_err * 1.05
            and current_params < best_params * 0.9
        )

        failure_type, failure_signals, failure_evidence, followup_actions = self._classify_failure(
            memory=memory,
            run=run,
            current_err=current_err,
            current_params=current_params,
            best_err=best_err,
            best_params=best_params,
        )

        if best_err == float("inf") or is_better_energy or is_better_efficiency:
            decision = "keep"
        else:
            decision = "discard"

        summary = f"Energy: {current_err:.2e} (vs {best_err:.2e}), Params: {current_params} (vs {best_params}). "
        evidence_for = []
        evidence_against = []
        if best_err == float("inf"):
            summary += "Established the first baseline."
            evidence_for.append("No prior accepted result existed in memory.")
        elif is_better_energy:
            summary += "Significant energy improvement."
            evidence_for.append("energy_error improved by at least 5%.")
        elif is_better_efficiency:
            summary += "Better parameter efficiency."
            evidence_for.append("Similar energy with materially fewer parameters.")
        else:
            summary += self._failure_summary_suffix(failure_type)
            evidence_against.extend(failure_evidence)

        return DecisionRecord(
            decision_id=f"{action.action_id}-decision",
            iteration=iteration,
            hypothesis_id=hypothesis.hypothesis_id,
            action_id=action.action_id,
            decision=decision,
            summary=summary,
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            confidence=0.75 if decision == "keep" else 0.55,
            failure_type=None if decision == "keep" else failure_type,
            failure_signals={} if decision == "keep" else failure_signals,
            selected_candidate_id=run.selected_candidate_id or metrics.get("selected_candidate_id"),
            selected_config_path=run.selected_config_path or metrics.get("selected_config_path"),
            followup_actions=[] if decision == "keep" else followup_actions,
        )

    def _interpret_failed_run(
        self,
        run: RunBundle,
    ) -> Tuple[str, FailureType, List[str], List[str], Dict[str, Any]]:
        failure_signals = {}
        if run.error_message:
            failure_signals["error_message"] = run.error_message
        if run.metrics:
            failure_signals["available_metrics"] = sorted(run.metrics.keys())
        summary = f"Execution failed for action {run.action.action_type}."
        if run.error_message:
            summary += f" {run.error_message}"
        
        evidence_against = ["The run did not yield a usable energy_error metric."]
        if run.error_message:
            evidence_against.append(f"Error captured: {run.error_message}")
            
        return (
            summary,
            "execution_failure",
            evidence_against,
            ["retry_same_strategy", "switch_strategy"],
            failure_signals,
        )

    def _classify_failure(
        self,
        *,
        memory: ResearchMemory,
        run: RunBundle,
        current_err: Any,
        current_params: Any,
        best_err: float,
        best_params: int,
    ) -> Tuple[FailureType, Dict[str, Any], List[str], List[str]]:
        metrics = run.metrics
        failure_signals: Dict[str, Any] = {}
        evidence_against: List[str] = []
        followup_actions: List[str] = []

        if isinstance(metrics.get("gradient_norm"), (int, float)) and float(metrics["gradient_norm"]) < 1e-8:
            failure_signals["gradient_norm"] = float(metrics["gradient_norm"])
            evidence_against.append("Gradient norm collapsed below the actionable threshold.")
            followup_actions.extend(["switch_strategy", "reduce_search_space"])
            return "gradient_collapse", failure_signals, evidence_against, followup_actions

        if metrics.get("warmstart_failed") or metrics.get("warmstart_reuse_ratio") == 0:
            failure_signals["warmstart_failed"] = bool(metrics.get("warmstart_failed", False))
            if "warmstart_reuse_ratio" in metrics:
                failure_signals["warmstart_reuse_ratio"] = metrics["warmstart_reuse_ratio"]
            evidence_against.append("Warm-start information did not transfer into a usable initialization.")
            followup_actions.extend(["reduce_search_space", "switch_strategy"])
            return "warmstart_failure", failure_signals, evidence_against, followup_actions

        if (
            isinstance(current_err, (int, float))
            and isinstance(current_params, (int, float))
            and best_err != float("inf")
            and current_err <= best_err * 1.05
            and current_params >= best_params
        ):
            failure_signals["energy_ratio_vs_best"] = float(current_err) / best_err if best_err else 1.0
            failure_signals["num_params"] = int(current_params)
            evidence_against.append("The candidate used at least as many parameters without compensating accuracy gain.")
            followup_actions.append("reduce_search_space")
            return "parameter_inefficiency", failure_signals, evidence_against, followup_actions

        actual_steps = metrics.get("actual_steps")
        max_steps = metrics.get("max_steps")
        if (
            isinstance(current_err, (int, float))
            and best_err != float("inf")
            and current_err >= best_err * 0.98
            and isinstance(actual_steps, (int, float))
            and isinstance(max_steps, (int, float))
            and actual_steps >= max_steps
        ):
            failure_signals["actual_steps"] = int(actual_steps)
            failure_signals["max_steps"] = int(max_steps)
            evidence_against.append("Optimization consumed the full step budget without producing a new Pareto point.")
            followup_actions.extend(["switch_strategy", "reduce_search_space"])
            return "optimizer_stall", failure_signals, evidence_against, followup_actions

        runs_for_strategy = 0
        if run.action.strategy_name:
            strategy_stats = memory.strategy_stats.get(run.action.strategy_name, {})
            runs_for_strategy = int(strategy_stats.get("runs", 0))
            failure_signals["strategy_runs"] = runs_for_strategy
        evidence_against.append("No meaningful energy or efficiency gain over the current best result.")
        if runs_for_strategy >= 2:
            followup_actions.append("switch_strategy")
        followup_actions.append("reduce_search_space")
        return "no_pareto_improvement", failure_signals, evidence_against, followup_actions

    def _failure_summary_suffix(self, failure_type: FailureType) -> str:
        mapping = {
            "gradient_collapse": "Gradient collapse suggests the current region is too flat.",
            "warmstart_failure": "Warm-start transfer failed to produce a useful initialization.",
            "parameter_inefficiency": "Comparable accuracy required too many parameters.",
            "optimizer_stall": "Optimization stalled before finding a better Pareto point.",
            "no_pareto_improvement": "No Pareto improvement.",
            "execution_failure": "Execution failed before producing valid metrics.",
        }
        return mapping.get(failure_type, "No Pareto improvement.")
