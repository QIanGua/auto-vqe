from typing import Any, Dict

from core.model.research_schemas import ActionSpec, DecisionRecord, HypothesisSpec, ResearchMemory, RunBundle


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
            summary += "No Pareto improvement."
            evidence_against.append("No meaningful energy or efficiency gain over the current best result.")

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
            selected_candidate_id=run.selected_candidate_id or metrics.get("selected_candidate_id"),
            selected_config_path=run.selected_config_path or metrics.get("selected_config_path"),
        )
