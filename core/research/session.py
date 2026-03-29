import json
import os
from typing import Any, Dict, List, Optional

from core.model.research_schemas import ActionSpec, DecisionRecord, RunBundle
from core.research.memory_store import ResearchMemoryStore


class ResearchSession:
    """
    Manages the autoresearch.jsonl and autoresearch.md files for an autonomous agent.
    """

    def __init__(self, system_dir: str, state_dir: Optional[str] = None):
        self.system_dir = system_dir
        self.state_dir = state_dir or system_dir
        self.store = ResearchMemoryStore(system_dir, self.state_dir)
        self.jsonl_path = self.store.jsonl_path
        self.md_path = self.store.md_path
        self.memory_path = self.store.memory_path

    def log_decision(
        self,
        iteration: int,
        hypothesis: str,
        action: str,
        results: Dict[str, Any],
        decision: str,
        rationale: str,
    ):
        """
        Compatibility wrapper that writes the legacy event shape.
        """
        action_spec = ActionSpec(
            action_id=f"legacy-action-{iteration}",
            hypothesis_id=f"legacy-hypothesis-{iteration}",
            system_dir=self.system_dir,
            action_type="run_strategy",
            rationale=action,
        )
        decision_record = DecisionRecord(
            decision_id=f"legacy-decision-{iteration}",
            iteration=iteration,
            hypothesis_id=action_spec.hypothesis_id,
            action_id=action_spec.action_id,
            decision=decision,  # type: ignore[arg-type]
            summary=rationale,
            evidence_for=[f"{key}={value}" for key, value in sorted(results.items())],
            confidence=0.5,
            selected_candidate_id=results.get("selected_candidate_id"),
            selected_config_path=results.get("selected_config_path"),
        )
        run = RunBundle(
            action=action_spec,
            metrics=results,
            selected_candidate_id=results.get("selected_candidate_id"),
            selected_config_path=results.get("selected_config_path"),
        )
        self.record_structured_decision(decision_record, run)
        self.store.append_legacy_record(
            iteration=iteration,
            hypothesis=hypothesis,
            action=action,
            results=results,
            decision=decision,
            rationale=rationale,
        )

    def record_structured_decision(self, decision: DecisionRecord, run: Optional[RunBundle] = None) -> None:
        """Primary write path for research decisions."""
        self.store.append_decision(decision, run)

    def update_brain(
        self,
        objective: str,
        best_config: Dict[str, Any],
        best_energy_error: float,
        dead_ends: List[str],
        next_hypotheses: List[str],
    ):
        """
        Updates the autoresearch.md 'brain' file.
        """
        memory = self.store.load()
        memory.objective = objective
        memory.best_energy_error = best_energy_error
        memory.dead_ends = list(dead_ends)
        memory.next_recommendations = list(next_hypotheses)
        config_path = os.path.join(self.state_dir, "best_config_snapshot.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(best_config, f, ensure_ascii=False, indent=2)
        memory.best_config_path = config_path
        self.store.save(memory)

    def apply_structured_outcome(
        self,
        decision: DecisionRecord,
        run: Optional[RunBundle] = None,
        *,
        objective: Optional[str] = None,
        best_config: Optional[Dict[str, Any]] = None,
        dead_ends: Optional[List[str]] = None,
        next_hypotheses: Optional[List[str]] = None,
        strategy_name: Optional[str] = None,
    ) -> None:
        memory = self.store.update_from_decision(
            decision,
            run,
            objective=objective,
            dead_ends=dead_ends,
            next_recommendations=next_hypotheses,
            strategy_name=strategy_name,
        )
        if best_config is not None:
            config_path = os.path.join(self.state_dir, "best_config_snapshot.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(best_config, f, ensure_ascii=False, indent=2)
            memory.best_config_path = config_path
            self.store.save(memory)

    def get_latest_iteration(self) -> int:
        if not os.path.exists(self.jsonl_path):
            return 0
        iterations = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    iterations.append(data.get("iteration", 0))
                except Exception:
                    continue
        return max(iterations) if iterations else 0

    def get_best_performance(self) -> float:
        best_err = float("inf")
        if not os.path.exists(self.jsonl_path):
            return best_err
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    err = data.get("results", {}).get("energy_error", float("inf"))
                    if err < best_err:
                        best_err = err
                except Exception:
                    continue
        return best_err
