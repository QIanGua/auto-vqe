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

    def record_structured_decision(self, decision: DecisionRecord, run: Optional[RunBundle] = None) -> None:
        """Primary write path for research decisions."""
        self.store.append_decision(decision, run)

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

    def consume_pending_action(self, action_id: str) -> None:
        self.store.consume_pending_action(action_id)
