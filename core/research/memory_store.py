import datetime
import json
import os
from typing import Any, Dict, List, Optional

from core.model.research_schemas import DecisionRecord, ResearchMemory, RunBundle


class ResearchMemoryStore:
    """Structured research memory with compatibility views for legacy files."""

    def __init__(self, system_dir: str, state_dir: Optional[str] = None):
        self.system_dir = system_dir
        self.state_dir = state_dir or system_dir
        os.makedirs(self.state_dir, exist_ok=True)
        self.memory_path = os.path.join(self.state_dir, "research_memory.json")
        self.jsonl_path = os.path.join(self.state_dir, "autoresearch.jsonl")
        self.md_path = os.path.join(self.state_dir, "autoresearch.md")

    def default_memory(self) -> ResearchMemory:
        return ResearchMemory(
            system=os.path.basename(self.system_dir),
            objective=f"Optimize {os.path.basename(self.system_dir)} VQE ansatz.",
        )

    def load(self) -> ResearchMemory:
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return ResearchMemory.model_validate(json.load(f))
        return self.default_memory()

    def save(self, memory: ResearchMemory) -> None:
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(memory.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
        markdown = self.render_markdown(memory)
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

    def append_decision(self, decision: DecisionRecord, run: Optional[RunBundle] = None) -> None:
        record: Dict[str, Any] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "iteration": decision.iteration,
            "hypothesis_id": decision.hypothesis_id,
            "action_id": decision.action_id,
            "decision": decision.decision,
            "summary": decision.summary,
            "evidence_for": decision.evidence_for,
            "evidence_against": decision.evidence_against,
            "confidence": decision.confidence,
            "selected_config_path": decision.selected_config_path,
            "selected_candidate_id": decision.selected_candidate_id,
            "followup_actions": decision.followup_actions,
            "schema_version": "2.0",
        }
        if run is not None:
            record["run"] = run.model_dump(mode="json")
            record["results"] = run.metrics
            record["artifact_paths"] = run.artifact_paths
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_legacy_record(
        self,
        *,
        iteration: int,
        hypothesis: str,
        action: str,
        results: Dict[str, Any],
        decision: str,
        rationale: str,
    ) -> None:
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "iteration": iteration,
            "hypothesis": hypothesis,
            "action": action,
            "results": results,
            "decision": decision,
            "rationale": rationale,
            "schema_version": "1.0",
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def apply_decision_to_memory(
        self,
        memory: ResearchMemory,
        decision: DecisionRecord,
        run: Optional[RunBundle] = None,
        *,
        objective: Optional[str] = None,
        dead_ends: Optional[List[str]] = None,
        next_recommendations: Optional[List[str]] = None,
        transferable_insights: Optional[List[str]] = None,
        strategy_name: Optional[str] = None,
    ) -> ResearchMemory:
        if objective:
            memory.objective = objective
        memory.last_decision = decision
        if decision.decision in {"keep", "promote"}:
            if decision.hypothesis_id not in memory.accepted_hypotheses:
                memory.accepted_hypotheses.append(decision.hypothesis_id)
            if decision.hypothesis_id in memory.rejected_hypotheses:
                memory.rejected_hypotheses.remove(decision.hypothesis_id)
        elif decision.decision == "discard":
            if decision.hypothesis_id not in memory.rejected_hypotheses:
                memory.rejected_hypotheses.append(decision.hypothesis_id)
        if dead_ends:
            for dead_end in dead_ends:
                if dead_end not in memory.dead_ends:
                    memory.dead_ends.append(dead_end)
        if next_recommendations is not None:
            memory.next_recommendations = list(next_recommendations)
        if transferable_insights:
            for insight in transferable_insights:
                if insight not in memory.transferable_insights:
                    memory.transferable_insights.append(insight)
        if strategy_name:
            stats = memory.strategy_stats.setdefault(
                strategy_name,
                {"runs": 0, "keeps": 0, "discards": 0, "promotions": 0},
            )
            stats["runs"] += 1
            if decision.decision == "keep":
                stats["keeps"] += 1
            elif decision.decision == "discard":
                stats["discards"] += 1
            elif decision.decision == "promote":
                stats["promotions"] += 1
        if run is not None:
            energy_error = run.metrics.get("energy_error")
            num_params = run.metrics.get("num_params")
            if isinstance(energy_error, (int, float)):
                best_error = memory.best_energy_error
                best_params = memory.best_num_params
                improves_error = best_error is None or energy_error < best_error
                improves_efficiency = (
                    best_error is not None
                    and best_params is not None
                    and energy_error <= best_error
                    and isinstance(num_params, (int, float))
                    and int(num_params) < best_params
                )
                if improves_error or improves_efficiency:
                    memory.best_energy_error = float(energy_error)
                    memory.best_config_path = run.selected_config_path or decision.selected_config_path
                    if isinstance(num_params, (int, float)):
                        memory.best_num_params = int(num_params)
                    if decision.selected_candidate_id:
                        memory.best_candidate_id = decision.selected_candidate_id
        return memory

    def update_from_decision(
        self,
        decision: DecisionRecord,
        run: Optional[RunBundle] = None,
        *,
        objective: Optional[str] = None,
        dead_ends: Optional[List[str]] = None,
        next_recommendations: Optional[List[str]] = None,
        transferable_insights: Optional[List[str]] = None,
        strategy_name: Optional[str] = None,
    ) -> ResearchMemory:
        memory = self.load()
        memory = self.apply_decision_to_memory(
            memory,
            decision,
            run,
            objective=objective,
            dead_ends=dead_ends,
            next_recommendations=next_recommendations,
            transferable_insights=transferable_insights,
            strategy_name=strategy_name,
        )
        self.save(memory)
        return memory

    def render_markdown(self, memory: ResearchMemory) -> str:
        dead_ends = "\n".join(f"- {item}" for item in memory.dead_ends) or "- None yet"
        next_hypotheses = "\n".join(f"- {item}" for item in memory.next_recommendations) or "- None yet"
        insights = "\n".join(f"- {item}" for item in memory.transferable_insights) or "- None yet"
        best_config = memory.best_config_path or "N/A"
        best_error = "N/A" if memory.best_energy_error is None else f"{memory.best_energy_error:.2e}"
        best_params = "N/A" if memory.best_num_params is None else str(memory.best_num_params)
        last_summary = memory.last_decision.summary if memory.last_decision else "No decisions recorded yet."
        return f"""# Research Brain: {os.path.basename(self.system_dir)}

## Objective
{memory.objective}

## Best Known Configuration
- **Energy Error**: {best_error}
- **Num Params**: {best_params}
- **Config Path**: `{best_config}`
- **Candidate ID**: `{memory.best_candidate_id or "N/A"}`

## Last Decision
{last_summary}

## Dead Ends
{dead_ends}

## Transferable Insights
{insights}

## Next Hypotheses
{next_hypotheses}

## Iteration History
(See autoresearch.jsonl for full details)
"""
