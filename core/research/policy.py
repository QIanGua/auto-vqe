import os
from typing import Optional

from core.model.research_schemas import ActionSpec, HypothesisSpec, ResearchMemory
from core.orchestration.controller import SearchController


class PolicyEngine:
    """Rule-based research policy for the initial agentized loop."""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    def _strategy_stats(self, memory: ResearchMemory) -> dict:
        stats = memory.strategy_stats.get(self.strategy_name, {})
        return {
            "runs": int(stats.get("runs", 0)),
            "keeps": int(stats.get("keeps", 0)),
            "discards": int(stats.get("discards", 0)),
            "promotions": int(stats.get("promotions", 0)),
        }

    def propose_hypothesis(
        self,
        memory: ResearchMemory,
        controller: SearchController,
        *,
        iteration: int,
        system_dir: str,
    ) -> HypothesisSpec:
        system_name = memory.system or os.path.basename(system_dir)
        stats = self._strategy_stats(memory)
        recent_dead_end = memory.dead_ends[-1] if memory.dead_ends else None
        if recent_dead_end:
            statement = f"Avoid the most recent dead end and search a narrower neighboring region: {recent_dead_end}."
        elif controller.consecutive_no_improvement > 0:
            statement = "Shrink or redirect the search region after repeated non-improvement."
        elif stats["discards"] > stats["keeps"] and stats["runs"] >= 3:
            statement = "Re-balance the search toward simpler configurations because recent runs were mostly discarded."
        elif memory.best_energy_error is None:
            statement = "Establish a first Pareto-efficient baseline in the current search space."
        else:
            statement = "Seek a lower energy error or a simpler configuration under similar accuracy."
        search_region = {"strategy": self.strategy_name}
        if recent_dead_end:
            search_region["avoid"] = recent_dead_end
        return HypothesisSpec(
            hypothesis_id=f"{self.strategy_name}-iter-{iteration:04d}",
            system=system_name,
            objective=memory.objective,
            statement=statement,
            expected_effect="Improve energy_error under complexity constraints.",
            search_region=search_region,
            assumptions=["Quantum environment definition remains fixed."],
        )

    def plan_next_action(
        self,
        memory: ResearchMemory,
        controller: SearchController,
        *,
        hypothesis: HypothesisSpec,
        system_dir: str,
        target_error: Optional[float] = None,
    ) -> ActionSpec:
        stats = self._strategy_stats(memory)
        if target_error is not None and memory.best_energy_error is not None and memory.best_energy_error < target_error:
            return ActionSpec(
                action_id=f"{hypothesis.hypothesis_id}-stop",
                hypothesis_id=hypothesis.hypothesis_id,
                system_dir=system_dir,
                action_type="stop_research",
                rationale="Target error already satisfied.",
            )

        if memory.dead_ends:
            last_dead_end = memory.dead_ends[-1].lower()
            patch = {"layers": "shrink", "entanglement": "simplify"}
            if "layer" in last_dead_end:
                patch["layers"] = "expand_cautiously"
            if "entanglement" in last_dead_end:
                patch["entanglement"] = "avoid_recent_pattern"
            return ActionSpec(
                action_id=f"{hypothesis.hypothesis_id}-redirect",
                hypothesis_id=hypothesis.hypothesis_id,
                system_dir=system_dir,
                action_type="reduce_search_space",
                strategy_name=self.strategy_name,
                rationale="Use recorded dead ends to avoid repeating an already rejected region.",
                search_space_patch=patch,
            )

        if controller.consecutive_no_improvement >= controller.no_improvement_limit or (
            stats["discards"] > stats["keeps"] and stats["runs"] >= 3
        ):
            return ActionSpec(
                action_id=f"{hypothesis.hypothesis_id}-reduce",
                hypothesis_id=hypothesis.hypothesis_id,
                system_dir=system_dir,
                action_type="reduce_search_space",
                strategy_name=self.strategy_name,
                rationale="Repeated non-improvement or poor keep rate suggests the current region is too broad.",
                search_space_patch={"layers": "shrink", "entanglement": "simplify"},
            )

        return ActionSpec(
            action_id=f"{hypothesis.hypothesis_id}-run",
            hypothesis_id=hypothesis.hypothesis_id,
            system_dir=system_dir,
            action_type="run_strategy",
            strategy_name=self.strategy_name,
            fidelity="medium" if stats["keeps"] > 0 else "quick",
            budget={"max_runs": controller.max_runs, "wall_clock_seconds": controller.max_wall_clock_seconds},
            rationale="Run the next research iteration with the current primary strategy.",
        )
