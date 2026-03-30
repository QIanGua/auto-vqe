import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Type

from core.model.research_schemas import ActionSpec, HypothesisSpec, ResearchMemory
from core.orchestration.controller import SearchController


class FailureHandler(ABC):
    """Base class for strategy responses to specific failure scenarios."""

    @abstractmethod
    def can_handle(self, failure_type: Optional[str], memory: ResearchMemory, controller: SearchController) -> bool:
        pass

    @abstractmethod
    def handle(
        self,
        engine: "PolicyEngine",
        memory: ResearchMemory,
        controller: SearchController,
        hypothesis: HypothesisSpec,
        system_dir: str,
    ) -> Optional[ActionSpec]:
        pass


class ExecutionFailureHandler(FailureHandler):
    def can_handle(self, failure_type: Optional[str], memory: ResearchMemory, controller: SearchController) -> bool:
        return failure_type == "execution_failure"

    def handle(self, engine, memory, controller, hypothesis, system_dir):
        alt_strats = engine.get_alternative_strategies(memory)
        if alt_strats:
            target_str = alt_strats[0]
            action = engine.build_switch_action(
                hypothesis, system_dir, target_str,
                rationale=f"Execution failed under {engine.strategy_name}; switch to {target_str} before retrying."
            )
            if target_str in ("ga", "multidim"):
                action.config_path = None
            return action
        return None


class StallHandler(FailureHandler):
    def can_handle(self, failure_type: Optional[str], memory: ResearchMemory, controller: SearchController) -> bool:
        return failure_type in {"optimizer_stall", "gradient_collapse", "adapt_saturation"}

    def handle(self, engine, memory, controller, hypothesis, system_dir):
        alt_strats = engine.get_alternative_strategies(memory)
        if alt_strats:
            target_str = alt_strats[0]
            rationale = f"Recent failure mode {memory.last_decision.failure_type} suggests trying {target_str}."
            action = engine.build_switch_action(hypothesis, system_dir, target_str, rationale)
            
            # Asymmetric Continuity Mechanism
            if target_str in ("adapt", "qubit_adapt"):
                # Pass the best configuration forward to ADAPT base
                action.config_path = getattr(memory, "best_config_path", None)
                if not action.config_path and memory.last_decision and memory.last_decision.selected_config_path:
                    action.config_path = memory.last_decision.selected_config_path
            elif target_str in ("ga", "multidim"):
                # Reset exploration space for GA
                action.config_path = None
                action.rationale += " Resetting exploration space."
            return action
        return None


class DeadEndHandler(FailureHandler):
    def can_handle(self, failure_type: Optional[str], memory: ResearchMemory, controller: SearchController) -> bool:
        return bool(memory.dead_ends) and failure_type not in {"execution_failure", "optimizer_stall", "gradient_collapse", "adapt_saturation"}

    def handle(self, engine, memory, controller, hypothesis, system_dir):
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
            strategy_name=engine.strategy_name,
            rationale="Use recorded dead ends to avoid repeating an already rejected region.",
            search_space_patch=patch,
        )


class StagnationHandler(FailureHandler):
    def can_handle(self, failure_type: Optional[str], memory: ResearchMemory, controller: SearchController) -> bool:
        # We check controller limit. Stats will be safely retrieved inside handle.
        return controller.consecutive_no_improvement >= controller.no_improvement_limit

    def handle(self, engine, memory, controller, hypothesis, system_dir):
        alt_strats = engine.get_alternative_strategies(memory)
        stats = engine._strategy_stats(memory)
        
        if alt_strats:
            target_str = alt_strats[0]
            action = engine.build_switch_action(
                hypothesis, system_dir, target_str,
                rationale="Repeated non-improvement suggests switching to another search regime."
            )
            if target_str in ("adapt", "qubit_adapt"):
                action.config_path = memory.last_decision.selected_config_path if memory.last_decision else None
            elif target_str in ("ga", "multidim"):
                action.config_path = None
                action.rationale += " Resetting exploration."
            return action
        
        return ActionSpec(
            action_id=f"{hypothesis.hypothesis_id}-reduce",
            hypothesis_id=hypothesis.hypothesis_id,
            system_dir=system_dir,
            action_type="reduce_search_space",
            strategy_name=engine.strategy_name,
            rationale="Repeated non-improvement or poor keep rate suggests the current region is too broad.",
            search_space_patch={"layers": "shrink", "entanglement": "simplify"},
        )


class PolicyEngine:
    """Rule-based research policy mapped across handler chain."""

    def __init__(self, strategy_name: str, available_strategies: tuple[str, ...] = ("ga", "multidim", "adapt", "qubit_adapt")):
        self.strategy_name = strategy_name
        self.available_strategies = available_strategies
        self.handlers: List[FailureHandler] = [
            ExecutionFailureHandler(),
            StallHandler(),
            DeadEndHandler(),
            StagnationHandler(),
        ]

    def _strategy_stats(self, memory: ResearchMemory) -> dict:
        stats = memory.strategy_stats.get(self.strategy_name, {})
        return {
            "runs": int(stats.get("runs", 0)),
            "keeps": int(stats.get("keeps", 0)),
            "discards": int(stats.get("discards", 0)),
            "promotions": int(stats.get("promotions", 0)),
        }

    def update_strategy(self, strategy_name: str) -> None:
        self.strategy_name = strategy_name

    def _normalize_pending_action(
        self,
        pending: ActionSpec,
        *,
        hypothesis: HypothesisSpec,
        memory: ResearchMemory,
    ) -> ActionSpec:
        strategy_name = pending.strategy_name
        if pending.action_type == "switch_strategy" and not strategy_name:
            alt = self.get_alternative_strategies(memory)
            strategy_name = alt[0] if alt else self.strategy_name
        budget = dict(pending.budget)
        budget["from_pending_queue"] = True
        budget["pending_action_id"] = pending.action_id
        return pending.model_copy(
            update={
                "action_id": f"{hypothesis.hypothesis_id}-{pending.action_type}",
                "hypothesis_id": hypothesis.hypothesis_id,
                "strategy_name": strategy_name,
                "budget": budget,
            }
        )

    def get_alternative_strategies(self, memory: ResearchMemory) -> List[str]:
        candidates = [name for name in self.available_strategies if name != self.strategy_name]
        if not candidates:
            return []
        ranked = sorted(
            candidates,
            key=lambda name: (
                int(memory.strategy_stats.get(name, {}).get("runs", 0)),
                int(memory.strategy_stats.get(name, {}).get("discards", 0)),
                name,
            ),
        )
        return ranked

    def build_switch_action(self, hypothesis: HypothesisSpec, system_dir: str, target_strategy: str, rationale: str) -> ActionSpec:
        return ActionSpec(
            action_id=f"{hypothesis.hypothesis_id}-switch",
            hypothesis_id=hypothesis.hypothesis_id,
            system_dir=system_dir,
            action_type="switch_strategy",
            strategy_name=target_strategy,
            fidelity="quick",
            rationale=rationale,
        )

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
        last_failure_type = memory.last_decision.failure_type if memory.last_decision else None

        if memory.pending_actions:
            normalized = self._normalize_pending_action(memory.pending_actions[0], hypothesis=hypothesis, memory=memory)
            if normalized.action_type != "switch_strategy" or normalized.strategy_name is not None:
                return normalized

        if target_error is not None and memory.best_energy_error is not None and memory.best_energy_error < target_error:
            return ActionSpec(
                action_id=f"{hypothesis.hypothesis_id}-stop",
                hypothesis_id=hypothesis.hypothesis_id,
                system_dir=system_dir,
                action_type="stop_research",
                rationale="Target error already satisfied.",
            )

        # Let the registered handlers decide based on recent failure patterns
        for handler in self.handlers:
            if handler.can_handle(last_failure_type, memory, controller):
                # We need a patch for StagnationHandler since it relies on controller, 
                # but the signature is simple enough.
                action = handler.handle(self, memory, controller, hypothesis, system_dir)
                if action:
                    return action

        # Default fallback
        # Check if poor keep rate is persistent
        if stats["discards"] > stats["keeps"] and stats["runs"] >= 3:
            alt_strats = self.get_alternative_strategies(memory)
            if alt_strats:
                return self.build_switch_action(
                    hypothesis, system_dir, alt_strats[0], 
                    rationale="Poor keep rate suggests switching strategy to explore a better region."
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
