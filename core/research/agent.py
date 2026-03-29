import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.model.research_schemas import ActionSpec, ResearchMemory, RunBundle
from core.orchestration.controller import SearchController
from core.research.executor import ExperimentExecutor
from core.research.interpreter import ResultInterpreter
from core.research.policy import PolicyEngine
from core.research.session import ResearchSession


def default_emit(message: str, log_path: Optional[str] = None) -> None:
    print(message)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def default_emit_stream_line(line: str, log_path: Optional[str] = None) -> None:
    print(line, end="")
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)


class ResearchAgent:
    """Single-agent outer loop coordinating policy, execution, and memory updates."""

    def __init__(
        self,
        *,
        system_dir: str,
        strategy: str,
        session: ResearchSession,
        policy_engine: Optional[PolicyEngine] = None,
        controller: Optional[SearchController] = None,
        executor: Optional[ExperimentExecutor] = None,
        interpreter: Optional[ResultInterpreter] = None,
        log_path: Optional[str] = None,
        session_dir: Optional[str] = None,
        emit: Optional[Callable[[str, Optional[str]], None]] = None,
        emit_stream_line: Optional[Callable[[str, Optional[str]], None]] = None,
    ):
        self.system_dir = system_dir
        self.strategy = strategy
        self.session = session
        self.policy_engine = policy_engine or PolicyEngine(strategy)
        self.controller = controller or SearchController()
        self.executor = executor or ExperimentExecutor()
        self.interpreter = interpreter or ResultInterpreter(session.jsonl_path)
        self.log_path = log_path
        self.session_dir = session_dir
        self.emit = emit
        self.emit_stream_line = emit_stream_line

    def _load_memory(self) -> ResearchMemory:
        return self.session.store.load()

    def _default_objective(self) -> str:
        return f"优化 {os.path.basename(self.system_dir)} 的 Ansatz，将能量误差降至更低。"

    def _default_recommendations(self) -> List[str]:
        return ["继续扩大搜索空间", "尝试增加更多层数"]

    def _default_dead_end(self, iteration: int) -> List[str]:
        return [f"{self.strategy}: no pareto improvement at iteration {iteration}"]

    def _load_best_config(self, selected_config_path: Optional[str]) -> Dict[str, Any]:
        config_path = selected_config_path
        if not isinstance(config_path, str) or not config_path:
            config_path = os.path.join(self.system_dir, "presets", "ga.json")
        if os.path.exists(config_path):
            import json

            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def execute_structured_step(
        self,
        iteration: int,
        *,
        memory: Optional[ResearchMemory] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[RunBundle]]:
        memory = memory or self._load_memory()
        hypothesis = self.policy_engine.propose_hypothesis(
            memory,
            self.controller,
            iteration=iteration,
            system_dir=self.system_dir,
        )
        action = self.policy_engine.plan_next_action(
            memory,
            self.controller,
            hypothesis=hypothesis,
            system_dir=self.system_dir,
        )
        if action.action_type == "stop_research":
            self.controller.stop(action.rationale or "Policy requested stop.")
            return False, {"decision": "stop"}, None

        run = self.executor.execute_action(
            action,
            iteration,
            session_dir=self.session_dir,
            log_path=self.log_path,
            emit=self.emit,
            emit_stream_line=self.emit_stream_line,
        )
        success = run.success
        metrics = run.metrics

        if not success:
            self.controller.report_result(metrics, is_failure=True)
            return False, metrics, run

        decision = self.interpreter.interpret_run(
            iteration=iteration,
            memory=memory,
            hypothesis=hypothesis,
            run=run,
        )
        self.session.record_structured_decision(decision, run)
        if decision.decision == "keep":
            best_config = self._load_best_config(decision.selected_config_path)
            self.session.apply_structured_outcome(
                decision,
                run,
                objective=self._default_objective(),
                best_config=best_config,
                next_hypotheses=self._default_recommendations(),
                strategy_name=self.strategy,
            )
        else:
            self.session.apply_structured_outcome(
                decision,
                run,
                dead_ends=self._default_dead_end(iteration),
                next_hypotheses=self._default_recommendations(),
                strategy_name=self.strategy,
            )

        self.controller.report_result(metrics, is_failure=False)
        return True, metrics, run

    def step(self, iteration: int) -> Tuple[bool, Dict[str, Any]]:
        success, metrics, _ = self.execute_structured_step(iteration)
        return success, metrics

    def run_until_stop(
        self,
        *,
        start_iteration: int,
        max_loops: int,
        target_error: Optional[float] = None,
        emit: Optional[Callable[[str, Optional[str]], None]] = None,
    ) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for iteration in range(start_iteration, max_loops + 1):
            if not self.controller.should_continue():
                break
            success, metrics = self.step(iteration)
            history.append(metrics)
            if not success:
                break
            current_err = metrics.get("energy_error", float("inf"))
            if target_error is not None and current_err < target_error:
                if emit:
                    emit(f"*** Target reached! Energy Error: {current_err:.2e}", self.log_path)
                break
            if emit:
                emit(
                    f"--- Iteration {iteration} finished. Best so far: {self.session.get_best_performance():.2e}",
                    self.log_path,
                )
        return history


def create_default_research_agent(
    *,
    system_dir: str,
    strategy: str,
    session: ResearchSession,
    log_path: Optional[str] = None,
    session_dir: Optional[str] = None,
    emit: Optional[Callable[[str, Optional[str]], None]] = None,
    emit_stream_line: Optional[Callable[[str, Optional[str]], None]] = None,
) -> ResearchAgent:
    return ResearchAgent(
        system_dir=system_dir,
        strategy=strategy,
        session=session,
        policy_engine=PolicyEngine(strategy),
        executor=ExperimentExecutor(subprocess_module=subprocess),
        log_path=log_path,
        session_dir=session_dir,
        emit=emit or default_emit,
        emit_stream_line=emit_stream_line or default_emit_stream_line,
    )


def run_single_iteration(
    *,
    system_dir: str,
    iteration: int,
    session: ResearchSession,
    strategy: str,
    session_dir: Optional[str] = None,
    log_path: Optional[str] = None,
    emit: Optional[Callable[[str, Optional[str]], None]] = None,
    emit_stream_line: Optional[Callable[[str, Optional[str]], None]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    agent = create_default_research_agent(
        system_dir=system_dir,
        strategy=strategy,
        session=session,
        log_path=log_path,
        session_dir=session_dir,
        emit=emit,
        emit_stream_line=emit_stream_line,
    )
    success, metrics, _ = agent.execute_structured_step(iteration)
    return success, metrics
