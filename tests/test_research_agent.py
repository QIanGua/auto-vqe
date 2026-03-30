from core.orchestration.controller import SearchController
from core.model.research_schemas import ActionSpec, ResearchMemory
from core.research.agent import ResearchAgent
from core.research.policy import PolicyEngine
from core.research.session import ResearchSession


def test_policy_engine_returns_stop_action_when_target_reached(tmp_path):
    session = ResearchSession(str(tmp_path / "experiments" / "lih"), str(tmp_path / "state"))
    memory = session.store.load()
    memory.best_energy_error = 1e-7
    session.store.save(memory)

    policy = PolicyEngine("ga")
    controller = SearchController()
    hypothesis = policy.propose_hypothesis(memory, controller, iteration=1, system_dir=str(tmp_path))
    action = policy.plan_next_action(
        memory,
        controller,
        hypothesis=hypothesis,
        system_dir=str(tmp_path),
        target_error=1e-6,
    )

    assert action.action_type == "stop_research"


def test_policy_engine_uses_dead_end_memory_to_reduce_search_space(tmp_path):
    policy = PolicyEngine("ga")
    controller = SearchController()
    memory = ResearchMemory(
        system="lih",
        objective="Optimize LiH ansatz.",
        dead_ends=["entanglement-heavy search region kept failing"],
        strategy_stats={"ga": {"runs": 4, "keeps": 1, "discards": 3}},
    )
    hypothesis = policy.propose_hypothesis(memory, controller, iteration=2, system_dir=str(tmp_path))
    action = policy.plan_next_action(
        memory,
        controller,
        hypothesis=hypothesis,
        system_dir=str(tmp_path),
    )

    assert action.action_type == "reduce_search_space"
    assert action.search_space_patch["entanglement"] == "avoid_recent_pattern"


def test_policy_engine_switches_strategy_after_optimizer_stall(tmp_path):
    policy = PolicyEngine("ga")
    controller = SearchController()
    controller.consecutive_no_improvement = controller.no_improvement_limit
    memory = ResearchMemory(
        system="lih",
        objective="Optimize LiH ansatz.",
        strategy_stats={"ga": {"runs": 4, "keeps": 1, "discards": 3}},
    )
    from core.model.research_schemas import DecisionRecord

    memory.last_decision = DecisionRecord(
        decision_id="d1",
        iteration=3,
        hypothesis_id="h1",
        action_id="a1",
        decision="discard",
        summary="Optimizer stalled.",
        evidence_against=["No improvement after full step budget."],
        confidence=0.6,
        failure_type="optimizer_stall",
    )
    hypothesis = policy.propose_hypothesis(memory, controller, iteration=4, system_dir=str(tmp_path))
    action = policy.plan_next_action(
        memory,
        controller,
        hypothesis=hypothesis,
        system_dir=str(tmp_path),
    )

    assert action.action_type == "switch_strategy"
    assert action.strategy_name == "adapt"


def test_policy_engine_limits_switches_to_available_strategies():
    policy = PolicyEngine("ga", available_strategies=("ga", "multidim", "qubit_adapt"))
    controller = SearchController()
    memory = ResearchMemory(
        system="tfim",
        objective="Optimize TFIM ansatz.",
    )
    from core.model.research_schemas import DecisionRecord

    memory.last_decision = DecisionRecord(
        decision_id="d1",
        iteration=1,
        hypothesis_id="h1",
        action_id="a1",
        decision="discard",
        summary="Execution failed.",
        evidence_against=["No usable metrics."],
        confidence=0.7,
        failure_type="execution_failure",
    )
    hypothesis = policy.propose_hypothesis(memory, controller, iteration=2, system_dir="experiments/tfim")
    action = policy.plan_next_action(
        memory,
        controller,
        hypothesis=hypothesis,
        system_dir="experiments/tfim",
    )

    assert action.action_type == "switch_strategy"
    assert action.strategy_name in {"multidim", "qubit_adapt"}
    assert action.strategy_name != "adapt"


def test_policy_engine_prioritizes_pending_actions():
    policy = PolicyEngine("ga", available_strategies=("ga", "multidim"))
    controller = SearchController()
    memory = ResearchMemory(
        system="lih",
        objective="Optimize LiH ansatz.",
        pending_actions=[
            ActionSpec(
                action_id="queued-reduce",
                hypothesis_id="h-old",
                system_dir="experiments/lih",
                action_type="reduce_search_space",
                strategy_name="ga",
                search_space_patch={"layers": "shrink"},
            )
        ],
    )
    hypothesis = policy.propose_hypothesis(memory, controller, iteration=3, system_dir="experiments/lih")
    action = policy.plan_next_action(
        memory,
        controller,
        hypothesis=hypothesis,
        system_dir="experiments/lih",
    )

    assert action.action_type == "reduce_search_space"
    assert action.budget["from_pending_queue"] is True
    assert action.budget["pending_action_id"] == "queued-reduce"


def test_research_agent_runs_iteration_loop_and_updates_controller(tmp_path):
    session = ResearchSession(str(tmp_path / "experiments" / "lih"), str(tmp_path / "state"))
    calls = []

    class DummyExecutor:
        def execute_action(self, action, iteration, *, session_dir=None, log_path=None, emit=None, emit_stream_line=None):
            calls.append((iteration, action.strategy_name, session_dir, log_path))
            from core.model.research_schemas import RunBundle

            return RunBundle(
                action=action,
                metrics={"energy_error": 0.01, "val_energy": -1.23, "num_params": 8},
                success=True,
            )

    controller = SearchController(max_runs=3)
    agent = ResearchAgent(
        system_dir=str(tmp_path / "experiments" / "lih"),
        strategy="ga",
        session=session,
        controller=controller,
        executor=DummyExecutor(),
        session_dir=str(tmp_path / "state"),
        log_path=str(tmp_path / "state" / "driver.log"),
    )

    history = agent.run_until_stop(start_iteration=1, max_loops=2)

    assert len(history) == 2
    assert calls[0][0] == 1
    assert calls[1][0] == 2
    assert controller.total_runs == 2


def test_research_agent_preserves_policy_action_instead_of_overwriting_it(tmp_path):
    session = ResearchSession(str(tmp_path / "experiments" / "lih"), str(tmp_path / "state"))
    captured = {}

    class DummyPolicy:
        def propose_hypothesis(self, memory, controller, *, iteration, system_dir):
            from core.model.research_schemas import HypothesisSpec

            return HypothesisSpec(
                hypothesis_id="h-custom",
                system="lih",
                objective="Optimize LiH ansatz.",
                statement="Redirect the search region.",
            )

        def plan_next_action(self, memory, controller, *, hypothesis, system_dir, target_error=None):
            from core.model.research_schemas import ActionSpec

            return ActionSpec(
                action_id="custom-action",
                hypothesis_id=hypothesis.hypothesis_id,
                system_dir=system_dir,
                action_type="reduce_search_space",
                strategy_name="ga",
                search_space_patch={"layers": "shrink"},
            )

    class DummyExecutor:
        def execute_action(self, action, iteration, *, session_dir=None, log_path=None, emit=None, emit_stream_line=None):
            captured["action"] = action
            from core.model.research_schemas import RunBundle

            return RunBundle(
                action=action,
                metrics={"energy_error": 0.01, "val_energy": -1.0, "num_params": 6},
                success=True,
            )

    agent = ResearchAgent(
        system_dir=str(tmp_path / "experiments" / "lih"),
        strategy="ga",
        session=session,
        policy_engine=DummyPolicy(),
        executor=DummyExecutor(),
    )

    success, metrics = agent.step(1)

    assert success is True
    assert metrics["energy_error"] == 0.01
    assert captured["action"].action_id == "custom-action"
    assert captured["action"].action_type == "reduce_search_space"


def test_research_agent_updates_strategy_after_switch_action(tmp_path):
    session = ResearchSession(str(tmp_path / "experiments" / "lih"), str(tmp_path / "state"))

    class DummyPolicy:
        def __init__(self):
            self.strategy_name = "ga"

        def update_strategy(self, strategy_name):
            self.strategy_name = strategy_name

        def propose_hypothesis(self, memory, controller, *, iteration, system_dir):
            from core.model.research_schemas import HypothesisSpec

            return HypothesisSpec(
                hypothesis_id="h-switch",
                system="lih",
                objective="Optimize LiH ansatz.",
                statement="Try a different regime.",
            )

        def plan_next_action(self, memory, controller, *, hypothesis, system_dir, target_error=None):
            from core.model.research_schemas import ActionSpec

            return ActionSpec(
                action_id="switch-action",
                hypothesis_id=hypothesis.hypothesis_id,
                system_dir=system_dir,
                action_type="switch_strategy",
                strategy_name="multidim",
            )

    class DummyExecutor:
        def execute_action(self, action, iteration, *, session_dir=None, log_path=None, emit=None, emit_stream_line=None):
            from core.model.research_schemas import RunBundle

            return RunBundle(
                action=action,
                metrics={"energy_error": 0.01, "val_energy": -1.23, "num_params": 8, "selected_strategy": "multidim"},
                success=True,
            )

    policy = DummyPolicy()
    agent = ResearchAgent(
        system_dir=str(tmp_path / "experiments" / "lih"),
        strategy="ga",
        session=session,
        policy_engine=policy,
        executor=DummyExecutor(),
    )

    success, _ = agent.step(1)

    assert success is True
    assert agent.strategy == "multidim"
    assert policy.strategy_name == "multidim"


def test_research_agent_consumes_pending_action_after_execution(tmp_path):
    session = ResearchSession(str(tmp_path / "experiments" / "lih"), str(tmp_path / "state"))
    memory = session.store.load()
    memory.pending_actions = [
        ActionSpec(
            action_id="queued-1",
            hypothesis_id="h0",
            system_dir=str(tmp_path / "experiments" / "lih"),
            action_type="reduce_search_space",
            strategy_name="ga",
            search_space_patch={"layers": "shrink"},
        )
    ]
    session.store.save(memory)

    class DummyPolicy:
        def propose_hypothesis(self, memory, controller, *, iteration, system_dir):
            from core.model.research_schemas import HypothesisSpec

            return HypothesisSpec(
                hypothesis_id="h-next",
                system="lih",
                objective="Optimize LiH ansatz.",
                statement="Use queued follow-up.",
            )

        def plan_next_action(self, memory, controller, *, hypothesis, system_dir, target_error=None):
            queued = memory.pending_actions[0]
            return queued.model_copy(
                update={
                    "action_id": "h-next-reduce_search_space",
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "budget": {"from_pending_queue": True, "pending_action_id": queued.action_id},
                }
            )

    class DummyExecutor:
        def execute_action(self, action, iteration, *, session_dir=None, log_path=None, emit=None, emit_stream_line=None):
            from core.model.research_schemas import RunBundle

            return RunBundle(
                action=action,
                metrics={"energy_error": 0.01, "val_energy": -1.0, "num_params": 6},
                success=True,
            )

    agent = ResearchAgent(
        system_dir=str(tmp_path / "experiments" / "lih"),
        strategy="ga",
        session=session,
        policy_engine=DummyPolicy(),
        executor=DummyExecutor(),
    )

    success, _ = agent.step(1)

    assert success is True
    assert session.store.load().pending_actions == []
