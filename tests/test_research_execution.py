from core.model.research_schemas import ActionSpec, ResearchMemory
from core.research.executor import ExperimentExecutor
from core.research.interpreter import ResultInterpreter


class DummyProcess:
    def __init__(self, lines):
        self.stdout = lines

    def wait(self):
        return 0


class DummySubprocess:
    PIPE = object()
    STDOUT = object()

    def __init__(self, lines):
        self.lines = lines

    def Popen(self, *args, **kwargs):
        return DummyProcess(self.lines)


def test_executor_returns_run_bundle_from_legacy_metrics(tmp_path):
    executor = ExperimentExecutor(
        subprocess_module=DummySubprocess(
            [
                "METRIC val_energy=-7.88\n",
                "METRIC energy_error=0.00042\n",
                "METRIC num_params=12\n",
                "METRIC selected_config_path=/tmp/best.json\n",
            ]
        )
    )
    action = ActionSpec(
        action_id="a1",
        hypothesis_id="h1",
        system_dir=str(tmp_path),
        action_type="run_strategy",
        strategy_name="ga",
        fidelity="quick",
    )

    run = executor.execute_action(action, 1)

    assert run.success is True
    assert run.metrics["energy_error"] == 0.00042
    assert run.selected_config_path == "/tmp/best.json"


def test_interpreter_accepts_run_bundle(tmp_path):
    jsonl_path = tmp_path / "autoresearch.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    interpreter = ResultInterpreter(str(jsonl_path))
    memory = ResearchMemory(system="lih", objective="Optimize LiH ansatz.")
    action = ActionSpec(
        action_id="a1",
        hypothesis_id="h1",
        system_dir=str(tmp_path),
        action_type="run_strategy",
        strategy_name="ga",
    )
    run = ExperimentExecutor(
        subprocess_module=DummySubprocess(
            [
                "METRIC val_energy=-7.88\n",
                "METRIC energy_error=0.00042\n",
                "METRIC num_params=12\n",
            ]
        )
    ).execute_action(action, 1)

    decision = interpreter.interpret_run(
        iteration=1,
        memory=memory,
        hypothesis=policy_hypothesis(),
        run=run,
    )

    assert decision.decision == "keep"
    assert decision.summary.startswith("Energy:")


def policy_hypothesis():
    from core.model.research_schemas import HypothesisSpec

    return HypothesisSpec(
        hypothesis_id="h1",
        system="lih",
        objective="Optimize LiH ansatz.",
        statement="Establish a first baseline.",
    )
