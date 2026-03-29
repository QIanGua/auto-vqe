import json

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
    memory = ResearchMemory(system="lih", objective="Optimize LiH ansatz.", best_energy_error=0.001, best_num_params=16)
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
                "METRIC energy_error=0.0012\n",
                "METRIC num_params=18\n",
            ]
        )
    ).execute_action(action, 1)

    decision = interpreter.interpret_run(
        iteration=1,
        memory=memory,
        hypothesis=policy_hypothesis(),
        run=run,
    )

    assert decision.decision == "discard"
    assert decision.summary.startswith("Energy:")


def test_executor_reduce_search_space_generates_patched_config_then_verifies(tmp_path):
    system_dir = tmp_path / "experiments" / "tfim"
    system_dir.mkdir(parents=True)
    (system_dir / "run.py").write_text("# test stub\n", encoding="utf-8")
    base_config_dir = system_dir / "presets"
    base_config_dir.mkdir()
    base_config_path = base_config_dir / "ga.json"
    base_config_path.write_text(
        json.dumps(
            {
                "layers": 4,
                "entanglement": "ring",
                "single_qubit_gates": ["ry"],
                "two_qubit_gate": "rzz",
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    class RecordingSubprocess(DummySubprocess):
        def Popen(self, *args, **kwargs):
            captured["cmd"] = args[0]
            captured["env"] = kwargs["env"]
            return DummyProcess(
                [
                    "val_energy         : -1.23\n",
                    "energy_error       : 0.02\n",
                    "num_params         : 6\n",
                ]
            )

    action = ActionSpec(
        action_id="reduce-1",
        hypothesis_id="h1",
        system_dir=str(system_dir),
        action_type="reduce_search_space",
        strategy_name="ga",
        target_candidate_id="cand-base",
        search_space_patch={"layers": "shrink", "entanglement": "simplify"},
        fidelity="quick",
    )
    executor = ExperimentExecutor(subprocess_module=RecordingSubprocess([]))

    run = executor.execute_action(action, 2, session_dir=str(tmp_path / "session"))

    assert run.success is True
    assert captured["cmd"][:4] == ["uv", "run", "python", str(system_dir / "run.py")]
    assert captured["env"]["AGENT_VQE_ACTION_TYPE"] == "reduce_search_space"
    assert captured["env"]["AGENT_VQE_TARGET_CANDIDATE_ID"] == "cand-base"
    assert run.metrics["base_config_path"] == str(base_config_path)
    assert run.target_candidate_id == "cand-base"
    assert run.selected_candidate_id == "cand-base:reduced:0002"
    assert run.selected_config_path is not None
    with open(run.selected_config_path, "r", encoding="utf-8") as f:
        patched = json.load(f)
    assert patched["layers"] == 3
    assert patched["entanglement"] == "linear"


def test_executor_promote_candidate_verifies_selected_config_with_fidelity_trials(tmp_path):
    system_dir = tmp_path / "experiments" / "lih"
    system_dir.mkdir(parents=True)
    (system_dir / "run.py").write_text("# test stub\n", encoding="utf-8")
    config_path = system_dir / "presets" / "multidim.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({"layers": 2, "entanglement": "linear"}), encoding="utf-8")

    captured = {}

    class RecordingSubprocess(DummySubprocess):
        def Popen(self, *args, **kwargs):
            captured["cmd"] = args[0]
            captured["env"] = kwargs["env"]
            return DummyProcess(
                [
                    "val_energy         : -7.88\n",
                    "energy_error       : 0.00042\n",
                    "num_params         : 12\n",
                ]
            )

    action = ActionSpec(
        action_id="promote-1",
        hypothesis_id="h1",
        system_dir=str(system_dir),
        action_type="promote_candidate",
        strategy_name="multidim",
        config_path=str(config_path),
        target_candidate_id="c-1",
        candidate_ids=["c-1"],
        fidelity="full",
    )
    executor = ExperimentExecutor(subprocess_module=RecordingSubprocess([]))

    run = executor.execute_action(action, 5, session_dir=str(tmp_path / "session"))

    assert run.success is True
    assert captured["cmd"][:4] == ["uv", "run", "python", str(system_dir / "run.py")]
    assert captured["cmd"][-2:] == ["--trials", "3"]
    assert captured["env"]["AGENT_VQE_ACTION_TYPE"] == "promote_candidate"
    assert captured["env"]["AGENT_VQE_TARGET_CANDIDATE_ID"] == "c-1"
    assert run.selected_config_path == str(config_path)
    assert run.target_candidate_id == "c-1"
    assert run.selected_candidate_id == "c-1"
    assert run.metrics["promoted_candidate_ids"] == ["c-1"]


def policy_hypothesis():
    from core.model.research_schemas import HypothesisSpec

    return HypothesisSpec(
        hypothesis_id="h1",
        system="lih",
        objective="Optimize LiH ansatz.",
        statement="Establish a first baseline.",
    )
