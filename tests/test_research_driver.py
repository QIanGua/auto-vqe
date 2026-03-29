import json

from core.research import runtime as research_runtime
from core.research.session import ResearchSession


class DummyProcess:
    def __init__(self, lines):
        self.stdout = lines

    def wait(self):
        return 0


def test_run_iteration_parses_string_metrics_and_uses_selected_config(tmp_path, monkeypatch):
    system_dir = tmp_path / "experiments" / "lih"
    system_dir.mkdir(parents=True)

    selected_config_path = system_dir / "presets" / "multidim.json"
    selected_config_path.parent.mkdir(parents=True)
    expected_config = {
        "init_state": "hf",
        "layers": 2,
        "single_qubit_gates": ["ry", "rz"],
        "two_qubit_gate": "rzz",
        "entanglement": "linear",
    }
    selected_config_path.write_text(json.dumps(expected_config), encoding="utf-8")

    lines = [
        "Running LiH benchmark...\n",
        "METRIC val_energy=-7.88\n",
        "METRIC energy_error=0.00042\n",
        "METRIC num_params=12\n",
        "METRIC selected_strategy=multidim\n",
        f"METRIC selected_config_path={selected_config_path}\n",
    ]

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return DummyProcess(lines)

    monkeypatch.setattr("core.research.agent.subprocess.Popen", fake_popen)

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session = ResearchSession(str(system_dir), str(session_dir))
    success, metrics = research_runtime.run_iteration(
        str(system_dir),
        2,
        session,
        "multidim",
        session_dir=str(session_dir),
    )

    assert success is True
    assert captured["cmd"][-5:] == ["research-step", "--strategy", "multidim", "--verify-trials", "2"]
    assert captured["env"]["AGENT_VQE_SESSION_DIR"] == str(session_dir)
    assert captured["env"]["AGENT_VQE_ITERATION"] == "iter_0002"
    assert metrics["selected_strategy"] == "multidim"
    assert metrics["selected_config_path"] == str(selected_config_path)
    memory = session.store.load()
    assert memory.best_energy_error == 0.00042
    assert memory.strategy_stats["multidim"]["keeps"] == 1
    assert session.jsonl_path is not None


def test_start_driver_resumes_from_latest_iteration(monkeypatch, tmp_path):
    called = {}
    def fake_resolve_session_dir(system_dir, strategy):
        session_dir = tmp_path / f"{strategy}_session"
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir)

    monkeypatch.setattr(research_runtime, "resolve_session_dir", fake_resolve_session_dir)

    class DummyAgent:
        def __init__(self, **kwargs):
            called["kwargs"] = kwargs

        def run_until_stop(self, *, start_iteration, max_loops, target_error, emit):
            called["run"] = (start_iteration, max_loops, target_error, emit)
            return []

    monkeypatch.setattr(research_runtime, "create_default_research_agent", lambda **kwargs: DummyAgent(**kwargs))

    research_runtime.start_driver_with_strategy(
        "experiments/lih",
        strategy="ga",
        target_error=1e-6,
        max_loops=5,
    )

    assert called["kwargs"]["session_dir"] == str(tmp_path / "ga_session")
    assert called["kwargs"]["log_path"] == str(tmp_path / "ga_session" / "driver.log")
    assert called["run"][:3] == (1, 5, 1e-6)


def test_runtime_resolve_session_dir_reuses_existing_pointer(tmp_path):
    system_dir = tmp_path / "experiments" / "lih"
    strategy = "ga"
    state_dir = system_dir / "artifacts" / "state"
    state_dir.mkdir(parents=True)
    existing = tmp_path / "existing_session"
    existing.mkdir()
    pointer = state_dir / f"current_autoresearch_{strategy}_session"
    pointer.write_text(str(existing), encoding="utf-8")

    resolved = research_runtime.resolve_session_dir(str(system_dir), strategy)

    assert resolved == str(existing)
