import json
from pathlib import Path

from core import research_driver


class DummyProcess:
    def __init__(self, lines):
        self.stdout = lines

    def wait(self):
        return 0


class DummySession:
    def __init__(self, jsonl_path):
        self.jsonl_path = str(jsonl_path)
        self.logged = None
        self.updated = None

    def get_best_performance(self):
        return float("inf")

    def log_decision(self, **kwargs):
        self.logged = kwargs

    def update_brain(self, **kwargs):
        self.updated = kwargs


def test_run_iteration_parses_string_metrics_and_uses_selected_config(tmp_path, monkeypatch):
    system_dir = tmp_path / "experiments" / "lih"
    system_dir.mkdir(parents=True)

    selected_config_path = system_dir / "multidim" / "best_config_multidim.json"
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

    session = DummySession(tmp_path / "autoresearch.jsonl")
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return DummyProcess(lines)

    monkeypatch.setattr(research_driver.subprocess, "Popen", fake_popen)

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    success, metrics = research_driver.run_iteration(
        str(system_dir),
        2,
        session,
        "multidim",
        session_dir=str(session_dir),
    )

    assert success is True
    assert captured["cmd"][-1] == "multidim"
    assert captured["env"]["AGENT_VQE_SESSION_DIR"] == str(session_dir)
    assert captured["env"]["AGENT_VQE_ITERATION"] == "iter_0002"
    assert metrics["selected_strategy"] == "multidim"
    assert metrics["selected_config_path"] == str(selected_config_path)
    assert session.logged is not None
    assert session.logged["results"]["selected_strategy"] == "multidim"
    assert session.updated is not None
    assert session.updated["best_config"] == expected_config


def test_start_driver_resumes_from_latest_iteration(monkeypatch):
    called = []
    tmp_root = Path.cwd() / "tmp_test_research_driver_session"
    tmp_root.mkdir(exist_ok=True)

    class ResumeSession:
        def __init__(self, system_dir, state_dir=None):
            self.system_dir = system_dir
            self.state_dir = state_dir

        def get_best_performance(self):
            return 1.0

        def get_latest_iteration(self):
            return 3

    def fake_run_iteration(system_dir, iteration, session, strategy, session_dir=None, log_path=None):
        called.append((iteration, strategy, session_dir, log_path))
        return False, {}

    monkeypatch.setattr(research_driver, "ResearchSession", ResumeSession)
    monkeypatch.setattr(research_driver, "run_iteration", fake_run_iteration)

    def fake_resolve_session_dir(system_dir, strategy):
        session_dir = tmp_root / f"{strategy}_session"
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir)

    monkeypatch.setattr(research_driver, "resolve_session_dir", fake_resolve_session_dir)

    research_driver.start_driver_with_strategy(
        "experiments/lih",
        strategy="ga",
        target_error=1e-6,
        max_loops=5,
    )

    assert called == [
        (
            4,
            "ga",
            str(tmp_root / "ga_session"),
            str(tmp_root / "ga_session" / "driver.log"),
        )
    ]
