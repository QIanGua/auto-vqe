import json

from core.model.research_schemas import ActionSpec, DecisionRecord, RunBundle
from core.research.memory_store import ResearchMemoryStore
from core.research.session import ResearchSession


def test_research_memory_store_persists_memory_and_jsonl(tmp_path):
    store = ResearchMemoryStore(str(tmp_path / "experiments" / "lih"), str(tmp_path / "state"))
    decision = DecisionRecord(
        decision_id="d1",
        iteration=1,
        hypothesis_id="h1",
        action_id="a1",
        decision="keep",
        summary="Energy improved without adding complexity.",
        evidence_for=["energy_error dropped to 1e-3"],
        confidence=0.8,
        selected_config_path="/tmp/best.json",
    )
    action = ActionSpec(
        action_id="a1",
        hypothesis_id="h1",
        system_dir="/tmp/system",
        action_type="run_strategy",
        strategy_name="ga",
        fidelity="quick",
    )

    store.append_decision(decision)
    updated = store.update_from_decision(
        decision,
        objective="Optimize LiH ansatz.",
        next_recommendations=["Try medium fidelity verification"],
        dead_ends=["Deep ring entanglement with tied params"],
    )
    updated.best_energy_error = 1e-3
    updated.best_num_params = 12
    updated.best_config_path = "/tmp/best.json"
    store.save(updated)

    loaded = store.load()
    assert loaded.objective == "Optimize LiH ansatz."
    assert loaded.dead_ends == ["Deep ring entanglement with tied params"]
    assert loaded.next_recommendations == ["Try medium fidelity verification"]
    assert loaded.best_num_params == 12
    assert store.memory_path.endswith("research_memory.json")
    assert store.jsonl_path.endswith("autoresearch.jsonl")
    assert "Try medium fidelity verification" in store.render_markdown(loaded)

    lines = store.jsonl_path
    with open(lines, "r", encoding="utf-8") as f:
        event = json.loads(f.readline())
    assert event["decision"] == "keep"
    assert event["schema_version"] == "2.0"
    assert action.action_id == "a1"


def test_research_memory_store_tracks_failure_taxonomy(tmp_path):
    store = ResearchMemoryStore(str(tmp_path / "experiments" / "lih"), str(tmp_path / "state"))
    decision = DecisionRecord(
        decision_id="d-fail",
        iteration=2,
        hypothesis_id="h2",
        action_id="a2",
        decision="discard",
        summary="Optimization stalled before finding a better candidate.",
        evidence_against=["No new Pareto point after full optimizer budget."],
        confidence=0.6,
        failure_type="optimizer_stall",
        failure_signals={"actual_steps": 200, "max_steps": 200},
        followup_actions=["switch_strategy", "reduce_search_space"],
    )
    run = RunBundle(
        action=ActionSpec(
            action_id="a2",
            hypothesis_id="h2",
            system_dir=str(tmp_path / "experiments" / "lih"),
            action_type="run_strategy",
            strategy_name="ga",
            fidelity="quick",
        ),
        metrics={"energy_error": 0.02, "num_params": 16},
        selected_config_path="/tmp/current.json",
    )

    updated = store.update_from_decision(
        decision,
        run,
        dead_ends=["ga: optimizer_stall at iteration 2"],
        next_recommendations=["Switch to multidim"],
        strategy_name="ga",
    )

    assert updated.failure_counts["optimizer_stall"] == 1
    assert updated.recent_failure_modes == ["optimizer_stall"]
    assert [action.action_type for action in updated.pending_actions] == ["switch_strategy", "reduce_search_space"]
    assert "optimizer_stall" in store.render_markdown(updated)


def test_research_memory_store_can_consume_pending_action(tmp_path):
    store = ResearchMemoryStore(str(tmp_path / "experiments" / "lih"), str(tmp_path / "state"))
    memory = store.load()
    memory.pending_actions = [
        ActionSpec(
            action_id="queued-1",
            hypothesis_id="h1",
            system_dir=str(tmp_path / "experiments" / "lih"),
            action_type="switch_strategy",
        )
    ]
    store.save(memory)

    updated = store.consume_pending_action("queued-1")

    assert updated.pending_actions == []


def test_research_session_preserves_legacy_files_and_metrics(tmp_path):
    system_dir = tmp_path / "experiments" / "lih"
    state_dir = tmp_path / "state"
    session = ResearchSession(str(system_dir), str(state_dir))

    action_spec = ActionSpec(
        action_id="legacy-action-2",
        hypothesis_id="legacy-hypothesis-2",
        system_dir=str(system_dir),
        action_type="run_strategy",
        strategy_name="ga",
        rationale="ga Search + Run Verification"
    )
    session.record_structured_decision(
        DecisionRecord(
            decision_id="legacy-decision-2",
            iteration=2,
            hypothesis_id="legacy-hypothesis-2",
            action_id="legacy-action-2",
            decision="keep",
            summary="Pareto improvement found.",
            evidence_for=["energy_error=0.0025", "num_params=10"],
            confidence=0.5,
            selected_config_path="/tmp/cfg.json",
        ),
        RunBundle(
            action=action_spec,
            metrics={"energy_error": 0.0025, "num_params": 10},
            selected_config_path="/tmp/cfg.json"
        )
    )
    session.apply_structured_outcome(
        DecisionRecord(
            decision_id="legacy-decision-2",
            iteration=2,
            hypothesis_id="legacy-hypothesis-2",
            action_id="legacy-action-2",
            decision="keep",
            summary="Pareto improvement found.",
        ),
        RunBundle(
            action=action_spec,
            metrics={"energy_error": 0.0025, "num_params": 10},
            selected_config_path="/tmp/cfg.json"
        ),
        objective="Optimize LiH ansatz to below 1e-6.",
        best_config={"layers": 2, "entanglement": "linear"},
        dead_ends=["full entanglement at 4 layers"],
        next_hypotheses=["Compare medium fidelity reruns"],
    )

    assert session.get_latest_iteration() == 2
    assert session.get_best_performance() == 0.0025
    assert state_dir.joinpath("autoresearch.jsonl").exists()
    assert state_dir.joinpath("autoresearch.md").exists()
    assert state_dir.joinpath("research_memory.json").exists()
    assert state_dir.joinpath("best_config_snapshot.json").exists()

    memory = json.loads(state_dir.joinpath("research_memory.json").read_text(encoding="utf-8"))
    assert memory["best_energy_error"] == 0.0025
    assert memory["best_num_params"] == 10
    assert memory["next_recommendations"] == ["Compare medium fidelity reruns"]
    md = state_dir.joinpath("autoresearch.md").read_text(encoding="utf-8")
    assert "Optimize LiH ansatz to below 1e-6." in md


def test_research_session_structured_decision_is_primary_write_path(tmp_path):
    system_dir = tmp_path / "experiments" / "lih"
    state_dir = tmp_path / "state"
    session = ResearchSession(str(system_dir), str(state_dir))
    decision = DecisionRecord(
        decision_id="d-structured",
        iteration=1,
        hypothesis_id="h1",
        action_id="a1",
        decision="keep",
        summary="Structured path accepted the first baseline.",
        evidence_for=["initial baseline established"],
        confidence=0.9,
        selected_candidate_id="cand-1",
        selected_config_path="/tmp/config.json",
    )
    run = RunBundle(
        action=ActionSpec(
            action_id="a1",
            hypothesis_id="h1",
            system_dir=str(system_dir),
            action_type="run_strategy",
            strategy_name="ga",
            target_candidate_id="cand-1",
        ),
        metrics={"energy_error": 0.001, "num_params": 8},
        target_candidate_id="cand-1",
        selected_candidate_id="cand-1",
        selected_config_path="/tmp/config.json",
    )

    session.record_structured_decision(decision, run)
    session.apply_structured_outcome(
        decision,
        run,
        objective="Optimize LiH ansatz.",
        best_config={"layers": 2},
        next_hypotheses=["Verify with medium fidelity"],
        strategy_name="ga",
    )

    with open(session.jsonl_path, "r", encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert record["schema_version"] == "2.0"
    assert record["decision"] == "keep"
    assert "run" in record

    memory = json.loads(state_dir.joinpath("research_memory.json").read_text(encoding="utf-8"))
    assert memory["strategy_stats"]["ga"]["keeps"] == 1
    assert memory["best_num_params"] == 8
    assert memory["best_candidate_id"] == "cand-1"
    assert memory["next_recommendations"] == ["Verify with medium fidelity"]
