import subprocess
import os
import json
from typing import Any, Callable, Dict, Optional, Tuple

from core.model.research_schemas import ActionSpec, RunBundle


def parse_metric_value(raw_value: str) -> Any:
    value = raw_value.strip()
    try:
        return float(value)
    except ValueError:
        return value


class ExperimentExecutor:
    """Action dispatcher for research execution."""

    def __init__(self, subprocess_module=subprocess):
        self.subprocess = subprocess_module
        self._handlers: Dict[str, Callable[..., RunBundle]] = {
            "run_strategy": self._execute_run_strategy_action,
            "verify_config": self._execute_verify_config_action,
            "promote_candidate": self._execute_promote_candidate_action,
            "reduce_search_space": self._execute_reduce_search_space_action,
        }

    def _action_dir(self, system_dir: str, session_dir: Optional[str]) -> str:
        base_dir = session_dir or os.path.join(system_dir, "artifacts", "state")
        action_dir = os.path.join(base_dir, "generated_actions")
        os.makedirs(action_dir, exist_ok=True)
        return action_dir

    def _target_candidate_id(self, action: ActionSpec) -> Optional[str]:
        if action.target_candidate_id:
            return action.target_candidate_id
        if action.candidate_ids:
            return action.candidate_ids[0]
        return None

    def _run_process(
        self,
        cmd: list[str],
        *,
        env: Dict[str, str],
        log_path: Optional[str] = None,
        emit=None,
        emit_stream_line=None,
    ) -> Dict[str, Any]:
        process = self.subprocess.Popen(
            cmd,
            stdout=self.subprocess.PIPE,
            stderr=self.subprocess.STDOUT,
            text=True,
            env=env,
        )

        metrics: Dict[str, Any] = {}
        if process.stdout is not None:
            for line in process.stdout:
                if emit_stream_line is not None:
                    emit_stream_line(line, log_path)
                stripped = line.strip()
                if stripped.startswith("METRIC "):
                    payload = stripped[len("METRIC ") :]
                    key, sep, raw_value = payload.partition("=")
                    if sep:
                        metrics[key] = parse_metric_value(raw_value)
                    continue
                key, sep, raw_value = stripped.partition(":")
                if sep and key.strip() in {"val_energy", "energy_error", "num_params", "actual_steps", "training_seconds"}:
                    metrics[key.strip()] = parse_metric_value(raw_value)

        return_code = process.wait()
        return {"metrics": metrics, "return_code": return_code}

    def execute_iteration(
        self,
        system_dir: str,
        strategy: str,
        iteration: int,
        *,
        session_dir: Optional[str] = None,
        log_path: Optional[str] = None,
        emit=None,
        emit_stream_line=None,
    ) -> Tuple[bool, Dict[str, Any]]:
        if emit is not None:
            emit(f"\n>>> Iteration {iteration} Starting... strategy={strategy}", log_path)

        bench_sh = os.path.join(system_dir, "orchestration", "autoresearch.sh")
        env = os.environ.copy()
        if session_dir:
            env["AGENT_VQE_SESSION_DIR"] = session_dir
        env["AGENT_VQE_ITERATION"] = f"iter_{iteration:04d}"
        result = self._run_process(
            ["bash", bench_sh, strategy],
            env=env,
            log_path=log_path,
            emit=emit,
            emit_stream_line=emit_stream_line,
        )
        metrics = result["metrics"]
        success = "energy_error" in metrics
        if not success and emit is not None:
            emit("!!! Benchmark failed to produce energy_error metric.", log_path)
        return success, metrics

    def _default_config_path(self, system_dir: str, strategy_name: Optional[str], session_dir: Optional[str]) -> Optional[str]:
        strategy = strategy_name or "ga"
        candidates = []
        if session_dir:
            candidates.extend(
                [
                    os.path.join(session_dir, strategy, f"best_config_{strategy}.json"),
                    os.path.join(session_dir, "best_config_snapshot.json"),
                ]
            )
        candidates.extend(
            [
                os.path.join(system_dir, strategy, f"best_config_{strategy}.json"),
                os.path.join(system_dir, "best_config_ga.json"),
                os.path.join(system_dir, "best_config_multidim.json"),
                os.path.join(system_dir, "best_config.json"),
            ]
        )
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _verification_trials(self, fidelity: Optional[str]) -> int:
        return {"quick": 1, "medium": 2, "full": 3}.get(fidelity or "medium", 2)

    def _verification_command(self, system_dir: str, config_path: str, fidelity: Optional[str]) -> list[str]:
        run_entry = os.path.join(system_dir, "run.py")
        return ["uv", "run", "python", run_entry, "--config", config_path, "--trials", str(self._verification_trials(fidelity))]

    def _verify_config(
        self,
        action: ActionSpec,
        iteration: int,
        *,
        config_path: str,
        log_path: Optional[str] = None,
        session_dir: Optional[str] = None,
        emit=None,
        emit_stream_line=None,
    ) -> RunBundle:
        if emit is not None:
            emit(f"\n>>> Iteration {iteration} Verifying config... action={action.action_type}", log_path)
        env = os.environ.copy()
        if session_dir:
            env["AGENT_VQE_SESSION_DIR"] = session_dir
        env["AGENT_VQE_ITERATION"] = f"iter_{iteration:04d}"
        env["AGENT_VQE_ACTION_TYPE"] = action.action_type
        env["AGENT_VQE_CONFIG_PATH"] = config_path
        if action.strategy_name:
            env["AGENT_VQE_STRATEGY"] = action.strategy_name
        target_candidate_id = self._target_candidate_id(action)
        if target_candidate_id:
            env["AGENT_VQE_TARGET_CANDIDATE_ID"] = target_candidate_id
        result = self._run_process(
            self._verification_command(action.system_dir, config_path, action.fidelity),
            env=env,
            log_path=log_path,
            emit=emit,
            emit_stream_line=emit_stream_line,
        )
        metrics = result["metrics"]
        metrics.setdefault("selected_config_path", config_path)
        if target_candidate_id:
            metrics.setdefault("target_candidate_id", target_candidate_id)
            metrics.setdefault("selected_candidate_id", target_candidate_id)
        success = "energy_error" in metrics
        return RunBundle(
            action=action,
            metrics=metrics,
            target_candidate_id=target_candidate_id,
            selected_candidate_id=target_candidate_id,
            selected_config_path=config_path,
            success=success,
            artifact_paths={"selected_config_path": config_path},
            error_message=None if success else "Verification failed to produce energy_error metric.",
        )

    def _apply_search_space_patch(self, config: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(config)
        for key, instruction in patch.items():
            if key == "layers" and isinstance(updated.get("layers"), int):
                layers = int(updated["layers"])
                if instruction == "shrink":
                    updated["layers"] = max(1, layers - 1)
                elif instruction == "expand_cautiously":
                    updated["layers"] = layers + 1
                elif isinstance(instruction, int):
                    updated["layers"] = max(1, instruction)
                continue
            if key == "entanglement":
                current = updated.get("entanglement")
                if instruction == "simplify":
                    updated["entanglement"] = "linear" if current != "linear" else current
                elif instruction == "avoid_recent_pattern":
                    updated["entanglement"] = "brick" if current == "linear" else "linear"
                else:
                    updated["entanglement"] = instruction
                continue
            updated[key] = instruction
        return updated

    def _write_generated_config(
        self,
        *,
        action: ActionSpec,
        session_dir: Optional[str],
        config: Dict[str, Any],
        suffix: str,
    ) -> str:
        action_dir = self._action_dir(action.system_dir, session_dir)
        config_path = os.path.join(action_dir, f"{action.action_id}_{suffix}.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return config_path

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _execute_run_strategy_action(
        self,
        action: ActionSpec,
        iteration: int,
        *,
        log_path: Optional[str] = None,
        session_dir: Optional[str] = None,
        emit=None,
        emit_stream_line=None,
    ) -> RunBundle:
        success, metrics = self.execute_iteration(
            action.system_dir,
            action.strategy_name or "ga",
            iteration,
            session_dir=session_dir,
            log_path=log_path,
            emit=emit,
            emit_stream_line=emit_stream_line,
        )
        selected_config_path = metrics.get("selected_config_path")
        selected_candidate_id = metrics.get("selected_candidate_id")
        if not isinstance(selected_candidate_id, str):
            selected_candidate_id = self._target_candidate_id(action)
        return RunBundle(
            action=action,
            metrics=metrics,
            target_candidate_id=self._target_candidate_id(action),
            selected_candidate_id=selected_candidate_id if isinstance(selected_candidate_id, str) else None,
            selected_config_path=selected_config_path if isinstance(selected_config_path, str) else None,
            success=success,
            artifact_paths={"selected_config_path": selected_config_path} if isinstance(selected_config_path, str) else {},
            error_message=None if success else "Benchmark failed to produce energy_error metric.",
        )

    def _execute_verify_config_action(
        self,
        action: ActionSpec,
        iteration: int,
        *,
        log_path: Optional[str] = None,
        session_dir: Optional[str] = None,
        emit=None,
        emit_stream_line=None,
    ) -> RunBundle:
        config_path = action.config_path or self._default_config_path(action.system_dir, action.strategy_name, session_dir)
        if not config_path:
            return RunBundle(
                action=action,
                success=False,
                target_candidate_id=self._target_candidate_id(action),
                error_message="No config available for verification.",
            )
        return self._verify_config(
            action,
            iteration,
            config_path=config_path,
            log_path=log_path,
            session_dir=session_dir,
            emit=emit,
            emit_stream_line=emit_stream_line,
        )

    def _execute_reduce_search_space_action(
        self,
        action: ActionSpec,
        iteration: int,
        *,
        log_path: Optional[str] = None,
        session_dir: Optional[str] = None,
        emit=None,
        emit_stream_line=None,
    ) -> RunBundle:
        base_config_path = action.config_path or self._default_config_path(action.system_dir, action.strategy_name, session_dir)
        target_candidate_id = self._target_candidate_id(action)
        if not base_config_path:
            return RunBundle(
                action=action,
                success=False,
                target_candidate_id=target_candidate_id,
                error_message="No base config available for search-space reduction.",
            )
        patched_config = self._apply_search_space_patch(self._load_config(base_config_path), action.search_space_patch)
        patched_config_path = self._write_generated_config(
            action=action,
            session_dir=session_dir,
            config=patched_config,
            suffix="reduced_search_space",
        )
        run = self._verify_config(
            action,
            iteration,
            config_path=patched_config_path,
            log_path=log_path,
            session_dir=session_dir,
            emit=emit,
            emit_stream_line=emit_stream_line,
        )
        if target_candidate_id:
            derived_candidate_id = f"{target_candidate_id}:reduced:{iteration:04d}"
            run.selected_candidate_id = derived_candidate_id
            run.metrics["target_candidate_id"] = target_candidate_id
            run.metrics["selected_candidate_id"] = derived_candidate_id
        run.artifact_paths["base_config_path"] = base_config_path
        run.artifact_paths["generated_config_path"] = patched_config_path
        run.metrics["base_config_path"] = base_config_path
        run.metrics["selected_config_path"] = patched_config_path
        return run

    def _execute_promote_candidate_action(
        self,
        action: ActionSpec,
        iteration: int,
        *,
        log_path: Optional[str] = None,
        session_dir: Optional[str] = None,
        emit=None,
        emit_stream_line=None,
    ) -> RunBundle:
        config_path = action.config_path or self._default_config_path(action.system_dir, action.strategy_name, session_dir)
        target_candidate_id = self._target_candidate_id(action)
        if not config_path:
            return RunBundle(
                action=action,
                success=False,
                target_candidate_id=target_candidate_id,
                error_message="No config available for promotion.",
            )
        run = self._verify_config(
            action,
            iteration,
            config_path=config_path,
            log_path=log_path,
            session_dir=session_dir,
            emit=emit,
            emit_stream_line=emit_stream_line,
        )
        if action.candidate_ids:
            run.metrics["promoted_candidate_ids"] = list(action.candidate_ids)
        if target_candidate_id:
            run.metrics["selected_candidate_id"] = target_candidate_id
            run.selected_candidate_id = target_candidate_id
        return run

    def execute_action(
        self,
        action: ActionSpec,
        iteration: int,
        *,
        log_path: Optional[str] = None,
        session_dir: Optional[str] = None,
        emit=None,
        emit_stream_line=None,
    ) -> RunBundle:
        handler = self._handlers.get(action.action_type)
        if handler is None:
            return RunBundle(
                action=action,
                success=False,
                target_candidate_id=self._target_candidate_id(action),
                error_message=f"Unsupported action_type: {action.action_type}",
            )
        return handler(
            action,
            iteration,
            log_path=log_path,
            session_dir=session_dir,
            emit=emit,
            emit_stream_line=emit_stream_line,
        )
