import subprocess
import os
from typing import Any, Dict, Optional, Tuple

from core.model.research_schemas import ActionSpec, RunBundle


def parse_metric_value(raw_value: str) -> Any:
    value = raw_value.strip()
    try:
        return float(value)
    except ValueError:
        return value


class ExperimentExecutor:
    """Compatibility executor for the legacy shell-based autoresearch entrypoint."""

    def __init__(self, subprocess_module=subprocess):
        self.subprocess = subprocess_module

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
        process = self.subprocess.Popen(
            ["bash", bench_sh, strategy],
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
                if line.startswith("METRIC"):
                    parts = line.split(" ")[1].split("=")
                    if len(parts) == 2:
                        metrics[parts[0]] = parse_metric_value(parts[1])

        process.wait()
        success = "energy_error" in metrics
        if not success and emit is not None:
            emit("!!! Benchmark failed to produce energy_error metric.", log_path)
        return success, metrics

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
        return RunBundle(
            action=action,
            metrics=metrics,
            selected_config_path=selected_config_path if isinstance(selected_config_path, str) else None,
            success=success,
            error_message=None if success else "Benchmark failed to produce energy_error metric.",
        )
