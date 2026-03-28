import sys
import os
import subprocess
import json
import datetime
from typing import Dict, Any, List, Optional

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from core.research_session import ResearchSession

VALID_STRATEGIES = ("ga", "multidim")


def _emit(message: str, log_path: Optional[str] = None) -> None:
    print(message)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def _emit_stream_line(line: str, log_path: Optional[str] = None) -> None:
    print(line, end="")
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)


def _session_pointer_path(system_dir: str, strategy: str) -> str:
    return os.path.join(system_dir, f".current_autoresearch_{strategy}_session")


def _create_session_dir(system_dir: str, strategy: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = os.path.join(system_dir, "autoresearch_runs")
    os.makedirs(session_root, exist_ok=True)
    session_dir = os.path.join(session_root, f"{timestamp}_{strategy}_autoresearch")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def resolve_session_dir(system_dir: str, strategy: str) -> str:
    pointer_path = _session_pointer_path(system_dir, strategy)
    if os.path.exists(pointer_path):
        with open(pointer_path, "r", encoding="utf-8") as f:
            existing = f.read().strip()
        if existing and os.path.isdir(existing):
            return existing

    session_dir = _create_session_dir(system_dir, strategy)
    with open(pointer_path, "w", encoding="utf-8") as f:
        f.write(session_dir)
    return session_dir


def _parse_metric_value(raw_value: str):
    value = raw_value.strip()
    try:
        return float(value)
    except ValueError:
        return value


def run_iteration(
    system_dir: str,
    iteration: int,
    session: ResearchSession,
    strategy: str,
    session_dir: Optional[str] = None,
    log_path: Optional[str] = None,
):
    """
    Runs a single research iteration:
    1. Run benchmark (autoresearch.sh)
    2. Parse metrics
    3. Update memory
    4. Propose next step
    """
    _emit(f"\n>>> Iteration {iteration} Starting... strategy={strategy}", log_path)
    
    # Run benchmark
    bench_sh = os.path.join(system_dir, "autoresearch.sh")
    env = os.environ.copy()
    if session_dir:
        env["AGENT_VQE_SESSION_DIR"] = session_dir
    env["AGENT_VQE_ITERATION"] = f"iter_{iteration:04d}"
    process = subprocess.Popen(
        ["bash", bench_sh, strategy],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    
    metrics = {}
    if process.stdout is not None:
        for line in process.stdout:
            _emit_stream_line(line, log_path)
            if line.startswith("METRIC"):
                parts = line.split(" ")[1].split("=")
                if len(parts) == 2:
                    metrics[parts[0]] = _parse_metric_value(parts[1])
                
    process.wait()
    
    if "energy_error" not in metrics:
        _emit("!!! Benchmark failed to produce energy_error metric.", log_path)
        return False, metrics

    # Decide with Pareto Logic
    best_err = session.get_best_performance()
    current_err = metrics["energy_error"]
    current_params = metrics.get("num_params", 999)
    
    # Get best params from previous record if exists
    best_params = 999
    if os.path.exists(session.jsonl_path):
        with open(session.jsonl_path, "r") as f:
            for line in f:
                d = json.loads(line)
                recorded_err = d.get("results", {}).get("energy_error")
                if isinstance(recorded_err, (int, float)) and recorded_err <= best_err:
                    best_params = min(best_params, d["results"].get("num_params", 999))

    # Pareto Improvement: 
    # 1. Significantly better energy (>5% improvement)
    # 2. Slightly better/same energy but fewer parameters
    is_better_energy = current_err < best_err * 0.95
    is_better_efficiency = (current_err < best_err * 1.05) and (current_params < best_params * 0.9)
    
    if is_better_energy or is_better_efficiency or best_err == float('inf'):
        decision = "keep"
    else:
        decision = "discard"
        
    rationale = f"Energy: {current_err:.2e} (vs {best_err:.2e}), Params: {current_params} (vs {best_params}). "
    if is_better_energy: rationale += "Significant energy improvement."
    elif is_better_efficiency: rationale += "Better parameter efficiency."
    else: rationale += "No Pareto improvement."
    
    # Log to memory
    session.log_decision(
        iteration=iteration,
        hypothesis="Continuous optimization of Ansatz space.",
        action=f"{strategy} Search + Run Verification",
        results=metrics,
        decision=decision,
        rationale=rationale
    )
    
    # Update brain (simplified for now)
    if decision == "keep":
        config_path = metrics.get("selected_config_path")
        if not isinstance(config_path, str) or not config_path:
            config_path = os.path.join(system_dir, "ga/best_config_ga.json")
        best_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                best_config = json.load(f)
        
        session.update_brain(
            objective="优化 LiH 的 Ansatz，将能量误差降至 1e-6 以下。",
            best_config=best_config,
            best_energy_error=current_err,
            dead_ends=[],
            next_hypotheses=["继续扩大搜索空间", "尝试增加更多层数"]
        )
    
    return True, metrics

def start_driver(system_dir: str, target_error: float = 1e-6, max_loops: int = 100):
    start_driver_with_strategy(system_dir, strategy="ga", target_error=target_error, max_loops=max_loops)


def start_driver_with_strategy(
    system_dir: str,
    strategy: str,
    target_error: float = 1e-6,
    max_loops: int = 100,
):
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")

    session_dir = resolve_session_dir(system_dir, strategy)
    log_path = os.path.join(session_dir, "driver.log")
    session = ResearchSession(system_dir, state_dir=session_dir)

    best_so_far = session.get_best_performance()
    if best_so_far < target_error:
        _emit(f"*** Target already reached in prior runs. Best Energy Error: {best_so_far:.2e}", log_path)
        return

    start_iteration = session.get_latest_iteration() + 1
    if start_iteration > max_loops:
        _emit(
            f"*** Resume point iteration {start_iteration} exceeds max_loops={max_loops}. "
            "Nothing to do.",
            log_path,
        )
        return

    _emit(f"Session directory: {session_dir}", log_path)
    _emit(f"Resuming research loop from iteration {start_iteration} (max {max_loops}).", log_path)

    for i in range(start_iteration, max_loops + 1):
        success, metrics = run_iteration(
            system_dir,
            i,
            session,
            strategy,
            session_dir=session_dir,
            log_path=log_path,
        )
        if not success:
            break
            
        current_err = metrics.get("energy_error", float('inf'))
        if current_err < target_error:
            _emit(f"*** Target reached! Energy Error: {current_err:.2e}", log_path)
            break
            
        _emit(f"--- Iteration {i} finished. Best so far: {session.get_best_performance():.2e}", log_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="System directory (e.g. experiments/lih)")
    parser.add_argument("--strategy", choices=VALID_STRATEGIES, default="ga")
    parser.add_argument("--target", type=float, default=1e-6)
    parser.add_argument("--max", type=int, default=100)
    args = parser.parse_args()

    start_driver_with_strategy(args.dir, args.strategy, args.target, args.max)
