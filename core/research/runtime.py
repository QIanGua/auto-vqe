import argparse
import datetime
import importlib.util
import os
import sys
from typing import Optional

# Allow `python core/research/runtime.py ...` from the repo root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.research.agent import create_default_research_agent, default_emit, run_single_iteration
from core.research.session import ResearchSession

VALID_STRATEGIES = ("ga", "multidim", "adapt", "qubit_adapt")


def discover_available_strategies(system_dir: str) -> tuple[str, ...]:
    run_path = os.path.join(system_dir, "run.py")
    if not os.path.exists(run_path):
        return VALID_STRATEGIES

    module_name = f"agent_vqe_runtime_manifest_{abs(hash(os.path.abspath(system_dir)))}"
    spec = importlib.util.spec_from_file_location(module_name, run_path)
    if spec is None or spec.loader is None:
        return VALID_STRATEGIES

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    manifest = getattr(module, "MANIFEST", None)
    searches = getattr(manifest, "searches", None)
    if isinstance(searches, dict) and searches:
        return tuple(searches.keys())
    return VALID_STRATEGIES


def session_pointer_path(system_dir: str, strategy: str) -> str:
    state_dir = os.path.join(system_dir, "artifacts", "state")
    os.makedirs(state_dir, exist_ok=True)
    return os.path.join(state_dir, f"current_autoresearch_{strategy}_session")


def create_session_dir(system_dir: str, strategy: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = os.path.join(system_dir, "artifacts", "runs", "autoresearch")
    os.makedirs(session_root, exist_ok=True)
    session_dir = os.path.join(session_root, f"{timestamp}_{strategy}_autoresearch")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def resolve_session_dir(system_dir: str, strategy: str) -> str:
    pointer_path = session_pointer_path(system_dir, strategy)
    if os.path.exists(pointer_path):
        with open(pointer_path, "r", encoding="utf-8") as f:
            existing = f.read().strip()
        if existing and os.path.isdir(existing):
            return existing

    session_dir = create_session_dir(system_dir, strategy)
    with open(pointer_path, "w", encoding="utf-8") as f:
        f.write(session_dir)
    return session_dir


def run_iteration(
    system_dir: str,
    iteration: int,
    session: ResearchSession,
    strategy: str,
    session_dir: Optional[str] = None,
    log_path: Optional[str] = None,
):
    return run_single_iteration(
        system_dir=system_dir,
        iteration=iteration,
        session=session,
        strategy=strategy,
        available_strategies=discover_available_strategies(system_dir),
        log_path=log_path,
        session_dir=session_dir,
        emit=default_emit,
    )


def start_driver_with_strategy(
    system_dir: str,
    strategy: str,
    target_error: float = 1e-6,
    max_loops: int = 100,
):
    available_strategies = discover_available_strategies(system_dir)
    if strategy not in available_strategies:
        raise ValueError(f"Unsupported strategy: {strategy}")

    session_dir = resolve_session_dir(system_dir, strategy)
    log_path = os.path.join(session_dir, "driver.log")
    session = ResearchSession(system_dir, state_dir=session_dir)

    best_so_far = session.get_best_performance()
    if best_so_far < target_error:
        default_emit(f"*** Target already reached in prior runs. Best Energy Error: {best_so_far:.2e}", log_path)
        return

    start_iteration = session.get_latest_iteration() + 1
    if start_iteration > max_loops:
        default_emit(
            f"*** Resume point iteration {start_iteration} exceeds max_loops={max_loops}. Nothing to do.",
            log_path,
        )
        return

    default_emit(f"Session directory: {session_dir}", log_path)
    default_emit(f"Resuming research loop from iteration {start_iteration} (max {max_loops}).", log_path)

    agent = create_default_research_agent(
        system_dir=system_dir,
        strategy=strategy,
        session=session,
        available_strategies=available_strategies,
        log_path=log_path,
        session_dir=session_dir,
        emit=default_emit,
    )
    agent.run_until_stop(
        start_iteration=start_iteration,
        max_loops=max_loops,
        target_error=target_error,
        emit=default_emit,
    )


def start_driver(system_dir: str, target_error: float = 1e-6, max_loops: int = 100):
    start_driver_with_strategy(system_dir, strategy="ga", target_error=target_error, max_loops=max_loops)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="System directory (e.g. experiments/lih)")
    parser.add_argument("--strategy", default="ga", help=f"Research strategy. Known strategies: {', '.join(VALID_STRATEGIES)}")
    parser.add_argument("--target", type=float, default=1e-6)
    parser.add_argument("--max", type=int, default=100)
    args = parser.parse_args()

    start_driver_with_strategy(args.dir, args.strategy, args.target, args.max)


if __name__ == "__main__":
    main()
