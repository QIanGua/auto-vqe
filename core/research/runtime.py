import datetime
import os
from typing import Optional

from core.research.agent import create_default_research_agent, default_emit, run_single_iteration
from core.research.session import ResearchSession

VALID_STRATEGIES = ("ga", "multidim")


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
    if strategy not in VALID_STRATEGIES:
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="System directory (e.g. experiments/lih)")
    parser.add_argument("--strategy", choices=VALID_STRATEGIES, default="ga")
    parser.add_argument("--target", type=float, default=1e-6)
    parser.add_argument("--max", type=int, default=100)
    args = parser.parse_args()

    start_driver_with_strategy(args.dir, args.strategy, args.target, args.max)


if __name__ == "__main__":
    main()
