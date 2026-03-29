# Experiments Layout

The canonical mental model is now:

1. `experiments/<system>/env.py`
   Defines the physical system.
2. `experiments/<system>/run.py`
   The only command surface for that system:
   verify, search, baseline, orchestration, and any system-specific utilities.

Shared execution logic lives in [experiments/shared.py](/Users/qianlong/tries/2026-03-10-auto-vqe/experiments/shared.py).

All repo-owned Python and pytest entrypoints should be run with
`PYTHONDONTWRITEBYTECODE=1` so normal development does not produce `.pyc` files
or `__pycache__` noise.

Everything else should be understood as one of three things:

- stable checked-in inputs such as `data/`, `presets/`, and `notes/`
- generated artifacts such as `artifacts/`

The goal is to keep "what can this system do?" in one `run.py` command surface
instead of spreading behavior across sibling scripts and folders.
