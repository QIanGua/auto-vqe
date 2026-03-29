# LiH Experiment Layout

## Canonical files

- `env.py`: LiH environment and exact-energy references.
- `run.py`: the only user-facing LiH entrypoint. It owns verification,
  `search`, `baseline`, `auto`, `scan`, `plot`, `compare`, and `research-step`.

The shared execution engine lives in
[experiments/shared.py](/Users/qianlong/tries/2026-03-10-auto-vqe/experiments/shared.py),
while LiH-specific policy and utilities stay inline in `run.py`.

## Other folders

- `data/`: checked-in LiH dataset, geometry grid, and dataset generation script.
- `presets/`: promoted best-known configs such as GA and MultiDim winners.
- `notes/`: human-written analysis summaries for those presets.
- `artifacts/`: timestamped runs, reports, caches, and resumable state.
