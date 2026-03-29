# TFIM Experiment Layout

## Canonical files

- `env.py`: TFIM environment definition.
- `run.py`: the only user-facing TFIM entrypoint. It owns verification,
  `search`, `baseline`, `auto`, `research-step`, and `scale-100q`.

The shared execution engine lives in
[experiments/shared.py](/Users/qianlong/tries/2026-03-10-auto-vqe/experiments/shared.py),
while TFIM-specific policy stays inline in `run.py`.

## Other folders

- `presets/`: promoted best-known configs such as GA and MultiDim winners.
- `notes/`: human-written analysis summaries for those presets.
- `artifacts/`: runtime outputs. New TFIM runs now belong here instead of the
  system top level.
