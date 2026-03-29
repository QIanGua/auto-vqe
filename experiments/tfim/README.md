# TFIM Experiment Layout

## Core entrypoints

- `run.py`: main TFIM verification run using the current best config.
- `env.py`: TFIM environment definition.

## Stable subfolders

- `baseline/`: explicit baseline runs for comparison.
- `ga/`: GA search entrypoint and persisted best config.
- `multidim/`: structured grid-search entrypoint and persisted best config.

## Auxiliary but useful

- `orchestration/`: higher-level multi-strategy orchestration demos.
- `scaling/`: specialized large-scale experiments such as 100-qubit MPS runs.

## Generated artifacts

- `__pycache__/`, timestamped experiment folders, and report outputs are runtime
  artifacts rather than primary scripts.
