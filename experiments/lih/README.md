# LiH Experiment Layout

## Top-level rule

Keep the top level limited to source-like entrypoints and stable subpackages.
Generated runs, cached external data, and one-off historical outputs live under
`artifacts/` instead of sitting beside the code.

## Core entrypoints

- `run.py`: main LiH verification run using the current best config.
- `env.py`: LiH environment and exact-energy references.

## Stable subfolders

- `data/`: checked-in LiH dataset, geometry grid, and dataset generation script.
- `baseline/`: explicit baseline runs for comparison.
- `ga/`: GA search entrypoint and persisted best config.
- `multidim/`: structured grid-search entrypoint and persisted best config.
- `orchestration/`: higher-level multi-strategy orchestration demos and
  autoresearch entrypoints.

## Auxiliary but useful

- `analysis/`: plotting or convenience wrappers for post-processing scans.
- `scratch/`: one-off comparison scripts that are useful for research but are
  not part of the normal repo workflow.

## Generated artifacts

- `artifacts/runs/`: timestamped experiment runs and autoresearch sessions.
- `artifacts/reports/`: saved reports or curated historical outputs.
- `artifacts/cache/`: external-tool caches such as MindQuantum molecule data.
- `artifacts/state/`: resumable session pointers for autoresearch.
