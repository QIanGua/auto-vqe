# Experiment Artifact Protocol

> Updated: 2026-03-29

This document defines the current artifact contract for `experiments/`.

## 1. Design goals

- Keep the default output small.
- Preserve enough information for audit and later comparison.
- Avoid duplicated truth across TSV, JSONL, Markdown, and image files.
- Keep normal experiment outputs separate from research-runtime state.

## 2. Canonical layout

Each system keeps only:

- `env.py`
- `run.py`
- stable inputs such as `data/`, `presets/`, and `notes/`
- `artifacts/`

## 3. Normal run outputs

A normal run writes to:

```text
experiments/<system>/artifacts/runs/<timestamp>_<exp_name>/
```

Default files:

- `run.log`
  - human-readable execution log
- `run.json`
  - canonical per-run audit record
- `events.jsonl`
  - append-only event stream for lightweight progress/result entries
- `config_snapshot.json`
  - emitted when the run has a concrete config/ansatz snapshot worth preserving

Optional heavy files:

- `report_*.md`
- `circuit_*.png`
- `convergence_*.png`
- `circuit_*.json`

These are no longer default outputs. They should only appear when explicitly requested by code paths that enable rendering.

## 4. System-level index

Each system keeps a single append-only index:

```text
experiments/<system>/artifacts/index.jsonl
```

This is a quick lookup table for runs, not the source of truth. The source of truth for a run is always that run's `run.json`.

## 5. Research runtime state

Research-runtime state is separate from normal experiment outputs.

When present, it belongs under:

```text
experiments/<system>/artifacts/runs/autoresearch/<timestamp>_<strategy>_autoresearch/
experiments/<system>/artifacts/state/
```

Typical state files:

- `research_memory.json`
- `autoresearch.jsonl`
- `autoresearch.md`
- `best_config_snapshot.json`
- `current_autoresearch_<strategy>_session`

If these files are absent, old session directories should not be treated as resumable state.

## 6. Cleanup policy

It is safe to delete historical run directories after:

1. rerunning the same experiment through the current interface
2. verifying that the key results match
3. confirming no active research session depends on the old files

In the current repo policy, we prefer to keep:

- the latest validated run for each system
- the current `index.jsonl`
- live research-runtime state only when the corresponding state files actually exist

## 7. Practical reading order

When investigating a result, read in this order:

1. `run.json`
2. `run.log`
3. `events.jsonl`
4. optional rendered artifacts
