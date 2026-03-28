# Using `pi-autoresearch` to Re-Optimize Agent-VQE

Date: 2026-03-27

## Executive Summary

`pi-autoresearch` is not primarily a new search algorithm. It is a lightweight "research operating system" for agents:

- persistent experiment session state,
- append-only experiment memory,
- benchmark/check separation,
- automatic keep/discard discipline,
- resume-after-context-loss behavior,
- confidence estimation against noisy metrics.

Agent-VQE already has stronger domain internals than `pi-autoresearch`:

- structured ansatz/config schemas,
- search strategies (GA, Grid, ADAPT),
- budget controller and orchestrator,
- multi-fidelity evaluation,
- rich `results.jsonl` experiment records.

What Agent-VQE lacks is the outer autonomous research loop that turns these components into a long-running, resumable, self-improving agent workflow.

So the best path is:

1. Do **not** replace the current search core.
2. Add a thin **autoresearch-style session layer** above the existing framework.
3. Use that layer to drive repeated proposal -> quick eval -> verification -> keep/discard -> log -> resume cycles.

## What Is Worth Borrowing from `pi-autoresearch`

From the repository and README:

- `autoresearch.md` as the durable "research brain".
- `autoresearch.jsonl` as append-only experiment memory.
- `autoresearch.sh` as a stable benchmark entrypoint.
- optional `autoresearch.checks.sh` as correctness backpressure.
- keep/discard/crash/checks_failed statuses.
- confidence score based on run-to-run noise.
- segment-based resets when the optimization target changes.
- "resume from files, not from chat history" as a core design principle.

These ideas are highly compatible with Agent-VQE because VQE experimentation is:

- iterative,
- noisy,
- multi-stage,
- expensive enough that failed rediscovery is costly.

## What Should Not Be Copied Literally

`pi-autoresearch` is Pi-specific infrastructure. Its extension tools, widget, and runtime integration are not directly reusable here.

Agent-VQE should therefore copy the **protocol**, not the implementation:

- copy the session-file model,
- copy the keep/discard loop,
- copy the noise-aware confidence idea,
- keep Agent-VQE's own engine, controller, schemas, and logs.

In other words, we should build an **Agent-VQE flavored autoresearch layer**, not embed Pi extension code.

## Current Agent-VQE Strengths We Should Preserve

The project already has the right scientific core:

- `core/controller.py`: budget and stop-rule governance.
- `core/engine.py`: unified optimization and structured logging.
- `core/search_algorithms.py`: GA strategy.
- `core/adapt_vqe.py`: constructive growth strategy scaffold.
- `core/schemas.py`: candidate/evaluation/warm-start abstractions.
- `doc/orchestration_protocol.md`: controller/orchestrator contract.
- `results.jsonl`: detailed machine-readable experiment database.

This means the missing piece is not "more search code". The missing piece is:

- session continuity,
- experiment memory optimized for agents,
- explicit proposal rationale,
- replay/resume,
- reliable outer-loop automation.

## Recommended Architecture

### 1. Add a Research Session Layer

Introduce four repo-level files:

- `autoresearch.md`
- `autoresearch.jsonl`
- `autoresearch.sh`
- `autoresearch.checks.sh` (optional)

Purpose:

- `autoresearch.md`
  - objective,
  - systems in scope (`tfim`, `lih`, future systems),
  - current best known configs,
  - dead ends,
  - next hypotheses,
  - "what we learned" across runs.
- `autoresearch.jsonl`
  - compact outer-loop records for each agent decision.
- `autoresearch.sh`
  - one canonical benchmark command for the current research segment.
- `autoresearch.checks.sh`
  - correctness gates such as tests/schema checks.

Important distinction:

- `results.jsonl` remains the **scientific raw record**.
- `autoresearch.jsonl` becomes the **agent decision log**.

Those two logs should coexist, not replace each other.

### 2. Treat Agent-VQE Search as a Two-Level Loop

Inner loop:

- existing `vqe_train`,
- GA / Grid / ADAPT / multi-fidelity evaluation,
- parameter warm-start,
- report generation.

Outer loop:

- choose a search hypothesis,
- write or update the benchmark target,
- run quick benchmark,
- inspect metrics and confidence,
- decide keep/discard,
- optionally promote to medium/full validation,
- update research memory.

This separation is exactly where `pi-autoresearch` is strongest.

### 3. Standardize a Benchmark Contract

Create a stable script contract where `autoresearch.sh` prints machine-readable metrics such as:

```bash
METRIC energy_error=0.00000285
METRIC val_energy=-5.226251
METRIC num_params=32
METRIC two_qubit_gates=12
METRIC runtime_sec=18.4
METRIC stability_std=0.0000007
```

For Agent-VQE, the primary metric should usually be:

- `energy_error` for physics quality.

Secondary metrics should include:

- `num_params`,
- `two_qubit_gates`,
- `runtime_sec`,
- `stability_std`,
- maybe `success_rate`.

Why this matters:

- the current framework logs these metrics after the fact,
- but an outer autoresearch loop needs a single stable command surface.

### 4. Keep Pareto Logic in Agent-VQE, Not in the Shell

Unlike generic optimization tasks, VQE is not truly one-metric.

So:

- benchmark script outputs multiple metrics,
- outer loop uses `energy_error` as the primary score,
- keep/discard policy still consults Pareto logic:
  - meaningful energy improvement wins,
  - near-tie with fewer params/gates also wins,
  - marginal energy gain with much higher complexity should usually discard.

This preserves the existing project philosophy better than collapsing everything into one scalar.

### 5. Add Confidence to Fight Noisy Science

`pi-autoresearch`'s confidence idea is especially valuable here.

For Agent-VQE, confidence should be computed over repeated runs of the same candidate or repeated seeds:

- baseline metric = first valid run in a segment,
- noise floor = MAD or robust std over repeated `energy_error`,
- confidence = `best_improvement / noise_floor`.

Use this to drive behavior:

- low confidence: rerun candidate with more seeds,
- medium confidence: promote to medium fidelity,
- high confidence: keep and update baseline.

This would improve scientific discipline beyond the current "single observed best" style.

### 6. Add Segment Semantics

Segments are useful when the optimization target changes materially, for example:

- TFIM quick-search segment,
- TFIM high-confidence verification segment,
- LiH geometry-transfer segment,
- 100-qubit topology-constrained segment.

Each segment should reset:

- baseline,
- confidence stats,
- current benchmark command,
- active constraints.

This prevents mixing incomparable experiments in one stream.

## Concrete Integration Plan

### Phase A: Minimal, High-Leverage Additions

Add outer-loop protocol only, without touching core search math.

Deliverables:

- `autoresearch.md` template for Agent-VQE,
- `autoresearch.sh` for one target such as TFIM,
- `autoresearch.checks.sh` running the most relevant tests,
- a small Python helper such as `tools/autoresearch_log.py` to append compact session entries.

Behavior:

- baseline run calls existing `experiments/tfim/auto_search.py` or a quicker search entrypoint,
- parse the resulting best metrics,
- log compact agent-facing memory.

This is the safest first step.

### Phase B: Add Resume and Decision Memory

Teach the orchestrator or a thin wrapper to:

- read prior `autoresearch.jsonl`,
- skip already-tried search neighborhoods,
- remember failed ideas,
- remember which config families underperform.

This is where we reduce rediscovery cost.

### Phase C: Connect Confidence to Promotion

Tie session confidence to the existing multi-fidelity machinery:

- low-confidence improvements stay in quick mode,
- medium-confidence improvements go to medium mode,
- high-confidence improvements trigger full verification and report generation.

This is the cleanest bridge between `pi-autoresearch` and the current `SearchOrchestrator`.

### Phase D: Turn Research Ideas into First-Class Objects

Promote proposal rationale from free text into structured records:

- hypothesis,
- edited search dimensions,
- expected mechanism,
- observed outcome,
- failure category,
- follow-up recommendation.

This can live alongside `CandidateSpec.metadata` and outer-loop JSONL entries.

That would make future meta-learning over "which research moves work" much easier.

## Best Initial Workflow for This Repo

If we apply the idea today, the most practical target is:

1. Start with `TFIM`.
2. Use a fast benchmark segment focused on `energy_error`.
3. Let the benchmark script run a short orchestrated search, not just a single training run.
4. Use checks to enforce core tests before keeping a structural change.
5. Record not only metrics, but also:
   - which search dimensions were opened,
   - whether warm-start helped,
   - whether gains survived reseeding,
   - whether complexity grew too much.

That gives us a real autonomous research loop without destabilizing the repo.

## Suggested File Ownership

If we implement this, likely new files should be:

- `doc/pi_autoresearch_integration_plan.md`
- `autoresearch.md`
- `autoresearch.sh`
- `autoresearch.checks.sh`
- `tools/autoresearch_log.py` or `core/research_session.py`

Likely modified files:

- `core/controller.py`
- `core/engine.py`
- `experiments/tfim/auto_search.py`
- `experiments/lih/auto_search.py`
- maybe `doc/orchestration_protocol.md`

## Recommended Order of Actual Engineering Work

1. Add `autoresearch.md` + `autoresearch.sh` for TFIM only.
2. Add compact outer-loop JSONL logging.
3. Add confidence computation from repeated runs.
4. Connect confidence to promotion and strategy switching.
5. Generalize from TFIM to LiH.
6. Only then consider deeper orchestrator refactors.

This order minimizes risk and gives fast feedback.

## Bottom Line

`pi-autoresearch` should be used here as a model for:

- persistence,
- resumability,
- autonomous keep/discard discipline,
- noise-aware benchmarking,
- agent-readable experiment memory.

It should **not** replace Agent-VQE's domain engine.

The right redesign is:

- keep Agent-VQE as the scientific kernel,
- wrap it in an autoresearch-style session protocol,
- let the agent optimize the optimizer over long horizons.

## Sources

- `pi-autoresearch` repository: <https://github.com/davebcn87/pi-autoresearch>
- README: <https://raw.githubusercontent.com/davebcn87/pi-autoresearch/main/README.md>
- extension entry: <https://raw.githubusercontent.com/davebcn87/pi-autoresearch/main/extensions/pi-autoresearch/index.ts>
- skill: <https://raw.githubusercontent.com/davebcn87/pi-autoresearch/main/skills/autoresearch-create/SKILL.md>
