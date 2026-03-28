# Agent Autoresearch Schema

Date: 2026-03-27

## Purpose

This document defines a **true agent-facing outer research loop** for Agent-VQE.

The goal is not to let an agent merely launch scripts, but to let it control the
highest-value scientific decisions when exhaustive search is impossible, especially
for large-scale systems such as **100-qubit TFIM with MPS**.

This schema is designed to sit **above** the deterministic scientific kernel:

- `core/engine.py`
- `core/controller.py`
- `core/search_algorithms.py`
- `experiments/*/run.py`
- `experiments/*/<strategy>/search.py`

It should become the durable protocol between:

- the deterministic search/evaluation engine,
- the outer research session memory,
- a future rule-based or LLM-based Agent proposer.

## Design Goal

For small systems, brute-force search or semi-brute-force search may still be feasible.
For large systems, it is not.

At 50-100+ qubits, the bottleneck is no longer "how to evaluate one config", but:

- which ansatz family should even be considered,
- which symmetries should be exploited,
- which parameter-sharing regime is physically justified,
- which backend/optimizer is computationally viable,
- which search region is worth spending budget on,
- when to tighten constraints versus relax them.

Therefore, the Agent must control:

1. **system diagnosis**
2. **search regime selection**
3. **proposal-space shaping**
4. **promotion / pruning policy**
5. **failure attribution**
6. **segment transitions**

## Non-Goals

This schema does not define:

- the LLM interface,
- prompt format,
- exact implementation of orchestration code,
- physics kernels,
- circuit factory internals.

Those should remain deterministic and separately testable.

## Core Principle

The inner search kernel should answer:

- "Given this constrained search region, what is the best candidate we found?"

The outer Agent loop should answer:

- "What constrained search region should we explore next, and why?"

That separation is the main reason this schema exists.

## Architecture

The full loop should be understood as a two-level system.

### Inner Loop

The inner loop is deterministic and already mostly exists:

- evaluate one ansatz,
- run GA / Grid / ADAPT / future strategies,
- optimize parameters,
- produce metrics,
- generate reports and artifacts.

### Outer Loop

The outer loop is agentic:

1. diagnose system physics and compute constraints,
2. choose a search regime,
3. generate a proposal,
4. execute a cheap evaluation plan,
5. decide keep / discard / promote / pivot,
6. update research memory,
7. decide the next proposal.

This document specifies the data model for that outer loop.

## Record Model

Each outer-loop step appends one JSON object to `autoresearch.jsonl`.

Recommended record shape:

```json
{
  "schema_version": "2.0",
  "timestamp": "2026-03-27T20:15:00+08:00",
  "iteration": 12,
  "segment_id": "tfim_100q_symmetry_locked_hva",
  "session_id": "20260327_201500_ga_autoresearch",
  "strategy_used": "agent_proposal_loop",
  "system": {},
  "diagnosis": {},
  "proposal": {},
  "evaluation_plan": {},
  "results": {},
  "decision": {},
  "next_action": {},
  "artifacts": {}
}
```

Each top-level block has a separate purpose.

## Top-Level Schema

### `schema_version`

Purpose:

- support future evolution,
- allow migration tooling.

Recommended initial value:

- `"2.0"`

### `timestamp`

ISO-8601 timestamp for the decision event.

### `iteration`

Monotonic outer-loop index inside one session.

### `segment_id`

A segment represents a stable optimization target and research regime.

Examples:

- `tfim_100q_symmetry_locked_hva`
- `tfim_100q_relaxed_tied_hva`
- `lih_4q_uccsd_local_refine`

Segment changes should happen only when the agent believes the current research
assumptions have materially changed.

### `session_id`

Stable identifier for one long-running outer-loop session directory.

### `strategy_used`

This should describe the outer-loop control policy, not just the inner searcher.

Examples:

- `agent_proposal_loop`
- `agent_symmetry_locked_search`
- `agent_local_refinement_loop`

## `system`

Purpose:

- identify the physical problem,
- capture scale,
- record what is assumed stable for this segment.

Suggested schema:

```json
{
  "name": "TFIM_100q",
  "problem_family": "tfim",
  "n_qubits": 100,
  "estimated_scale": "extreme",
  "hamiltonian_tags": ["nearest_neighbor", "uniform_coupling", "lattice_1d"],
  "backend_requirements": ["mps"],
  "notes": "Uniform nearest-neighbor chain."
}
```

Recommended fields:

- `name`
- `problem_family`
- `n_qubits`
- `estimated_scale`: `small`, `medium`, `large`, `extreme`
- `hamiltonian_tags`
- `backend_requirements`
- `notes`

## `diagnosis`

Purpose:

- encode the Agent's current physical diagnosis,
- determine what search families are even plausible.

This is the most important block in large-scale settings.

Suggested schema:

```json
{
  "translational_invariance": true,
  "particle_number_conserving": false,
  "spin_conserving": false,
  "parity_symmetry": true,
  "time_reversal_real_ansatz": true,
  "requires_mps": true,
  "avoid_autograd": true,
  "preferred_optimizer_family": "gradient_free",
  "confidence": 0.92,
  "rationale": "Uniform nearest-neighbor TFIM strongly suggests translational HVA with MPS and gradient-free optimization."
}
```

Recommended fields:

- symmetry booleans:
  - `translational_invariance`
  - `particle_number_conserving`
  - `spin_conserving`
  - `parity_symmetry`
  - `time_reversal_real_ansatz`
- computational flags:
  - `requires_mps`
  - `avoid_autograd`
  - `preferred_optimizer_family`
- `confidence`
- `rationale`

### Diagnostic Rule

For large systems, the Agent must prefer **symmetry diagnosis before proposal generation**.

This is the main lesson from [100q_hva_strategy.md](/Users/qianlong/tries/2026-03-10-auto-vqe/doc/100q_hva_strategy.md):

- a large, symmetric lattice model should not be treated like a generic chemistry ansatz search problem,
- the Agent should first decide whether `translational` sharing is physically justified.

## `proposal`

Purpose:

- define the next search region or action,
- explain why it is chosen,
- clearly separate mutable from frozen assumptions.

Suggested schema:

```json
{
  "proposal_id": "iter12_p1",
  "proposal_type": "search_space_update",
  "family_candidates": ["hva"],
  "search_space": {
    "layers": [2, 3, 4],
    "single_qubit_gates": [["ry"], ["ry", "rx"]],
    "two_qubit_gate": ["rzz"],
    "entanglement": ["linear"],
    "param_strategy": ["translational"],
    "use_mps": [true]
  },
  "frozen_fields": ["problem_family", "use_mps", "param_strategy"],
  "mutable_fields": ["layers", "single_qubit_gates"],
  "expected_mechanism": "Exploit translation symmetry to compress O(N) parameters to O(1).",
  "expected_failure_mode": "May underfit if symmetry lock is too restrictive.",
  "parent_proposal_id": "iter11_p2"
}
```

Recommended fields:

- `proposal_id`
- `proposal_type`
- `family_candidates`
- `search_space`
- `frozen_fields`
- `mutable_fields`
- `expected_mechanism`
- `expected_failure_mode`
- `parent_proposal_id`

## `evaluation_plan`

Purpose:

- define how expensive this proposal is allowed to be,
- encode promotion and rejection criteria.

Suggested schema:

```json
{
  "stage": "quick",
  "backend": "mps",
  "optimizer": "cobyla",
  "seed_count": 1,
  "max_steps": 120,
  "budget": {
    "max_candidates": 16,
    "max_wall_clock_sec": 1800
  },
  "promote_if": {
    "energy_error_better_than_baseline_by": 0.05,
    "runtime_sec_below": 120.0
  },
  "reject_if": {
    "nan_or_failure_rate_above": 0.2,
    "runtime_sec_above": 600.0
  }
}
```

Recommended fields:

- `stage`: `quick`, `medium`, `full`
- `backend`
- `optimizer`
- `seed_count`
- `max_steps`
- `budget`
- `promote_if`
- `reject_if`

### Promotion Semantics

Promotion should not be purely metric-based. It should be **confidence-aware**.

Examples:

- `quick -> medium`
  - if error meaningfully improves,
  - and run is not obviously unstable.
- `medium -> full`
  - if gains survive reseeding,
  - and complexity increase is still justified.

## `results`

Purpose:

- summarize the proposal's observed outcome at the outer-loop level.

Suggested schema:

```json
{
  "val_energy": -125.334747,
  "energy_error": 0.018,
  "num_params": 9,
  "two_qubit_gates": 300,
  "runtime_sec": 57.11,
  "stability_std": null,
  "success_rate": 1.0,
  "failure_mode": null,
  "summary": "Strong energy improvement with extremely low parameter count."
}
```

Recommended fields:

- `val_energy`
- `energy_error`
- `num_params`
- `two_qubit_gates`
- `runtime_sec`
- `stability_std`
- `success_rate`
- `failure_mode`
- `summary`

### Failure Mode Vocabulary

Recommended normalized labels:

- `underexpressive`
- `overparameterized`
- `optimization_stall`
- `barren_plateau_suspected`
- `mps_memory_pressure`
- `symmetry_lock_too_strong`
- `proposal_invalid`
- `backend_unstable`
- `no_pareto_gain`

## `decision`

Purpose:

- capture the scientific keep/discard judgment.

Suggested schema:

```json
{
  "status": "keep",
  "pareto_result": "keep",
  "confidence_score": 0.81,
  "reason": "Large error reduction at only 9 parameters justifies promotion.",
  "baseline_comparison": {
    "previous_best_error": 0.041,
    "previous_best_params": 12
  }
}
```

Allowed `status` values:

- `keep`
- `discard`
- `promote`
- `pivot`
- `crash`
- `checks_failed`

Notes:

- `keep` means acceptable new baseline inside this segment,
- `promote` means worth more budget but not yet baseline,
- `pivot` means current search regime is likely wrong,
- `discard` means no research value at current cost.

## `next_action`

Purpose:

- make the next step explicit,
- reduce ambiguity for a future automated proposer.

Suggested schema:

```json
{
  "action_type": "promote",
  "next_stage": "medium",
  "next_proposal_type": "increase_expressivity",
  "next_hypothesis": "Increase layers from 3 to 4 while keeping translational sharing.",
  "why": "Current result still appears underexpressive rather than unstable."
}
```

Recommended `action_type` values:

- `promote`
- `rerun`
- `increase_expressivity`
- `reduce_cost`
- `relax_symmetry`
- `lock_symmetry`
- `switch_family`
- `end_segment`

## `artifacts`

Purpose:

- provide a durable pointer trail to generated outputs.

Suggested schema:

```json
{
  "session_dir": "experiments/tfim/autoresearch_runs/20260327_201500_ga_autoresearch",
  "proposal_path": "proposals/proposal_iter0012.json",
  "config_path": "ga/best_config_ga.json",
  "report_path": "20260327_201550_tfim_vqe/report_20260327_201621.md",
  "search_log_path": "iterations/iter_0012_ga/ga_search.log",
  "verify_log_path": "iterations/iter_0012_ga/ga_verify.log"
}
```

## Proposal Types

The proposal type should define the category of scientific move being attempted.

Recommended set:

### `lock_symmetry`

Use when the system appears to have strong exploitable structure.

Typical effects:

- force `family=hva`
- force `param_strategy=translational`
- force `use_mps=true`
- prefer gradient-free optimizers

### `relax_symmetry`

Use when a symmetry-locked ansatz seems too restrictive.

Typical effects:

- `translational -> tied`
- retain family,
- widen local flexibility without exploding to `independent`.

### `increase_expressivity`

Use when error remains high but current model is cheap and stable.

Typical actions:

- increase `layers`
- add one more gate type
- enlarge local search neighborhood

### `reduce_cost`

Use when energy is already competitive but complexity is too high.

Typical actions:

- decrease `layers`
- shrink gate family
- restrict entanglement range

### `switch_family`

Use when current ansatz family appears mismatched to the physics.

Examples:

- `hea -> hva`
- `hea -> chemistry-preserving`
- `hva -> tied hea`

This should be considered a major move and used sparingly.

### `backend_pivot`

Use when the numerical method, not the physical prior, is the current bottleneck.

Examples:

- `autograd -> cobyla`
- `statevector -> mps`

### `segment_reset`

Use when the current baseline is no longer comparable.

Examples:

- moving from quick exploration to full validation,
- changing target geometry,
- moving from 4q to 100q,
- switching from local chemistry refinement to symmetry-locked lattice search.

## State Machine

The outer loop should behave like a state machine rather than an unstructured retry loop.

Recommended states:

### `diagnose`

Entry conditions:

- new segment,
- no prior diagnosis,
- major system change.

Output:

- `diagnosis`
- initial `proposal_type`

### `propose`

Construct a bounded search region.

Output:

- `proposal`
- `evaluation_plan`

### `quick_eval`

Run cheap, fast, noisy evaluation.

Transition:

- to `promote`
- to `prune`
- to `pivot`

### `promote`

Escalate the same proposal to more faithful evaluation.

Transition:

- to `medium_eval`
- to `full_eval`

### `prune`

Mark the proposal as exhausted or poor.

Transition:

- back to `propose`
- or to `pivot`

### `pivot`

Used when the current regime appears wrong.

Typical triggers:

- repeated no-improvement,
- repeated numerical failures,
- persistent underfitting despite expressivity increases.

Transition:

- to `diagnose`
- or to `propose` with a different `proposal_type`

### `segment_end`

Used when:

- target reached,
- budget exhausted,
- confidence plateau reached,
- strategy clearly no longer suitable.

## Recommended Transition Rules

### Rule 1: Diagnose before widening

If `n_qubits >= 50`, the Agent should strongly prefer:

- diagnosing symmetry,
- then restricting the search space,

rather than widening a generic ansatz search.

### Rule 2: Prefer expressivity changes before freedom explosion

If a translational HVA appears underexpressive:

1. increase layers,
2. then maybe widen gate family,
3. only then consider `translational -> tied`,
4. avoid jumping directly to `independent`.

### Rule 3: Separate physical failure from numerical failure

Examples:

- poor error with low params may indicate `underexpressive`,
- OOM / severe slowdown indicates `backend_pivot`,
- tiny gradients and no motion may indicate `barren_plateau_suspected`.

### Rule 4: Promotion requires confidence

A candidate should not move to expensive full verification simply because it produced one low error once.

Promotion should require:

- sufficient improvement over baseline,
- acceptable runtime,
- reasonable stability.

### Rule 5: Segment when assumptions change

Do not mix:

- 4q LiH chemistry refinement,
- 100q TFIM translational HVA,
- geometry-transfer chemistry scans,

in one segment baseline.

## Example A: 100q TFIM

This example is directly motivated by [100q_hva_strategy.md](/Users/qianlong/tries/2026-03-10-auto-vqe/doc/100q_hva_strategy.md).

### Diagnosis

```json
{
  "translational_invariance": true,
  "particle_number_conserving": false,
  "spin_conserving": false,
  "parity_symmetry": true,
  "time_reversal_real_ansatz": true,
  "requires_mps": true,
  "avoid_autograd": true,
  "preferred_optimizer_family": "gradient_free",
  "confidence": 0.95,
  "rationale": "Uniform 1D lattice TFIM with 100 qubits strongly favors translational HVA and MPS-based forward-only optimization."
}
```

### Proposal

```json
{
  "proposal_type": "lock_symmetry",
  "family_candidates": ["hva"],
  "search_space": {
    "layers": [2, 3, 4],
    "single_qubit_gates": [["ry"], ["ry", "rx"]],
    "two_qubit_gate": ["rzz"],
    "entanglement": ["linear"],
    "param_strategy": ["translational"],
    "use_mps": [true]
  },
  "frozen_fields": ["param_strategy", "use_mps", "entanglement"],
  "mutable_fields": ["layers", "single_qubit_gates"],
  "expected_mechanism": "Exploit exact or near-exact translation symmetry to compress parameters from O(N) to O(1).",
  "expected_failure_mode": "Insufficient expressivity if the chosen layer count is too low."
}
```

### Evaluation Plan

```json
{
  "stage": "quick",
  "backend": "mps",
  "optimizer": "cobyla",
  "seed_count": 1,
  "max_steps": 100,
  "promote_if": {
    "energy_error_better_than_baseline_by": 0.05
  },
  "reject_if": {
    "runtime_sec_above": 120.0
  }
}
```

### Typical Next Actions

If good:

- `increase_expressivity` via `layers: [4, 5]`

If underfitting:

- `relax_symmetry` from `translational` to `tied`

If MPS still struggles:

- `backend_pivot`

### Key Point

For this system, the Agent's main value is not "search more".
It is "know when to lock the search into the physically correct low-dimensional manifold".

## Example B: LiH

LiH is the opposite kind of example.

It benefits from:

- chemistry-aware structure,
- particle-number and spin constraints,
- local refinement,

but not from translational invariance.

### Diagnosis

```json
{
  "translational_invariance": false,
  "particle_number_conserving": true,
  "spin_conserving": true,
  "parity_symmetry": maybe,
  "time_reversal_real_ansatz": true,
  "requires_mps": false,
  "avoid_autograd": false,
  "preferred_optimizer_family": "gradient_based_or_small_scale_gradient_free",
  "confidence": 0.88,
  "rationale": "Small molecular system with strong local chemistry structure; translational sharing would erase chemically distinct orbital roles."
}
```

### Proposal

```json
{
  "proposal_type": "local_refinement",
  "family_candidates": ["uccsd", "adapt", "ga", "multidim"],
  "search_space": {
    "init_state": ["hf"],
    "layers": [1, 2, 3],
    "single_qubit_gates": [["ry"], ["ry", "rz"]],
    "two_qubit_gate": ["rzz", "cnot"],
    "entanglement": ["linear", "ring"],
    "param_strategy": ["independent", "tied"]
  },
  "frozen_fields": ["translational"],
  "mutable_fields": ["layers", "gate_set", "entanglement", "family"],
  "expected_mechanism": "Find a compact chemistry-respecting ansatz without imposing nonphysical global parameter sharing.",
  "expected_failure_mode": "Local search may overfit shallow neighborhoods or miss chemistry-inspired operator structure."
}
```

### Typical Next Actions

If GA repeatedly rediscovers similar configs:

- switch to `multidim` refinement around known-good neighborhoods

If UCC-style candidates dominate:

- freeze chemistry-inspired structure and only tune lightweight fields

If simple HEA-like configs are competitive:

- `reduce_cost`

### Key Point

For LiH, Agent value is not symmetry locking.
It is:

- selecting the right chemistry-aware search family,
- preventing nonphysical over-sharing,
- tightening local search neighborhoods using memory.

## Comparison: 100q TFIM vs LiH

### 100q TFIM

Agent priority:

- detect translational symmetry,
- force HVA-like parameter sharing,
- choose MPS + gradient-free optimizer,
- avoid brute-force generic ansatz search.

### LiH

Agent priority:

- reject translational sharing,
- preserve chemistry structure,
- manage local refinement over compact neighborhoods,
- compare search families using Pareto logic.

## Minimal Implementation Roadmap

This schema can be implemented incrementally.

### Phase 1: Schema Upgrade

Extend `autoresearch.jsonl` records to include:

- `diagnosis`
- `proposal`
- `evaluation_plan`
- `decision`
- `next_action`
- `failure_mode`

### Phase 2: Proposal Artifact

Persist each proposal to a standalone file:

- `proposals/proposal_iter_0001.json`

This allows:

- replay,
- audit,
- agent-independent debugging.

### Phase 3: Rule-Based Proposer

Implement a deterministic `agent_propose_next_step()`:

- no LLM yet,
- only rule-based diagnosis and transitions.

This is the right first implementation target.

### Phase 4: LLM-Based Proposer

Once schema and transitions stabilize, replace the rule-based proposer with:

- an LLM reading `autoresearch.jsonl`,
- proposal artifacts,
- summary reports.

At that point the LLM will genuinely control the research strategy rather than just launching scripts.

## Bottom Line

A true Agent for large-scale VQE should not be asked:

- "Which random config should I try next?"

It should be asked:

- "Which physically justified low-dimensional search regime should I enter next, and why?"

That is the correct level of agency for:

- 100q TFIM,
- future large lattice models,
- and eventually larger chemistry systems with stronger symmetry-aware constraints.
