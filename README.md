# Agent-VQE

[中文版](#agent-vqe-中文说明)

Agent-VQE is a configurable framework for automated ansatz search in VQE workflows. It currently targets 4-qubit benchmark systems such as TFIM and LiH, while keeping the architecture ready for larger structured-search experiments.

## What is current as of 2026-03-29

- The `core/` package has been refactored into focused subpackages:
  - `core/foundation/`: environment abstractions
  - `core/representation/`: config-to-circuit compilation and structural edits
  - `core/evaluator/`: training, reporting, logging, experiment folders
  - `core/generator/`: GA, grid, ADAPT, and strategy interfaces
  - `core/molecular/`: reusable molecular Hamiltonian builders, registry, loader, and CLI
  - `core/orchestration/`: controller and multi-strategy orchestration
  - `core/research/`: resumable Agent runtime for outer-loop research
  - `core/warmstart/`: parameter/config mapping for structure changes
- Experiment folders now follow a lower-cognitive-load model:
  - each system has `env.py` and one `run.py`
  - shared execution logic lives in `experiments/shared.py`
  - search/baseline/orchestration are subcommands, not sibling scripts
  - runtime outputs live under `artifacts/`
- The structured experiment log schema in active use is `1.2`.
- The research loop now has a structured Agent runtime built around:
  - `ResearchAgent`
  - `PolicyEngine`
  - `ExperimentExecutor`
  - `ResultInterpreter`
  - `ResearchSession` + `ResearchMemoryStore`

## Repository map

```text
core/
  evaluator/
  foundation/
  generator/
  model/
  molecular/
  orchestration/
  rendering/
  representation/
  research/
  warmstart/
experiments/
  lih/
  tfim/
doc/
tests/
```

## Main workflows

Install dependencies:

```bash
uv sync
```

Quick CLI-only checks:

```bash
uv run python experiments/tfim/run.py --help
uv run python experiments/lih/run.py --help
uv run python experiments/tfim/run.py search ga --help
uv run python experiments/tfim/run.py search multidim --help
uv run python experiments/lih/run.py search ga --help
uv run python experiments/lih/run.py search multidim --help
uv run python experiments/tfim/run.py auto --help
uv run python experiments/lih/run.py auto --help
uv run python core/research/runtime.py --help
uv run python core/molecular/generate.py --list
```

## How To Use The Project

### 1. Run a normal verification experiment

Use this when you want to evaluate one config or the default best-known config for a system.

These commands start real optimization runs.

```bash
uv run python experiments/tfim/run.py
uv run python experiments/lih/run.py
```

Use an explicit config when needed:

```bash
uv run python experiments/tfim/run.py --config experiments/tfim/presets/ga.json --trials 5
uv run python experiments/lih/run.py --config experiments/lih/presets/multidim.json --trials 2
```

### 2. Run structural search

These commands start real search jobs and can take substantial time.

```bash
uv run python experiments/tfim/run.py search ga
uv run python experiments/tfim/run.py search multidim
uv run python experiments/lih/run.py search ga
uv run python experiments/lih/run.py search multidim
```

CLI-only smoke:

```bash
uv run python experiments/tfim/run.py search ga --help
uv run python experiments/lih/run.py search multidim --help
```

### 3. Run orchestrated search demos

These also launch real search/evaluation work rather than a smoke test.

```bash
uv run python experiments/tfim/run.py auto
uv run python experiments/lih/run.py auto
```

CLI-only smoke:

```bash
uv run python experiments/tfim/run.py auto --help
uv run python experiments/lih/run.py auto --help
```

### 4. Run the Agent outer-loop research runtime

Use this when you want the project to keep session memory, resume across runs, and let the Agent manage keep/discard decisions.

```bash
uv run python core/research/runtime.py --dir experiments/lih --strategy ga --target 1e-6 --max 100
uv run python core/research/runtime.py --dir experiments/lih --strategy multidim --target 1e-6 --max 100
uv run python core/research/runtime.py --dir experiments/tfim --strategy ga --target 1e-6 --max 50
```

CLI-only smoke or zero-work check:

```bash
uv run python core/research/runtime.py --help
uv run python core/research/runtime.py --dir experiments/tfim --strategy ga --target 1e-6 --max 0
```

### 5. Generate molecular Hamiltonian datasets from the shared core layer

Use this when you want PySCF/OpenFermion-generated Hamiltonian data without writing a system-specific generation script from scratch.

List registered systems:

```bash
uv run python core/molecular/generate.py --list
```

Generate datasets with the built-in presets:

```bash
uv run python core/molecular/generate.py --system lih
uv run python core/molecular/generate.py --system h2
uv run python core/molecular/generate.py --system beh2
```

Override the scan grid when needed:

```bash
uv run python core/molecular/generate.py --system lih --grid 1.0,1.2,1.4,1.6
uv run python core/molecular/generate.py --system h2 --grid 0.5,0.74,1.0 --out artifacts/molecular/h2_custom.json
```

The default output path is:

```text
artifacts/molecular/<system>_pyscf_data.json
```

CLI-only smoke:

```bash
uv run python core/molecular/generate.py --help
uv run python core/molecular/generate.py --list
```

## How To Use The Agent

The current Agent is not a chat wrapper. It is the structured outer-loop runtime under `core/research/`.

Current runtime flow:

```text
ResearchAgent
  -> PolicyEngine
  -> ExperimentExecutor
  -> ResultInterpreter
  -> ResearchSession / ResearchMemoryStore
```

Relevant files:

- `core/research/agent.py`
- `core/research/policy.py`
- `core/research/executor.py`
- `core/research/interpreter.py`
- `core/research/session.py`
- `core/research/memory_store.py`
- `core/research/runtime.py`

What the Agent currently does:

- reads `research_memory.json`
- chooses a research action
- executes one search step
- interprets results into `DecisionRecord`
- updates `dead_ends`, `strategy_stats`, and best-known performance
- resumes from the last known session pointer

What the Agent currently does not do:

- it does not modify the quantum environment definition
- it does not replace the numerical optimizer
- it does not yet run as a free-form LLM planner; current `PolicyEngine` is rule-based

## Main Commands

Install dependencies:

```bash
uv sync
```

Run verification:

```bash
uv run python experiments/tfim/run.py
uv run python experiments/lih/run.py
```

CLI-only smoke:

```bash
uv run python experiments/tfim/run.py --help
uv run python experiments/lih/run.py --help
```

Run search:

```bash
uv run python experiments/tfim/run.py search ga
uv run python experiments/lih/run.py search multidim
```

CLI-only smoke:

```bash
uv run python experiments/tfim/run.py search ga --help
uv run python experiments/lih/run.py search multidim --help
```

Run Agent runtime:

```bash
uv run python core/research/runtime.py --dir experiments/lih --strategy ga --target 1e-6 --max 100
```

CLI-only smoke:

```bash
uv run python core/research/runtime.py --help
```

Run molecular dataset generation:

```bash
uv run python core/molecular/generate.py --system lih
uv run python core/molecular/generate.py --system h2 --grid 0.5,0.74,1.0
```

CLI-only smoke:

```bash
uv run python core/molecular/generate.py --list
```

Regenerate report assets for an existing run:

```bash
uv run python core/evaluator/render_report.py --run-dir experiments/lih/artifacts/runs/<timestamp>_lih_vqe
uv run python core/evaluator/render_report.py --run-dir experiments/lih/artifacts/runs/<timestamp>_lih_vqe --markdown-only
uv run python core/evaluator/render_report.py --run-dir experiments/lih/artifacts/runs/<timestamp>_lih_vqe --recompute-if-missing
```

CLI-only smoke:

```bash
uv run python core/evaluator/render_report.py --help
```

Run tests:

```bash
just quick-test
just test-all
```

## Output layout

Normal experiment runs are written into timestamped folders created by `prepare_experiment_dir()`:

- `experiments/<system>/artifacts/runs/<timestamp>_<exp_name>/run.log`
- `experiments/<system>/artifacts/runs/<timestamp>_<exp_name>/run.json`
- `experiments/<system>/artifacts/runs/<timestamp>_<exp_name>/events.jsonl`
- `experiments/<system>/artifacts/runs/<timestamp>_<exp_name>/config_snapshot.json` when a concrete ansatz/config snapshot is available

Optional heavy artifacts are no longer produced by default. Markdown reports, circuit images, convergence plots, and circuit JSON are opt-in render outputs.

Each system also maintains a compact append-only index at:

- `experiments/<system>/artifacts/index.jsonl`

Generated molecular Hamiltonian datasets now default to:

- `artifacts/molecular/<system>_pyscf_data.json`

Outer-loop autoresearch sessions are grouped under:

- `experiments/<system>/artifacts/runs/autoresearch/<timestamp>_<strategy>_autoresearch/`

Resume pointers are stored under:

- `experiments/<system>/artifacts/state/current_autoresearch_<strategy>_session`

Structured Agent memory currently lives in:

- `research_memory.json`
- `autoresearch.jsonl`
- `autoresearch.md`

## Core components

- `core/foundation/base_env.py`: immutable quantum-environment base abstraction
- `core/molecular/builders.py`: generic molecular builder spec + geometry factories
- `core/molecular/presets.py`: built-in `H2` / `LiH` / `BeH2` presets
- `core/molecular/registry.py`: shared molecular builder registry
- `core/molecular/generate.py`: CLI entrypoint for dataset generation
- `core/representation/compiler.py`: build ansatz callables from structured configs
- `core/generator/ga.py`: GA-based structural search
- `core/generator/grid.py`: grid-search strategy wrapper
- `core/generator/adapt.py`: ADAPT-style constructive strategy prototype
- `core/orchestration/controller.py`: `SearchController` and `SearchOrchestrator`
- `core/research/runtime.py`: session pointer + resume wiring for the Agent runtime
- `core/research/agent.py`: structured outer-loop research agent
- `core/research/session.py`: structured decision recording and session state updates
- `core/warmstart/config_mapper.py`: config-level warm-start mapping
- `core/warmstart/ansatz_mapper.py`: ansatz-level parameter inheritance
- `core/evaluator/training.py`: optimization loop
- `core/evaluator/report.py`: Markdown report + structured JSONL audit record

## Current experiment defaults

- TFIM verification loads configs in priority order:
  1. explicit `--config`
  2. `presets/ga.json`
  3. `presets/multidim.json`
  4. local fallback config
- LiH verification follows the same priority and also includes a geometry-scan path in `experiments/lih/run.py`.

## Testing

```bash
just quick-test
just test-all
```

Notable coverage areas:

- schemas and typed models
- molecular builder/registry/CLI helpers
- warm-start mapping
- GA/grid search behavior
- orchestration and promotion logic
- report rendering and logging
- ADAPT prototype behavior
- run-script CLI smoke tests

## Related docs

- `Plan.md`: current project status and roadmap
- `program.md`: operating rules for agent-driven research
- `doc/agent_runtime_guide.md`: how the current Agent runtime works and how to run it
- `doc/molecular_hamiltonian_guide.md`: shared molecular builder and dataset generation guide
- `doc/logging_spec.md`: current logging schema
- `doc/experiment_artifact_protocol.md`: current experiment output and retention contract
- `doc/evaluation_protocol.md`: comparison and budget guidelines
- `doc/orchestration_protocol.md`: orchestration contract
- `doc/agent_architecture.md`: architecture and module-level design notes
- `doc/parameter_mapping.md`: warm-start and parameter inheritance rules
- `doc/tfim_100q_hva_strategy.md`: large-scale TFIM notes
- `doc/beyond_template_search.md`: longer-term research direction memo

---

# Agent-VQE 中文说明

Agent-VQE 是一个面向自动 Ansatz 搜索的 VQE 实验框架。当前仓库主要围绕 4-qubit 的 TFIM 与 LiH 基准体系展开，但代码结构已经拆分为可扩展的搜索、评估、编排与 warm-start 子系统。

## 当前状态

- `core/` 已完成分层重构，不再使用早期扁平文件布局。
- `experiments/lih` 与 `experiments/tfim` 已统一为“源码入口 + `artifacts/` 产物目录”的组织方式。
- 每个 system 现在只保留一个 `run.py` 命令面，搜索、基线、编排与扩展工具都作为子命令进入。
- `core/research/` 现在已经不是简单脚本集合，而是结构化 Agent runtime。
- 结构化实验日志当前采用 `schema_version = 1.2`。

## 如何运行项目

1. 安装依赖

```bash
uv sync
```

2. 运行普通验证

```bash
uv run python experiments/tfim/run.py
uv run python experiments/lih/run.py
```

3. 运行结构搜索

```bash
uv run python experiments/tfim/run.py search ga
uv run python experiments/lih/run.py search multidim
```

4. 运行 Agent 外层研究循环

```bash
uv run python core/research/runtime.py --dir experiments/lih --strategy ga --target 1e-6 --max 100
```

5. 运行测试

```bash
just quick-test
just test-all
```

## 如何使用 Agent

当前 Agent 的主链路是：

```text
ResearchAgent
  -> PolicyEngine
  -> ExperimentExecutor
  -> ResultInterpreter
  -> ResearchSession / ResearchMemoryStore
```

主要文件：

- `core/research/agent.py`
- `core/research/policy.py`
- `core/research/executor.py`
- `core/research/interpreter.py`
- `core/research/session.py`
- `core/research/memory_store.py`
- `core/research/runtime.py`

当前 Agent 负责：

- 读取 session memory
- 选择下一轮研究动作
- 执行搜索
- 解释 keep / discard
- 更新 `dead_ends`、`strategy_stats`、最优误差
- 从上一次 session pointer 继续恢复

当前 Agent 还不是：

- 自由聊天式科学家
- 任意修改底层物理定义的系统
- 完全 LLM 驱动的策略器

## 结果产物

常规实验会写入 `experiments/<system>/artifacts/runs/<timestamp>_<exp_name>/`，其中默认包含：

- `run.log`
- `run.json`
- `events.jsonl`
- `config_snapshot.json`（若存在可落盘的结构快照）

可选重文件如 `report_*.md`、线路图、收敛曲线、线路 JSON 不再默认生成。

同时，`experiments/<system>/artifacts/index.jsonl` 会保留体系级的 append-only 轻量索引。

## 项目重点

- 用结构化 `config` 描述 Ansatz，而不是把搜索逻辑写死在单个脚本里
- 统一 GA、Grid、ADAPT、Baseline 的评估口径
- 把实验日志、报告、Git 状态和运行环境一起记录下来，保证可审计
- 为更长周期的自动科研循环保留 session / resume 机制
