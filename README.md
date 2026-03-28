![Agent-VQE Toriyama Edition](assets/article/toriyama_variation_4.png)

# Agent-VQE (Automatic Variational Quantum Eigensolver)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/prisma/prisma)

[中文版](#agent-vqe-自动量子变分求解工程师)

An experiment to let LLMs automatically explore quantum circuit structures (Ansatz) to approximate the ground state energy of quantum systems such as the 1D Transverse Field Ising Model (TFIM) and LiH.

## Overview

Agent-VQE is a **pluggable search framework** designed to let AI agents (LLMs or classical search algorithms) automatically design optimal quantum circuit structures (Ansatz). It bridges the gap between high-level "Conjectures" (ansatz definitions) and the "Objective Reality" of quantum Hamiltonians.

**Key Evolutions:**
- **From Scripts to Framework**: No longer tied to a specific search algorithm; it provides a unified abstraction layer for any structural optimization strategy.
- **Architecture Prepared for Scale**: Built with future high-qubit systems (50-100+) in mind, employing structural config-driven orchestration. Current experimental validation is focused on small-scale systems (4-qubit benchmarks).
- **Auditable Science**: Every run is captured in a structured, searchable JSONL database with automated human-readable reports.

## Core Architecture

- **`core/strategy_base.py` (The Protocol)**: The base interface for all search algorithms. It allows you to plug in anything from Genetic Algorithms to ADAPT-VQE or Reinforcement Learning as a "Strategy".
- **`core/controller.py` (The Brain)**: Manages experiment budgets and stopping rules. Includes the **`SearchOrchestrator`**, which can chain multiple strategies (e.g., broad GA search followed by fine-grained grid scanning).
- **`core/circuit_factory.py` (The Compiler)**: Translates structured JSON configs into executable quantum circuits. Supports **Block & Operator level growth** and hardware-aware metrics.
- **`core/engine.py` (The Valve)**: The unified VQE training loop and multi-fidelity evaluation engine. Decouples candidate evaluation from physical optimization.
- **`core/adapt_vqe.py` (The Sculptor)**: Implements greedy structural growth strategies (ADAPT-VQE protocol) for automated ansatz construction.
- **`doc/logging_spec.md`**: Defines the "Single Source of Truth" schema for experiment records.
- **2026-03-11**: **基础设施堡垒化**。完成了 Phase 3 中的参数映射协议、优化器抽象、以及全面的自动化测试（Schema, Orchestrator, Parameter Mapping）。实现了实验产物与代码的完全解耦。
- **2026-03-11**: 完成 Phase 3 中的规范化与审计增强。制定了 `evaluation_protocol.md` 与 `config_schema.md`，日志升级至 `schema 1.2`。
ion rules (budgets, metrics, ranking).
- **`doc/config_schema.md`**: Formal schema for Ansatz, SearchSpace, and Optimizer configurations.
- **`program.md`**: The experimental protocol and rules guiding the AI's exploration.
- **`baselines/` (Baseline Zoo)**: A standardized set of strong VQE baselines, each exposing `build_ansatz(env, config) -> AnsatzSpec`:
  - `baselines.hea` – hardware-efficient ansatz
  - `baselines.uccsd` – UCCSD-style chemistry-inspired ansatz
  - `baselines.hva` – Hamiltonian-Variational / QAOA-style ansatz
  - `baselines.adapt` – ADAPT-VQE style (proxy, forwards-compatible)
  - `baselines.qubit_adapt` – Qubit-ADAPT-VQE style (proxy, forwards-compatible)

## Key Principles

- **Occam's Razor**: Between two models with similar energy errors, the simpler one (fewer parameters, shallower depth) is preferred.
- **The Refutation Loop**: 
  1. Propose a new circuit hypothesis.
  2. Run the experiment to test the hypothesis.
  3. Evaluate the results (`val_energy`, `num_params`).
  4. Decide to keep or discard the change.

## The Agent Workflow Loop

Agent-VQE is designed for an iterative "Guess & Check" cycle. It's not just a single-shot execution, but a continuous optimization loop:

1.  **Phase 1: Exploration (Conjecture)**
    - Run structural search (GA, Grid, or Orchestrator) to explore the massive ansatz space.
    - Example: `uv run python experiments/tfim/ga_search.py`.
    - Result: The best configuration is automatically saved as **`best_config_ga.json`**.

2.  **Phase 2: Consolidation (Verification)**
    - Run the standardized execution engine. It automatically detects and loads the search results.
    - Example: `uv run python experiments/tfim/run.py`.
    - Benefit: Ensures the found ansatz is stable and performs consistently across different seeds.

3.  **Phase 3: Refinement (Policy Update)**
    - Analyze the automated **Markdown Report** and **Circuit Visualizations**.
    - Based on the energy error and complexity, refine the search space in the code or adjust the strategy for the next iteration.

---

## Getting Started

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

### Install dependencies

```bash
uv sync
```

### Run TFIM experiment

```bash
uv run python experiments/tfim/run.py
```

### Run LiH experiment

```bash
uv run python experiments/lih/run.py
```

Both experiments will:

- write detailed optimization logs to `experiments/<system>/vqe_*.log`
- append summary rows to `experiments/<system>/results.tsv`
- generate human-readable Markdown reports and circuit visualizations in the same directory
- append a **structured experiment record** to `experiments/<system>/results.jsonl`

### Structured Experiment Database (`results.jsonl`)

In addition to the lightweight `results.tsv`, each completed run now writes a
JSONL record capturing the full experiment context:

- `experiment_id`, `timestamp`, `system`, `exp_name`
- `seed`, `n_qubits`
- `ansatz_spec`: a standardized ansatz description dict (compatible with
  `baselines.AnsatzSpec.to_logging_dict()`), including:
  - `name` / `family` (e.g. `"hea"`, `"uccsd"`, `"hva"`, `"ga"`, `"multidim"`)
  - `env_name`, `n_qubits`, `num_params`
  - `config`: the structured config dict used to build the circuit
  - `metadata`: extra tags such as search strategy (`"ga"` / `"multidim"` / `"baseline"`)
- `optimizer_spec`: optimizer + scheduler hyperparameters
- `measurement_spec`: observable and exact energy used for evaluation
- `metrics`: `val_energy`, `energy_error`, `num_params`, `two_qubit_gates`, `runtime_sec`, etc.
- `decision`, `parent_experiment`, `change_summary`
- `config_path_used`: the relative path to the configuration file that produced this record.
- `schema_version`: current version is `1.1` (includes environmental fingerprinting).
- `runtime_env`: captures Python version, OS, and key library versions (TensorCircuit, Torch).
- `git_info`: contains `commit` SHA, `dirty` status, and a short `diff_hash` of the local changes.
- `git_diff`: patch of the current working tree vs. `HEAD`
- `artifact_paths`: paths to report Markdown, circuit PNG/JSON, convergence plots
- `schema_version`: current version is `1.2` (reflecting improved artifact decoupling and parameter mapping).

### Infrastructure & Testing (The "Fortress")

Agent-VQE now includes a comprehensive test suite to protect core behaviors:
- **`tests/test_schemas.py`**: Ensures Pydantic enforcement for all configuration objects.
- **`tests/test_parameter_mapping.py`**: Verifies that structural changes (like adding layers) preserve parameter identity and maintain physical consistency (Warm-start).
- **`tests/test_orchestration.py`**: Validates the `SearchController` budget logic and strategy switching signals.
- **`tests/test_search_algorithms.py`**: Ensures GA and Grid search strategies are robust and converge on mock environments (**81% coverage**).
- **`tests/test_adapt_strategy.py`**: Verifies greedy structural growth and operator pool selection logic.
- **`tests/test_circuit_factory_blocks.py`**: Validates complex gate sets (RXX, RYY, RZZ) and nested block compilation.

**Git Enforcement**: A `pre-push` hook is implemented to ensure `just test` passes before any code is pushed to the repository. Use `just install-hooks` to set it up.

**Artifact Decoupling**: All experiment-generated binary/data files (PNG, JSONL, TSV, Log) are strictly ignored by Git to keep the repository lightweight. Use the `prepare_experiment_dir` utility to ensure your results are saved in isolated, timestamped folders outside the version-controlled source tree.

This JSONL stream can be:

- loaded into pandas / DuckDB for analysis,
- converted into a SQLite / Parquet experiment database,
- used to train meta‑models / surrogate models over the ansatz space.

### Run Evolutionary Ansatz Search (GA)

For intelligent exploration over complex, multi-dimensional ansatz spaces using Genetic Algorithms:

```bash
# TFIM: evolutionary search over layers, gates, and topologies
uv run python experiments/tfim/ga_search.py

# LiH: evolutionary search over initialization, layers, and gate sets
uv run python experiments/lih/ga/search.py
```

The best found architecture is automatically saved to **`ga/best_config_ga.json`** in the same directory. Subsequently, `run.py` will prioritize loading this file for validation.

### Prompting AI Agents (Antigravity / Gemini-CLI / Claude Code / Codex)

You can ask an AI Agent to take over the research loop with a high-level prompt:

> **Example Prompt:**
> "I want to optimize the TFIM ansatz. Your goal is to achieve an energy error below 1e-6. Fully automate the research loop: first run `ga_search.py` to explore, then verify with `run.py`. Analyze the reports and **autonomously decide** the next search dimension (e.g., layers or gates) and re-run the search. Continue this loop independently until the target is met or the budget is exhausted."

### Run Multi-Dimensional Search (Grid Search)

For comprehensive analysis of specific ansatz dimensions (e.g., layers vs topology):
The findings from multi-dimensional search experiments (as documented in the analysis reports in the **`multidim/`** folder) identify the most efficient configurations. These optimal "Ockham's Razor" configurations are saved as **`multidim/best_config_multidim.json`**.

```bash
# LiH: exhaustive scan over a structured ansatz grid
uv run python experiments/lih/multidim/search.py
```

For a resumable outer-loop benchmark that sticks to a single strategy for the entire run:

```bash
# LiH agent loop with fixed GA strategy
uv run python core/research_driver.py --dir experiments/lih --strategy ga --target 1e-6 --max 100

# LiH agent loop with fixed MultiDim strategy
uv run python core/research_driver.py --dir experiments/lih --strategy multidim --target 1e-6 --max 100
```

For long-running background execution, you can launch the same command with `nohup`:

```bash
nohup uv run python core/research_driver.py --dir experiments/lih --strategy ga --target 1e-6 --max 100 >/dev/null 2>&1 &
```

Each long-running outer-loop session is automatically grouped under a dedicated folder such as `experiments/lih/autoresearch_runs/<timestamp>_<strategy>_autoresearch/`, which contains:

- `driver.log`
- `autoresearch.jsonl`
- `autoresearch.md`
- per-iteration logs under `iterations/`
- all search and verification artifacts created during that session

### Run Automated Multi-Strategy Search (The Orchestrator)

For complex workflows involving multiple strategies (e.g., "GA for broad search, then Grid for fine-tuning"), the orchestrator manages the handover and budget:

```bash
# Automatically switch between strategies based on budget and progress
uv run python experiments/tfim/auto_search.py
```

---

---

# Agent-VQE (自动量子变分求解工程师)

[English Version](#agent-vqe-automatic-variational-quantum-eigensolver)

这是一个让 AI Agent 自动探索量子线路结构（Ansatz）以逼近量子体系（例如一维横场伊辛模型 TFIM、LiH）基态能量的实验。

## 项目概览

Agent-VQE 是一个**可插拔的搜索框架**，旨在让 AI Agent（LLM 或经典搜索算法）自动设计最优量子线路结构（Ansatz）。它在“客观现实”（哈密顿量）与“猜想空间”（线路定义）之间建立了一个结构化的迭代反馈环。

**核心演进：**
- **从脚本到框架**：不再绑定于特定算法。提供统一的抽象层，任何结构优化策略（GA、Grid、ADAPT-VQE 等）均可作为插件接入。
- **面向大规模扩展的架构**：架构专为未来 50-100+ 比特的大规模量子系统实验预留，采用配置驱动的策略编排。当前实验验证主要在小规模系统（4-qubit）上进行。
- **可审计的科学实验**：每一轮实验均产生结构化的 JSONL 记录与自动生成的 Markdown 实验报告，确保过程可回溯、结果可复现。

## 核心架构

- **`core/strategy_base.py`（策略协议库）**：定义了所有搜索算法的统一接口 `SearchStrategy`。支持将 GA、BO、ADAPT-VQE、RL 等搜索逻辑作为插件热插拔。
- **`core/controller.py`（决策大脑）**：管理实验预算与停止规则。包含 **`SearchOrchestrator`**（策略编排器），支持多策略链式执行（如先粗搜再精扫）。
- **`core/circuit_factory.py`（线路编译器）**：支持 **Block/Operator 级动态生长**，并内置硬件成本（Gate Count/Depth）估算器。
- **`core/engine.py`（运行基石）**：统一的 VQE 训练引擎，支持多保真（Multi-fidelity）评估逻辑。
- **`core/adapt_vqe.py`（ADAPT 策略）**：实现了基于梯度贪心选择的算符池生长算法。
- **`doc/logging_spec.md`**：定义了本项目统一的结构化实验事件流规范。
- **`doc/evaluation_protocol.md`**：标准评估协议（预算、指标、排名规则）。
- **`doc/config_schema.md`**：Ansatz、搜索空间与优化器的结构化 Schema。
- **`program.md`**：指导 AI 探索的实验手册和规则。
- **`baselines/`（Baseline Zoo）**：一组**标准化基线 ansatz**，统一实现接口 `build_ansatz(env, config) -> AnsatzSpec`，便于：
  - Agent 在“强基线池”上进行探索与对比；
  - 系统性做 ablation 实验；
  - 论文中清晰回答“与哪些基线比较过”。 

## 核心原则

- **奥卡姆剃刀原则**: 在能量误差相近的情况下，优先选择更简单的模型（参数更少、线路更浅）。
- **证伪循环**:
  1. 提出新的线路假设。
  2. 运行实验验证假设。
  3. 评估指标（`val_energy`, `num_params`）。
  4. 决定保留或舍弃该改动。

## Agent 工作流闭环

Agent-VQE 的核心在于其迭代式的“猜想与验证”闭环。通过多个组件的协作，Agent 能够不断逼近最优线路：

1.  **第一阶段：探索 (Exploration / Conjecture)**
    - 使用进化算法、网格搜索或编排器在大规模 Ansatz 空间中进行智能探索。
    - **执行**：`uv run python experiments/tfim/ga_search.py`
    - **产出**：系统会自动将发现的最优结构持久化为 **`best_config_ga.json`**。

2.  **第二阶段：固化 (Consolidation / Verification)**
    - 运行通用执行引擎。引擎会优先检测并自动加载上一步产出的 `best_config.json`。
    - **执行**：`uv run python experiments/tfim/run.py`
    - **价值**：在标准评估下验证该线路的鲁棒性，生成最终的物理结果。

3.  **第三阶段：精炼 (Refinement / Policy Update)**
    - 查阅自动生成的 **Markdown 实验报告** 与 **线路可视化图**。
    - 根据能量误差与线路复杂度（奥卡姆剃刀原则），在代码中缩小搜索范围或调整搜索策略，开启下一轮迭代。

---

## 快速开始

确保已安装 [uv](https://github.com/astral-sh/uv)。

### 安装依赖

```bash
uv sync
```

### 运行 TFIM 实验

```bash
uv run python experiments/tfim/run.py
```

### 运行 LiH 实验

```bash
uv run python experiments/lih/run.py
```

- 在 `experiments/<system>/` 下写入优化日志 `vqe_*.log`
- 追加实验摘要到 `results.tsv`
- 自动生成 Markdown 报告与量子线路可视化图像
- 将完整的实验上下文记录为结构化 JSON 行，写入 `results.jsonl`

### 结构化实验数据库（`results.jsonl`）

除了方便人眼扫描的 `results.tsv` 之外，每次实验结束时还会向
`experiments/<system>/results.jsonl` 追加一条结构化记录，包括：

- `experiment_id`、`timestamp`、`system`、`exp_name`
- `seed`、`n_qubits`
- `ansatz_spec`：统一的 ansatz 描述字典（与 `baselines.AnsatzSpec.to_logging_dict()` 兼容），包括：
  - `name` / `family`（如 `"hea"`、`"uccsd"`、`"hva"`、`"ga"`、`"multidim"`）
  - `env_name`、`n_qubits`、`num_params`
  - `config`：用于构造线路的结构化配置
  - `metadata`：额外标签（例如搜索策略 `"ga"` / `"multidim"` / `"baseline"`）
- `optimizer_spec`：优化器与学习率调度相关超参数
- `measurement_spec`：测量算符与精确能量
- `metrics`：`val_energy`、`energy_error`、`num_params`、`two_qubit_gates`、`runtime_sec`、`actual_steps` 等
- `decision`、`parent_experiment`、`change_summary`
- `config_path_used`：产生此记录的配置文件相对路径
- `schema_version`：当前版本为 `1.1`（包含环境指纹）
- `runtime_env`：捕捉 Python 版本、操作系统、关键库（TensorCircuit, Torch）版本
- `git_info`：包含 `commit` SHA、`dirty` 状态以及本地修改的 `diff_hash`
- `git_diff`：当前工作区相对于 `HEAD` 的补丁 diff
- **`tests/test_search_algorithms.py`**：确保 GA 与网格搜索策略在模拟环境下的鲁棒性（**覆盖率 81%**）。
- **`tests/test_adapt_strategy.py`**：验证贪心生长逻辑与算符池选择的正确性。
- **`tests/test_circuit_factory_blocks.py`**：验证复杂门集（RXX, RYY, RZZ）与嵌套 Block 的编译。

**Git 强制检查**：通过 `pre-push` 钩子确保在 push 代码前必须通过 `just test`。执行 `just install-hooks` 即可完成本地安装。

**产物与源码分离**：所有实验生成的二进制/数据文件（PNG, JSONL, TSV, Log）均被 Git 忽略。使用 `prepare_experiment_dir` 工具确保结果保存在隔离的、带时间戳的目录下。

这使得后续可以：

- 构建真正的“实验 lineage”（通过 `parent_experiment` 串起实验树）；
- 训练 ansatz / optimizer 的 meta‑model 或 surrogate model；
- 直接用 pandas / DuckDB / SQLite 做统计分析和论文绘图。

### 运行遗传算法进化搜索 (GA Search)

使用进化策略在多维空间中智能寻找最优 ansatz：

```bash
# TFIM: 在层数、门类型、拓扑和参数策略维度进行演化
uv run python experiments/tfim/ga_search.py

# LiH: 在初始态、层数、量子门组合等维度进行演化
uv run python experiments/lih/ga/search.py
```

最优配置会自动持久化为该目录下的 **`ga/best_config_ga.json`**。

### 驱动 AI Agent (Antigravity / Gemini-CLI / Claude Code / Codex)

你可以通过一个高层级的指令，让 AI Agent 接管整个科研闭环：

> **提示词示例：**
> “我想优化 TFIM 的 Ansatz，目标是将能量误差降至 1e-6 以下。请**全自动执行**科研闭环：先运行 `ga_search.py` 探索空间，再用 `run.py` 验证。根据生成的报告**自主决定**下一轮的搜索维度（如层数或门类型）并自动启动新一轮搜索。请独立持续迭代，直到达成目标或用尽预算。”

### 运行多维网格搜索 (Multi-Dimensional Search)

用于对特定维度进行穷举分析。通过该策略发现的高效极简配置（遵循奥卡姆剃刀原则）会保存为 **`multidim/best_config_multidim.json`**。

```bash
# LiH: 在统一策略目录下运行 MultiDim 搜索
uv run python experiments/lih/multidim/search.py
```

若要启动可续跑的外层 Agent loop，请显式固定一种策略，避免在同一条长跑任务中混用：

```bash
uv run python core/research_driver.py --dir experiments/lih --strategy ga --target 1e-6 --max 100
uv run python core/research_driver.py --dir experiments/lih --strategy multidim --target 1e-6 --max 100
```

如果要在后台长期运行，可以直接用 `nohup` 启动同样的命令：

```bash
nohup uv run python core/research_driver.py --dir experiments/lih --strategy ga --target 1e-6 --max 100 >/dev/null 2>&1 &
```

每次长跑实验都会自动创建一个独立目录，例如 `experiments/lih/autoresearch_runs/<timestamp>_<strategy>_autoresearch/`，并将以下内容统一收纳进去：

- `driver.log`
- `autoresearch.jsonl`
- `autoresearch.md`
- `iterations/` 下的逐轮日志
- 本次 session 产生的搜索与验证产物

### 运行多策略自动搜索 (Auto-Search Orchestration)

用于执行复杂的策略序列（例如「先用 GA 粗搜，再用网格扫邻域」），编排器会自动监测进度并触发切换：

```bash
# 演示 GA -> Grid 的自动切换流程
uv run python experiments/tfim/auto_search.py
```

**Note**: `run.py` will automatically prioritize loading configurations in this order: `ga/` > `multidim/` > root directory config. We recommend checking the console output for the actual configuration path used.

## Current Status & Limitations

- **Validated Scale**: Most strategies are currently validated on 4-qubit systems (TFIM, LiH).
- **Backend**: Execution currently relies on `tensorcircuit` + `PyTorch` CPU/GPU simulation.
- **Hardware Integration**: Real quantum hardware execution is a roadmap item; current focus is on algorithmic structural search.

## Acknowledgements

- Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
