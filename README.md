# Auto-VQE (Automatic Variational Quantum Eigensolver)

[中文版](#auto-vqe-自动量子物理学家)

An experiment to let LLMs automatically explore quantum circuit structures (Ansatz) to approximate the ground state energy of quantum systems such as the 1D Transverse Field Ising Model (TFIM) and LiH.

## Overview

The goal of this project is to use AI agents to iteratively design and optimize quantum circuits.  
The agent explores the "Conjecture" space (ansatz definitions in `experiments/*/run.py`) while being constrained by "Objective Reality" (Hamiltonians in `experiments/*/env.py` and the shared environment base class in `core/base_env.py`).

## Core Components

- **`core/base_env.py` (Objective Reality base)**: Defines the abstract `QuantumEnvironment` with immutable physical laws (name, qubit count, exact energy).
- **`experiments/tfim/env.py`, `experiments/lih/env.py` (Concrete Objective Reality)**: Implement specific Hamiltonians (4-qubit TFIM, LiH in a Pauli basis). These files are **read-only** for the AI agent.
- **`experiments/tfim/run.py`, `experiments/lih/run.py` (The Conjecture)**: Contain the ansatz construction (`create_circuit`) and experiment loop for each system. This is where the agent can modify circuit structures, gate types, and optimizer logic.
- **`core/engine.py`**: Provides the reusable VQE training loop, logging utilities, and automatic experiment report generation.
- **`program.md`**: The experimental protocol and rules guiding the AI's exploration.

## Key Principles

- **Occam's Razor**: Between two models with similar energy errors, the simpler one (fewer parameters, shallower depth) is preferred.
- **The Refutation Loop**: 
  1. Propose a new circuit hypothesis.
  2. Run the experiment to test the hypothesis.
  3. Evaluate the results (`val_energy`, `num_params`).
  4. Decide to keep or discard the change.

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

### Run simple ansatz search (experimental)

For automatic exploration over a small, discrete ansatz space:

```bash
# TFIM: vary the number of layers
uv run python experiments/tfim/search.py

# LiH: vary the number of layers and whether to use HF initialization
uv run python experiments/lih/search.py
```

These search scripts call a shared `ansatz_search` helper in `core/engine.py` and apply an Occam-style ranking:

- primarily minimize `val_energy`
- for nearly-equal energies (difference < 1e-4), prefer ansatz with fewer parameters (`num_params`)

---

# Auto-VQE (自动量子物理学家)

[English Version](#auto-vqe-automatic-variational-quantum-eigensolver)

这是一个让 LLM 自动探索量子线路结构（Ansatz）以逼近量子体系（例如一维横场伊辛模型 TFIM、LiH）基态能量的实验。

## 项目概览

本项目旨在通过 AI Agent 迭代设计和优化量子线路。  
Agent 在“客观现实”（`core/base_env.py` 以及 `experiments/*/env.py` 中的哈密顿量）的约束下，探索“假设空间”（`experiments/*/run.py` 中的 ansatz 定义）。

## 核心组件

- **`core/base_env.py`（客观现实基类）**：定义抽象的 `QuantumEnvironment`，包含体系名称、量子比特数、精确能量等只读物理信息。
- **`experiments/tfim/env.py`、`experiments/lih/env.py`（具体客观现实）**：实现特定体系的哈密顿量（4 量子比特 TFIM、LiH 的 Pauli 展开）。这些文件对 AI Agent **只读**。
- **`experiments/tfim/run.py`、`experiments/lih/run.py`（猜想空间）**：包含 ansatz 构造函数 `create_circuit` 以及实验主循环，是 AI 可以修改线路结构、门类型和优化策略的主要场所。
- **`core/engine.py`**：提供通用的 VQE 训练循环、日志记录与自动报告生成。
- **`program.md`**：指导 AI 探索的实验手册和规则。

## 核心原则

- **奥卡姆剃刀原则**: 在能量误差相近的情况下，优先选择更简单的模型（参数更少、线路更浅）。
- **证伪循环**:
  1. 提出新的线路假设。
  2. 运行实验验证假设。
  3. 评估指标（`val_energy`, `num_params`）。
  4. 决定保留或舍弃该改动。

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

上述命令会：

- 在 `experiments/<system>/` 下写入优化日志 `vqe_*.log`
- 追加实验摘要到 `results.tsv`
- 自动生成 Markdown 报告与量子线路可视化图像
