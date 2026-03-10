# Auto-VQE (Automatic Variational Quantum Eigensolver)

[中文版](#auto-vqe-自动量子物理学家)

An experiment to let LLMs automatically explore quantum circuit structures (Ansatz) to approximate the ground state energy of the 1D Transverse Field Ising Model (TFIM).

## Overview

The goal of this project is to use AI agents to iteratively design and optimize quantum circuits. The agent explores the "Conjecture" space (`ansatz.py`) while being constrained by "Objective Reality" (`environment.py`).

## Core Components

- **`environment.py` (Objective Reality)**: Defines the physical laws (4-qubit TFIM) and evaluation constraints. This file is **read-only** for the AI agent.
- **`ansatz.py` (The Conjecture)**: The mutable laboratory where the AI agent modifies circuit structures, gate types, and optimizer logic.
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

```bash
# Run the VQE optimization
uv run python ansatz.py
```

---

# Auto-VQE (自动量子物理学家)

[English Version](#auto-vqe-automatic-variational-quantum-eigensolver)

这是一个让 LLM 自动探索量子线路结构（Ansatz）以逼近一维横场伊辛模型 (TFIM) 基态能量的实验。

## 项目概览

本项目旨在通过 AI Agent 迭代设计和优化量子线路。Agent 在“客观现实”（`environment.py`）的约束下，探索“假设空间”（`ansatz.py`）。

## 核心组件

- **`environment.py` (客观现实)**: 定义了物理定律（4 量子比特 TFIM）和评估约束。此文件对 AI Agent 是**只读**的。
- **`ansatz.py` (猜想空间)**: AI Agent 进行实验的实验室，可以自由修改线路结构、门类型和优化器逻辑。
- **`program.md`**: 指导 AI 探索的实验手册和规则。

## 核心原则

- **奥卡姆剃刀原则**: 在能量误差相近的情况下，优先选择更简单的模型（参数更少、线路更浅）。
- **证伪循环**:
  1. 提出新的线路假设。
  2. 运行实验验证假设。
  3. 评估指标（`val_energy`, `num_params`）。
  4. 决定保留或舍弃该改动。

## 快速开始

确保已安装 [uv](https://github.com/astral-sh/uv)。

```bash
# 运行 VQE 优化
uv run python ansatz.py
```
