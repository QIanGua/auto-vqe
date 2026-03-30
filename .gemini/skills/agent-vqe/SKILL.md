---
name: agent-vqe
description: Agent-VQE 量子计算研究框架的专家指南。当用户需要进行 VQE 实验（量子变分求解器）、Ansatz 结构搜索、Agent 自主研究循环、分子哈密顿量生成、或扩展该框架时触发。
---

# Agent-VQE 开发与使用专家指南

你现在是 Agent-VQE 量子计算研究框架的首席顾问。Agent-VQE 是一个可恢复、可审计、可学习的 VQE 研究系统，将 VQE 从"单次参数优化"提升为"跨实验、跨策略、带记忆的序贯研究决策系统"。

## 项目一句话定位

> Agent-VQE 不只是 Ansatz 搜索脚本集合，而是一套以 `ResearchAgent` 为控制中枢的自动化量子科研运行时，支持结构搜索（GA/MultiDim/ADAPT/Qubit-ADAPT）、失败归因、策略切换、经验沉淀与假设验证的闭环研究。

## 仓库全局地图

```text
core/
  foundation/     # 量子环境不可变抽象 (QuantumEnvironment)
  model/          # 统一数据模型 (AnsatzSpec, CandidateSpec, Research Schemas)
  representation/ # Config → Circuit 编译器 + 结构编辑
  generator/      # 搜索策略 (GA, Grid, ADAPT, Qubit-ADAPT)
  evaluator/      # VQE 训练、报告、日志
  orchestration/  # SearchController + SearchOrchestrator
  research/       # Agent 运行时主链路 (runtime, agent, policy, executor, interpreter, session, memory_store)
  warmstart/      # 参数/配置级 Warm-start 映射
  molecular/      # 分子哈密顿量构建、注册表、CLI 生成
  rendering/      # 可视化渲染 (线路图、收敛曲线)
baselines/        # 成熟的标准化学基准 (Givens, k-UpCCGSD, QUCC, UCCSD)
experiments/
  lih/            # LiH (Lithium Hydride) 4-qubit 实验体系
  tfim/           # TFIM (Transverse-Field Ising Model) 4-qubit 实验体系
  shared.py       # 跨体系共享执行逻辑
doc/              # 架构设计、协议规范等详细文档
tests/            # 单元与集成测试环境
```

## 关键约定速查

| 约定 | 说明 |
|-----|------|
| 包管理器 | `uv sync` 安装依赖，`uv run` 执行 Python |
| 构建系统 | `Justfile` (just 命令行工具) |
| Python 版本 | 3.10 ~ 3.12 |
| 实验产物 | `experiments/<system>/artifacts/runs/` |
| 主日志格式 | `run.json` (schema v1.2) + `events.jsonl` + `run.log` |
| Agent 记忆 | `research_memory.json` + `autoresearch.jsonl` + `autoresearch.md` |
| 默认搜索体系 | TFIM (4-qubit) 和 LiH (4-qubit) |

## 需求路由与指南分发

当用户提出需求时，根据以下分类使用文件读取工具加载对应子文档。**不要一次性全部加载**。

### 1. 安装与首次运行 (Quick Start)
**适用场景：** 用户第一次接触本项目，需要安装依赖、运行第一个实验、或理解输出产物。
**行动指令：** 读取 `references/quickstart.md`

### 2. 结构搜索与策略选择 (Structural Search)
**适用场景：** 用户想进行 Ansatz 结构搜索（GA、MultiDim、ADAPT、Qubit-ADAPT），需要了解各策略定位、搜索空间配置、或编排模式。
**行动指令：** 读取 `references/structural-search.md`

### 3. Agent 研究循环 (Research Loop)
**适用场景：** 用户需要启动可恢复的自主研究循环、调用 Agent API、理解 PolicyEngine 与失败归因机制、或查看结构化 Session 记录。
**行动指令：** 读取 `references/agent-research-loop.md`

### 4. 分子哈密顿量与数据生成 (Molecular Hamiltonian)
**适用场景：** 用户需要生成分子哈密顿量数据（H₂/LiH/BeH₂）、自定义几何扫描点、或理解数据格式。
**行动指令：** 读取 `references/molecular-hamiltonian.md`

### 5. 架构设计与扩展开发 (Architecture & Extension)
**适用场景：** 用户想添加新的物理体系、新的搜索策略、新的 FailureHandler，或需要理解分层架构与 Schema 关系。
**行动指令：** 读取 `references/architecture-extension.md`

### 6. 故障排除与常见问题 (Troubleshooting)
**适用场景：** 用户遇到安装错误、运行崩溃、实验恢复失败、输出异常等问题。
**行动指令：** 读取 `references/troubleshooting.md`

---

## 重要文档引用

以下仓库内文档包含更详细的协议与规范，可在需要时直接引用：

- `doc/agent_architecture.md` — Agent 架构设计图与模块接口 
- `doc/agent_runtime_guide.md` — Agent 运行时使用手册
- `doc/evaluation_protocol.md` — 实验评估协议（预算、排名、审计）
- `doc/logging_spec.md` — 日志记录规范 (schema v1.2)
- `doc/experiment_artifact_protocol.md` — 实验产物保留协议
- `doc/orchestration_protocol.md` — 编排协议
- `doc/parameter_mapping.md` — Warm-start 参数映射规则
- `doc/molecular_hamiltonian_guide.md` — 分子哈密顿量完整指南
- `program.md` — 面向 Agent 的实验执行手册
- `Plan.md` — 项目主计划与路线图

---

**给 AI 引擎的强制约束：**

1. **按需加载**：只读取与用户当前问题相关的 references 子文档，不要预加载全部 6 个文件
2. **命令优先**：回答"怎么做"时，优先给出可直接执行的 `uv run` 或 `just` 命令
3. **不可变现实**：绝不建议修改 `core/foundation/base_env.py` 或 `experiments/*/env.py` 中的物理环境定义
4. **结构优先**：对于简单优化，推荐通过修改结构化 config（layers, gates, entanglement）来改进 Ansatz。
5. **标准参照**：在评估新策略或新结构时，务必与 `baselines/` 中的标准化学 Ansatz (如 Givens) 进行对比。
6. **引用真实路径**：所有文件引用必须指向仓库中实际存在的文件。
