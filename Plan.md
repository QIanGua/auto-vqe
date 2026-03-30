# Agent-VQE 主计划（Plan）

> 版本：2026-03-29  
> 定位：本文件是仓库的主计划文档，用于统一工程路线图与研究议程，而不只是开发 TODO。  
> 当前核心目标：该仓库已从“自动搜索 ansatz”演进为“可恢复、可审计、可学习的 VQE 研究系统”。

## Changelog

- **2026-03-29**：确认 R1~R4 研究系统特性（Failure taxonomy, Strategy switching, Memory as prior, Hypothesis loop）在 `core/research/` 的重构中已被全量实现并稳定验证，Agent-VQE 自动代理化研究基础框架搭建竣工。
- **2026-03-29**：`core/research/` 已完成单 Agent 外层运行时第一轮收束，形成 `runtime / agent / policy / executor / interpreter / session / memory_store / research_schemas` 主链路。
- **2026-03-29**：`ResearchSession` 已以结构化 `DecisionRecord + RunBundle` 作为主写入路径，`autoresearch.jsonl` 保留兼容视图。
- **2026-03-29**：`AnsatzSpec` 已统一为单一抽象，baseline、generator、evaluator、report 共用 `core/model/schemas.py` 中的同一模型。
- **2026-03-29**：真实 `ADAPT / Qubit-ADAPT` 梯度选算符流程已接入标准搜索入口，成为与 `GA / MultiDim` 并列的实验能力。
- **2026-03-29**：TFIM 参考能量与边界条件已更严格对齐，避免 baseline 与目标哈密顿量不一致。
- **2026-03-29**：实验目录已收束到 `artifacts/` 模型，主记录继续沿用 `run.json` 的 `schema_version = 1.2`。
- **2026-03-11**：完成统一日志、策略接口与编排器的第一轮实现。

## 1. 项目定位 / Why This Repo Exists

Agent-VQE 不再只是一个 baseline zoo，也不再只是 4-qubit TFIM / LiH 的 toy benchmark 仓库。它的长期目标是把 VQE 从“单次参数优化问题”提升为“跨实验、跨策略、带记忆的序贯研究决策问题”。

项目的两条并行主线是：

- **工程主线**：持续把 `AnsatzSpec`、ADAPT/Qubit-ADAPT、research runtime、多保真/warm-start、实验协议与报告体系收口为稳定平台。
- **研究主线**：把 `ResearchAgent -> PolicyEngine -> ExperimentExecutor -> ResultInterpreter -> ResearchSession` 升级为真正的 VQE 研究决策系统。

本仓库要解决的不是“再写一个 ansatz 脚本”，而是如何让结构发现、失败归因、策略切换、预算控制与经验沉淀进入同一套协议。

## 2. 项目边界与基本立场

从 [program.md](/Users/qianlong/tries/2026-03-10-auto-vqe/program.md) 抽象出的基本立场如下：

- 物理问题定义默认属于 **Objective Reality**，不应频繁修改。
- 搜索与改进主要发生在 **The Conjecture**，也就是 ansatz 结构、operator pool、参数继承、预算与策略。
- 默认工作方式不是重写物理环境，而是在固定问题定义下不断优化猜想空间。

当前明确不作为近期主线的事项：

- 不替换底层模拟后端。
- 不把 50-100 qubit 运行作为默认常规验证工作流。
- 不把自由形式 LLM 规划器接入 `PolicyEngine` 作为当前阶段前提。

## 3. 当前系统现状

### 3.1 结构表示与编译

- `core/model/schemas.py` 现在是统一的 `AnsatzSpec / CandidateSpec / EvaluationSpec` 真相源。
- `core/representation/compiler.py` 负责从结构化 config、blocks 和 operator 构造线路，并估算参数量与线路成本。
- `core/representation/edits.py` 负责结构编辑，支持“把 ansatz 当作可生长对象”处理。
- `core/warmstart/` 负责配置级与参数级映射，是结构扩展后的参数迁移基础。

当前判断：

- 结构表示层已经具备支撑“静态 baseline + 构造式搜索 + 自适应增长”的统一抽象。
- `AnsatzSpec` 双轨问题已经解决，后续重点不再是抽象统一，而是让所有策略共享更多行为级协议。

### 3.2 搜索与自适应生长

- `core/generator/ga.py`：GA 搜索
- `core/generator/grid.py`：多维/网格搜索
- `core/generator/adapt.py`：真实 ADAPT / Qubit-ADAPT 梯度选算符流程
- `experiments/lih/run.py` 与 `experiments/tfim/run.py`：提供标准搜索入口

当前判断：

- `GA / MultiDim / ADAPT / Qubit-ADAPT` 已形成同一平台内的多种搜索 regime。
- 当前最有价值的问题不再是“是否支持某种搜索”，而是“何时切换哪种搜索最合理”。

### 3.3 评估、日志与可复现性

- `core/evaluator/training.py`：统一 VQE 优化循环
- `core/evaluator/api.py`：评估包装、多保真辅助、实验目录创建
- `core/evaluator/report.py`：`run.json` 主记录与可选重产物渲染
- `core/evaluator/logging_utils.py`：`events.jsonl` 和文本日志

当前判断：

- 仓库已经具备较强的实验审计能力，能恢复 config、git 状态、运行环境与结果产物。
- 下一步的重点不是“再多记几个字段”，而是让日志显式表达研究过程中的判断依据，例如 operator trace、gradient、promotion reason、warm-start reuse。

### 3.4 研究代理与 session / memory

- `core/research/runtime.py` 是当前标准 outer-loop 入口。
- `core/research/agent.py` 持有单 Agent 外层研究循环。
- `core/research/policy.py` 负责动作选择；当前仍是规则驱动。
- `core/research/executor.py` 是结构化执行层；当前仍有 CLI 兼容层。
- `core/research/interpreter.py` 负责把运行结果解释为研究判断。
- `core/research/session.py` 与 `core/research/memory_store.py` 负责 session、memory 与兼容视图。

从 `doc/agent_runtime_guide.md` 和 `doc/agent_architecture.md` 提炼出的当前事实：

- `core/research/` 已经不是脚本集合，而是结构化 outer-loop runtime。
- `PolicyEngine` 目前仍是规则驱动，不是自由形式 planner。
- `ExperimentExecutor` 仍部分依赖现有 CLI 入口，而不是全量纯 Python API。
- `ResearchSession` 已显著结构化，但与主 `run.json / index.jsonl` 体系仍有语义重叠。

## 4. 设计原则

- **分层优化**：结构搜索与连续参数优化解耦。
- **统一抽象**：baseline、search、report、runtime 必须围绕同一个 ansatz 模型工作。
- **统一审计**：实验记录必须能追溯到 config、代码状态、环境指纹与产物路径。
- **预算优先**：搜索过程由 `SearchController` 和上层策略管理，而不是无限试错。
- **策略可插拔**：GA、Grid、ADAPT、Qubit-ADAPT 及未来方法共享同一策略平面。
- **研究导向**：系统不仅要记录最好结果，还要逐步积累失败模式、切换规律与可迁移经验。
- **文档同步**：`README.md`、`program.md`、协议文档和 `Plan.md` 必须反映当前真实入口和真实能力。

## 5. 最值得继续深入的创新方向

当前最有潜力的主创新点不是“再增加一个 ansatz 家族”，而是把研究代理做成真正的 **VQE 研究决策系统**。

### 5.1 Failure-aware research agent

要解决的问题：

- VQE 研究里昂贵的往往不是单次优化，而是大量失败尝试和重复踩坑。
- 传统自动化脚本通常只保留最优结果，不会显式建模失败类型。

相比普通自动化脚本的新意：

- agent 不只是执行实验，而是把失败解释成结构化知识，例如表达力不足、梯度塌缩、warm-start 失效、optimizer stuck。
- 后续动作选择显式依赖这些失败归因，而不是只看当前最好能量。

对应演进模块：

- `core/research/interpreter.py`
- `core/research/policy.py`
- `core/research/memory_store.py`

### 5.2 Meta-policy over search regimes

要解决的问题：

- 当前仓库已经同时拥有 `GA / MultiDim / ADAPT / Qubit-ADAPT`，但仍主要依赖人工决定何时切换。
- 不同 Hamiltonian、不同阶段、不同预算下，最合适的搜索 regime 很可能不同。

相比普通自动化脚本的新意：

- policy 不是只在某个策略内部调参数，而是在多种 search regime 之间做决策。
- 系统研究的对象不再只是 ansatz 本身，而是“策略切换规律”。

对应演进模块：

- `core/research/policy.py`
- `core/research/executor.py`
- `experiments/shared.py`

### 5.3 Research memory as scientific prior

要解决的问题：

- 实验历史通常只是日志，很少真正变成可复用的先验。
- VQE 的经验常常被留在研究者脑中，而不是沉淀到系统里。

相比普通自动化脚本的新意：

- memory 不只保存最好 config，而是保存“什么类型的问题更适合什么策略、什么 operator 常在早期被选中、什么 warm-start 迁移有效”。
- memory 从运行历史升级为可迁移的科研先验。

对应演进模块：

- `core/research/memory_store.py`
- `core/research/session.py`
- `core/research/research_schemas.py`

### 5.4 Closed-loop hypothesis testing

要解决的问题：

- 当前自动化系统大多只是在“继续尝试”，而不是“提出假设并验证假设”。
- 真正有研究价值的系统应能围绕“为什么有效”组织实验。

相比普通自动化脚本的新意：

- agent 可以围绕某个假设生成最小对照实验，而不是盲目扩展搜索。
- 系统输出的不只是最优结果，还包括被支持或被否定的研究判断。

对应演进模块：

- `core/research/policy.py`
- `core/research/interpreter.py`
- `core/research/session.py`

## 6. 工程路线图

### 6.1 Platform Track

#### P1：统一抽象与标准入口（DONE）

- [x] `AnsatzSpec` 已统一为单一模型
- [x] baseline / generator / evaluator / report 共用统一结构表示
- [x] `GA / MultiDim / ADAPT / Qubit-ADAPT` 已有标准搜索入口
- [x] TFIM / LiH 实验入口已收束到 `experiments/*/run.py`

#### P2：实验协议与可复现性（DONE）

- [x] `run.json + events.jsonl + run.log` 已成为主产物组合
- [x] `schema_version = 1.2` 已稳定用于主记录
- [x] `doc/evaluation_protocol.md`、`doc/logging_spec.md` 已存在
- [x] `artifacts/runs/` 与系统级 `index.jsonl` 已形成约定
- [x] 让 `run.json / index.jsonl / ResearchSession` 的跨文件语义进一步统一

#### P3：结构增长与参数迁移（DONE）

- [x] `core/warmstart/` 已具备配置级与参数级映射
- [x] 结构化 operator / excitation / pauli exponential 已能进入统一编译器
- [x] 把 warm-start 与多保真 promotion 更紧密耦合
- [x] 把报告层显式扩展到 operator trace、gradient、reuse 比例等研究过程指标

### 6.2 Research Track

#### R1：Failure taxonomy（DONE）

- [x] 在 `ResultInterpreter` 中标准化失败类型
- [x] 把失败证据写入 `DecisionRecord / ResearchSession`
- [x] 让 policy 能显式避免重复失败模式

#### R2：Strategy switching（DONE）

- [x] 把 `GA / MultiDim / ADAPT / Qubit-ADAPT / verify` 纳入统一动作空间
- [x] 让 `PolicyEngine` 能根据预算、停滞、失败模式切换 regime
- [x] 让 `ExperimentExecutor` 提供更统一的执行接口，而不是主要通过 CLI 兼容层拼接

#### R3：Memory as prior（DONE）

- [x] 从 session 历史中提取 pattern summary，而不只是追加记录
- [x] 支持“体系特征 -> 策略偏好”的轻量总结
- [x] 为跨 session、跨策略经验复用建立稳定 schema

#### R4：Hypothesis loop（DONE）

- [x] 在研究动作中显式增加 hypothesis / evidence / confidence
- [x] 支持围绕假设生成小规模验证实验
- [x] 让研究代理产出“判断更新”，而不只是最好能量更新

## 7. 近期优先级（未来研究重点：应用与实战）

当前 VQE 研究决策引擎的主体框架搭建与重构工作已完成且已跑通。不再将“自动化脚本功能的延伸”视为最核心要务。本工作正式结束工具化开发期，下一步的重点将转向纯场景实战应用开发：

1. **执行自动化探索**：对 LiH、TFIM 或其他大分子进行真实的高负载外层自主研究过程运行。
2. **比较不同策略收益效益**：使用系统在无监督环境下的输出数据对比不同 regim (GA vs MultiDim vs ADAPT) 下的收敛效果。
3. **检验研究决策系统价值**：收集代理生成的结构化假设（Hypothesis）、失败归因和模式（Memory Prior），检验其在科研探索阶段能够省去的成本或发现的有价值结论。

## 8. 验收标准

- M1：文档中只引用当前存在的模块、入口、策略名与目录。
- M2：`Plan.md` 明确把“研究代理”写成主创新方向，并宣告主功能框架基本竣备。
- M3：`AnsatzSpec` 统一抽象已在计划中被显式反映。
- M4：`ADAPT / Qubit-ADAPT` 已被描述为标准实验入口，而不是原型或占位实现。
- M5：近期任务足够具体，可以直接转成实现工单或 agent 子任务。
- M6：文档叙事与 `README.md`、[program.md](/Users/qianlong/tries/2026-03-10-auto-vqe/program.md)、`doc/agent_runtime_guide.md`、`doc/agent_architecture.md` 不冲突。

## 9. 当前默认实现判断

以下判断作为当前计划的默认前提：

- `core/research/runtime.py` 是标准 outer-loop 入口。
- `PolicyEngine` 当前仍是规则驱动；短期重点不是接自由形式 planner，而是先丰富结构化动作与失败归因。
- `ExperimentExecutor` 仍带有 CLI 兼容层；短期重点是统一接口，而不是强行去 CLI 化。
- `program.md` 继续作为运行手册，`doc/agent_*` 继续作为设计细节文档；`Plan.md` 负责总纲与优先级。

本计划已按 2026-03-29 当前仓库状态更新。
