# Agent-VQE 开发计划（Plan）

> 版本：2026-03-11  
> 目标读者：项目维护者 & 贡献者  
> 目标：把 Agent-VQE 从「4-qubit demo」演进为「可扩展、可插拔、可评估」的自动 Ansatz 搜索框架，并为几十到上百比特的后续实验打基础。

## Changelog

- **2026-03-11**: 完成 Phase 2。实现了 `SearchStrategy` 插件化架构与 `SearchOrchestrator` 编排器，支持基于控制信号的自动策略切换。
- **2026-03-11**: 完成 Phase 1。统一了 `results.jsonl` 日志规范，对齐了 GA、MultiDim 和 Baseline 的输出格式。

---

## 1. 总体目标与范围

- 将线路设计从「手工启发式」系统化为「可优化的问题」：
  - 搜索对象从 Python 代码转为结构化配置（`config dict / AnsatzSpec`）。
  - 把 GA、多维网格搜索（MultiDim）、基线 Zoo 统一在同一评估与日志体系下。
- 面向更大规模体系（几十到上百比特）提前预留策略与架构：
  - 搜索 **结构与超参数**，连续参数交给局部优化器。
  - 支持 ADAPT-VQE/贪心构造等更可扩展的方法。
- 保持「客观现实」不变（`core/base_env.py`, `experiments/*/env.py`），所有改动集中在「猜想空间」与搜索/控制逻辑。

不在本阶段解决的问题：
- 不更改底层模拟后端（继续使用 tensorcircuit + PyTorch）。
- 不直接实现上百比特的真实硬件实验，仅在架构上做好预留。

---

## 2. 当前框架现状（2026-03-11）

代码结构要点：
- `core/base_env.py`：量子环境抽象（客观现实基类），只读。
- `experiments/tfim/env.py`, `experiments/lih/env.py`：TFIM & LiH 具体环境，只读。
- `core/circuit_factory.py`：从结构化 `config` 构造量子线路，已经支持：
  - `layers`, `single_qubit_gates`, `two_qubit_gate`, `entanglement`,
  - `init_state`, `param_strategy` 等。
- `core/engine.py`：
  - `vqe_train`：统一的 VQE 训练循环，现已支持结构化日志（`results.jsonl`）。
  - `GridSearchStrategy`：封装了多维网格搜索（MultiDim）。
  - 报告生成、结果记录（TSV + Markdown + 图像）。
- `core/search_algorithms.py`：
  - `GASearchStrategy`：基于 `circuit_factory` 的 GA 搜索插件。
- `core/controller.py`：
  - `SearchController`：核心预算与停止规则（max runs, wall-clock, no-improvement, failures）。
  - `SearchOrchestrator`：策略编排器，负责多策略（Strategy Chain）的顺序执行与热切换。
- `core/strategy_base.py`：定义了 `SearchStrategy` 抽象接口。
- `experiments/*/run.py`：
  - 已支持从 `ga/best_config_ga.json`、`multidim/best_config_multidim.json` 等加载配置。
  - 与 Baseline/Baseline Zoo 的 `AnsatzSpec` 已经对齐。

现有策略：
- GA：`core/search_algorithms.GASearchStrategy` + `experiments/*/ga_search.py`。
- MultiDim：基于 `core.engine.GridSearchStrategy` 的结构化网格搜索。
- Baseline：`baselines/` 中的固定 ansatz 对照。

---

## 3. 设计原则

- **分层优化**：
  - 全局搜索：GA / MultiDim / 未来的 BO、RL 等主要针对「结构与超参数」。
  - 局部优化：`vqe_train` 对连续参数做梯度/基于采样的局部优化。
- **统一记录 & 可审计**：
  - 所有策略写入统一的 `results.jsonl`（含 `ansatz_spec`, `optimizer_spec`, `git_diff`）。
  - 结果可复现、可对比、可追踪来源代码状态。
- **预算约束 & 自动停止**：
  - 所有搜索都必须通过 `SearchController` 管理预算与停止规则。
  - 长时间无改进/连续失败会触发策略切换或空间收缩。
- **可插拔策略**：
  - 通过 `SearchStrategy` 接口挂载「策略插件」。
  - 支持在后续阶段新增 ADAPT-VQE、BO、RL 等搜索后端。

---

## 4. 开发路线图（按阶段）

### Phase 1：策略基础设施整理与统一日志（2026-03-11, DONE）

目标：把现有 GA、MultiDim、Baseline 在工程上拉齐，为后续扩展做地基。

主要工作：
1. 整理统一的「实验记录规范」
   - [x] 在 `doc/` 中补充一份简短规范，定义 `results.jsonl` 的字段约定。
2. 完成 GA / MultiDim / Baseline 的输出对齐
   - [x] GA/MultiDim 均支持自动保存最优 `best_config_*.json`。
3. 增强 `SearchController`
   - [x] 实现详细日志与回调接口。

---

### Phase 2：策略插件化与控制器抽象（2026-03-11, DONE）

目标：从「单策略脚本」演进为「多策略编排器」，允许在一个实验中按预算自动切换策略。

主要工作：
1. 引入 `SearchStrategy` 抽象
   - [x] 在 `core/` 新增 `strategy_base.py`。
   - [x] 将 `GAOptimizer` 适配为 `GASearchStrategy`。
   - [x] 提供 `GridSearchStrategy` 封装。
2. 引入 `SearchOrchestrator`
   - [x] 实现顺序驱动多个搜索策略。
   - [x] 响应「无改进」信号实现自动切换。
3. 更新实验脚本入口
   - [x] 创建 `experiments/*/auto_search.py`。

产出：
- 一个统一的策略抽象层，支持热插拔。

---

### Phase 3：规范化、参数映射与构造性策略（进行中）

目标：从“功能扩张”转向“规范化与稳健性”，引入支持高比特扩展的构造性协议与映射机制。

主要工作：
1. **评估协议与局部优化器抽象 (P0)**
   - [ ] 制定 `doc/evaluation_protocol.md`，统一预算、种子、早停与排名指标。
   - [ ] 在 `core/engine.py` 中为 `vqe_train` 提供通用的优化器接口，支持 Adam/SPSA 等热切。
2. **Warm-start 与参数映射协议 (P1)**
   - [ ] 定义显式的参数映射规则（结构编辑前后参数如何投影与继承）。
   - [ ] 优化热启动逻辑，加速大比特线路收敛。
3. **ADAPT-VQE / 贪心构造原型 (P1)**
   - [ ] 在 `core/` 新增 `adapt_vqe.py` 并封装为 `AdaptVQEStrategy` 插件。
   - [ ] 在 4-qubit 体系中基于统一协议对比 ADAPT vs GA 的收敛效率。
4. **审计增强 (P1)**
   - [ ] 日志 `results.jsonl` 加入 `schema_version` 与环境指纹（commit, version等）。
   - [ ] 优化 `git_diff` 记录，避免日志文件过大。

产出：
- 统一的评估协议文档与优化器抽象接口。
- 显式的参数映射协议。
- 可运行且公平评估的 ADAPT-VQE 策略插件。

---

### Phase 4：高阶策略与大规模验证（远期规划）

目标：在统一协议下尝试更智能的搜索方法，并验证架构的可扩展性。

工作方向：
- **贝叶斯优化 (BO)**：适合点数少、开销大的精细搜索阶段。
- **强化学习 (RL)**：将「构造线路」视为 Markov 决策过程。
- **高比特场景验证**：在架构上引入硬件拓扑约束，探索近似评估方法。

---

## 5. 任务分解与建议优先级

### 已完成 (Archived Progress)
- [x] Phase 1: 统一日志记录 & GA/MultiDim/Baseline 对齐
- [x] Phase 2: 抽象 `SearchStrategy` & 引入 `SearchOrchestrator` & 策略热切换

### 当前重点 (Current Priority)
1. **(P0) 制定评估协议** (`doc/evaluation_protocol.md`)：定义预算、种子数、排名指标。
2. **(P0) 日志规范升级**：加入版本号、环境指纹、显式 Config 路径记录。
3. **(P1) 局部优化器抽象**：支持动态配置 Adam/SPSA。
4. **(P1) 参数映射协议**：规范 Warm-start 的参数继承行为。
5. **(P1) ADAPT-VQE 实现**：基于新协议实现插件。

---

## 6. 里程碑与检验标准

- M1 (DONE)：GA / MultiDim / Baseline 生成统一格式 JSONL（2026-03-11）。
- M2 (DONE)：`SearchOrchestrator` 自动完成策略切换示例（2026-03-11）。
- M3：**协议固化**。完成评估协议制定与日志版本化升级。
- M4：**构造验证**。ADAPT-VQE 在 4-qubit 系统上完成公平对比验证。
- M5：**外推论证**。完成高比特（50-100+）场景下的架构约束与方案论证。

本 Plan 将随代码演进而更新。
