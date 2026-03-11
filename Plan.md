# Auto-VQE 开发计划（Plan）

> 版本：2026-03-11  
> 目标读者：项目维护者 & 贡献者  
> 目标：把 Auto-VQE 从「4-qubit demo」演进为「可扩展、可插拔、可评估」的自动 Ansatz 搜索框架，并为几十到上百比特的后续实验打基础。

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
  - `ansatz_search`：离散 config 列表的搜索（多维网格等）。
  - 报告生成、结果记录（TSV + Markdown + 图像）。
- `core/search_algorithms.py`：
  - `GAOptimizer`：基于 `circuit_factory` 的 GA 搜索，遵守奥卡姆剃刀原则。
  - 输出最优配置并写入标准化 `ansatz_spec`。
- `core/controller.py`：
  - `SearchController`：核心预算与停止规则（max runs, wall-clock, no-improvement, failures）。
  - 目前仅有「触发策略切换」占位逻辑，未真正管理多策略。
- `experiments/*/run.py`：
  - 已支持从 `ga/best_config_ga.json`、`multidim/best_config_multidim.json` 等加载配置。
  - 与 Baseline/Baseline Zoo 的 `AnsatzSpec` 正在逐步对齐。

现有策略：
- GA：`core/search_algorithms.GAOptimizer` + `experiments/*/ga_search.py`。
- MultiDim：基于 `core.engine.ansatz_search` 的结构化网格搜索。
- Baseline：`baselines/` 中的固定 ansatz 对照（已开始对齐 `AnsatzSpec`）。

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
  - 不在 GA/Multidim 代码里硬编码逻辑，而是通过统一接口挂载「策略插件」。
  - 支持在后续阶段新增 ADAPT-VQE、BO、RL 等搜索后端。

---

## 4. 开发路线图（按阶段）

### Phase 1：策略基础设施整理与统一日志（短期，可立即执行）

目标：把现有 GA、MultiDim、Baseline 在工程上拉齐，为后续扩展做地基。

主要工作：
1. 整理统一的「实验记录规范」
   - [ ] 在 `doc/` 中补充一份简短规范，定义 `results.jsonl` 的字段约定（`experiment_id`, `ansatz_spec`, `optimizer_spec`, `metrics`, `decision`, `git_diff` 等）。
   - [ ] 确认 `core/engine.py` 中 JSONL 写入逻辑与 GA 写入逻辑字段一致。
2. 完成 GA / MultiDim / Baseline 的输出对齐
   - [ ] GA：确保 `GAOptimizer.run()` 输出：
     - 最优 `config`，最优 `ansatz_spec`，并在对应 `experiments/*/ga/` 下写 `best_config_ga.json`。
   - [ ] MultiDim：通过 `ansatz_search` 在 `experiments/*/multidim/` 下写 `best_config_multidim.json`。
   - [ ] Baseline：为每个 Baseline 生成 `AnsatzSpec` 风格描述，并写入同一 JSONL。
3. 增强 `SearchController`
   - [ ] 检查并补充日志输出，便于后续策略切换调试。
   - [ ] 将 `handle_persistent_failure` / `handle_no_improvement` 的 TODO 变为明确的回调接口（例如挂载到 `SearchOrchestrator`，参见 Phase 2）。

产出：
- 一套统一的结果记录格式。
- GA、MultiDim、Baseline 的最优配置文件位置和命名规则固定下来。

---

### Phase 2：策略插件化与控制器抽象（中期，架构演进）

目标：从「单策略脚本」演进为「多策略编排器」，允许在一个实验中按预算自动切换策略。

主要工作：
1. 引入 `SearchStrategy` 抽象
   - [ ] 在 `core/` 新增 `strategy_base.py`（或直接放在 `search_algorithms.py` 中）：
     - 定义抽象类 `SearchStrategy`，至少包含：
       - `run(self) -> dict`：返回 `{"best_results": ..., "best_config": ..., "best_spec": ...}`。
       - `name` / `metadata` 字段。
   - [ ] 将 `GAOptimizer` 适配为 `SearchStrategy` 的一个实现。
   - [ ] 提供一个 `GridSearchStrategy` 封装现有的 `ansatz_search` 调用。
2. 引入 `SearchOrchestrator`
   - [ ] 在 `core/controller.py` 或新文件中引入 `SearchOrchestrator`：
     - 接收一个策略列表（如 `[GA, Grid, BaselineEval]`）和一个共享的 `SearchController`。
     - 响应 `SearchController` 中「无改进」「失败太多」事件，完成：
       - 从当前策略切换到下一个策略；
       - 缩小搜索空间（可由策略内部实现）。
   - [ ] 支持在一次高层调用中完成「先 GA 粗搜再 MultiDim 精扫」的流程。
3. 更新实验脚本入口
   - [ ] 把 `experiments/tfim/ga_search.py` 和 `experiments/tfim/multidim/multidim_search.py` 统一迁移到：
     - `experiments/tfim/auto_search.py`（LiH 同理），由该文件配置 orchestrator。
   - [ ] Justfile 中保留：
     - `ga-*` / `multidim-*` 命令作为策略单独运行入口；
     - `auto-*` 命令作为组合策略入口。

产出：
- 一个统一的策略抽象层，后续新增策略只需实现 `SearchStrategy` 接口。
- 控制器能够驱动策略切换，而不仅仅是打印警告。

---

### Phase 3：面向更大系统的策略设计（中长期，方法扩展）

目标：在架构上为几十到上百比特的实验做准备，并给出具体实现路线。

核心思路：
- **搜索重心上移**：只在「结构与超参数」层做全局探索，连续参数交给 `vqe_train` 等局部优化器。
- **结构构造策略**：引入 ADAPT-VQE / 贪心式构造方法。

主要工作：
1. ADAPT-VQE / 贪心构造原型
   - [ ] 在 `core/` 新增 `adapt_vqe.py`：
     - 定义一个算符池（`Pauli` 字串等），与具体系统的 `env` 解耦。
     - 实现简单版本的「梯度驱动算符选择 + 逐步扩展 ansatz」。
   - [ ] 将其包装为一个 `SearchStrategy`（例如 `AdaptVQEStrategy`）。
2. 局部优化器抽象
   - [ ] 在 `core/engine.py` 中为 `vqe_train` 提供更通用的优化器接口：
     - 支持选择 Adam、L-BFGS、SPSA 等。
     - 优化器与搜索策略通过配置耦合，而不是硬编码。
3. 初步可扩展性验证（仍在 4-qubit 体系中）
   - [ ] 使用 ADAPT-VQE + 局部优化器重现 TFIM / LiH 结果，验证日志与策略接口兼容性。
   - [ ] 对比 GA / MultiDim / ADAPT 在相同预算下的表现。

产出：
- 一个可运行的 ADAPT-VQE 策略插件。
- 更通用的局部优化器接口，为高比特设置优化预算打基础。

---

### Phase 4：高阶策略（BO / RL / 元学习）（远期规划）

目标：在现有策略基础上，尝试更智能的结构搜索方法，主要针对「结构/超参数」层。

候选方向：
- 贝叶斯优化（BO）
  - 作用在结构超参数上（层数、拓扑类型、参数绑定模式等）。
  - 每次评估开销大、点数少，适合作为 GA 之后的精细搜索阶段。
- 强化学习（RL）
  - 把「构造线路」视为 Markov 决策过程：状态 = 当前电路，动作 = 加门/改拓扑。
  - policy 生成 ansatz 结构，`vqe_train` 作为环境反馈能量。
- 元学习 / 迁移
  - 同一类 Hamiltonian 上训练得到模板 ansatz 和初始化参数，在新系统上只做微调。

本阶段主要产出：
- 若干策略设计草图与小规模验证实验；
- 统一的评估基准（在 TFIM/LiH 上的对比），为未来几十/上百比特实验提供数据支撑。

---

## 5. 任务分解与建议优先级

为便于实际执行，给出按优先级排序的任务列表（粗略）：

1. （P0）统一 JSONL 记录 & GA/MultiDim/Baseline 对齐（Phase 1 全部）
2. (P1) 抽象 `SearchStrategy` + `GridSearchStrategy` + `GA` 适配（Phase 2 的 1、2 步）
3. (P1) 引入 `SearchOrchestrator`，实现「GA → MultiDim」的自动切换示例
4. (P2) 实现 ADAPT-VQE 原型并注册为策略插件（Phase 3 的 1 步）
5. (P2) 抽象优化器接口，支持多种局部优化算法（Phase 3 的 2 步）
6. (P3) 撰写高比特场景的设计文档草案（如何压缩搜索空间、如何强制硬件拓扑约束），与本 Plan 一同演进。

---

## 6. 里程碑与检验标准

- M1：GA / MultiDim / Baseline 均能生成统一格式的 JSONL，并可通过一个分析脚本自动输出 TFIM/LiH 的 Pareto 前沿（Expected：1–2 天）。
- M2：`SearchOrchestrator` 可在单次运行中按预算自动完成「GA 粗搜 → MultiDim 精扫 → 生成报告」（Expected：3–5 天）。
- M3：ADAPT-VQE 原型在 4-qubit TFIM/LiH 上达到与 GA/Multidim 同量级的能量误差，并完整写入日志与报告（Expected：1–2 周）。
- M4：完成至少一篇描述「策略插件化架构 + 初步高阶策略实验」的技术文档或博文，形成对外可分享的成果（Expected：1 个月内）。

本 Plan 将随代码演进而更新；建议每完成一个里程碑，在 `Plan.md` 顶部追加「Changelog」条目并标注意完成的阶段与变更摘要。

