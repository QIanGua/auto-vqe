# Agent-VQE 开发计划（Plan）

> 版本：2026-03-29  
> 目标：维护一个可扩展、可插拔、可审计的自动 Ansatz 搜索框架，并确保计划文档与当前代码状态一致。

## Changelog

- **2026-03-29**: `core/research/` 已完成单 Agent 外层运行时第一轮收束，新增 `runtime / agent / policy / executor / interpreter / memory_store / research_schemas`。
- **2026-03-29**: `core/research/driver.py` 已移除，CLI 与恢复入口统一收口到 `core/research/runtime.py`。
- **2026-03-29**: `ResearchSession` 已以结构化 `DecisionRecord + RunBundle` 作为主写入路径，`autoresearch.jsonl` 保留兼容视图。
- **2026-03-29**: 计划同步到重构后的仓库结构，`core/` 已拆分为 `foundation / representation / evaluator / generator / orchestration / research / warmstart`。
- **2026-03-29**: 确认实验目录规范已迁移到 `artifacts/`，当前 `run.json` 记录沿用 `1.2` 字段集。
- **2026-03-29**: 确认 `AdaptVQEStrategy` 已有原型与测试，但仍缺少与 GA / MultiDim 并列的标准实验入口。
- **2026-03-11**: 完成统一日志、策略接口与编排器的第一轮实现。

## 1. 总体目标

- 把 ansatz 设计从“手工调脚本”提升为“结构化、可比较、可自动搜索”的问题。
- 把搜索、评估、日志、报告、恢复与策略切换统一到同一套工程协议下。
- 为更大规模量子系统保留架构，但当前默认验证仍以 4-qubit TFIM / LiH 为主。

当前阶段暂不处理：

- 不替换底层模拟后端。
- 不把 50-100 qubit 运行作为默认常规工作流。

## 2. 当前框架现状（2026-03-29）

代码结构：

- `core/foundation/base_env.py`：量子环境抽象，作为“客观现实”层保持稳定。
- `core/representation/compiler.py`：由结构化 config 构造线路，并估算线路成本。
- `core/evaluator/`：
  - `training.py`：统一 VQE 优化循环
  - `report.py`：`run.json` 主记录与可选 Markdown/图像渲染
  - `logging_utils.py`：`events.jsonl` 与文本日志
  - `api.py`：实验目录创建、多保真评估与晋级辅助
- `core/generator/`：
  - `strategy.py`：统一策略接口
  - `ga.py`：GA 搜索
  - `grid.py`：网格 / 多维搜索
  - `adapt.py`：ADAPT-VQE 原型
- `core/orchestration/controller.py`：`SearchController` 与 `SearchOrchestrator`
- `core/research/runtime.py`：session pointer、resume wiring 与运行时入口
- `core/research/agent.py`：单 Agent 外层研究循环 owner
- `core/research/policy.py`：规则驱动的 `HypothesisSpec / ActionSpec` 选择器
- `core/research/executor.py`：结构化执行层，产出 `RunBundle`
- `core/research/interpreter.py`：研究判断层，产出 `DecisionRecord`
- `core/research/session.py`：结构化 session API
- `core/research/memory_store.py`：`research_memory.json` / `autoresearch.jsonl` / `autoresearch.md`
- `core/warmstart/`：配置级与参数级映射

实验入口：

- `experiments/tfim/run.py`
- `experiments/lih/run.py`
- `experiments/*/run.py search ga`
- `experiments/*/run.py search multidim`
- `experiments/*/run.py auto`

输出约定：

- 每次运行生成独立时间戳目录
- LiH 与 autoresearch 的生成产物集中放入 `artifacts/runs/`
- TFIM 的生成产物也统一放入 `artifacts/runs/`
- `run.json` 当前记录沿用 `schema_version = 1.2`
- `experiments/<system>/artifacts/index.jsonl` 保留系统级轻量索引

## 3. 设计原则

- **分层优化**：结构搜索与连续参数优化解耦。
- **统一审计**：实验记录必须能追溯到 config、代码状态、环境指纹与产物路径。
- **预算优先**：搜索过程由 `SearchController` 管理，而不是无限试错。
- **策略可插拔**：GA、Grid、ADAPT 及未来方法共享同一生成器表面。
- **文档同步**：README、Plan、协议文档必须反映当前真实入口和目录。

## 4. 路线图

### Phase 1：统一日志与基础搜索入口（DONE）

- [x] GA / MultiDim / Baseline 输出对齐
- [x] 默认实验产物已收束为 `run.json + events.jsonl + run.log`
- [x] 产出 `best_config_*.json`

### Phase 2：策略插件化与编排器（DONE）

- [x] 引入统一策略接口
- [x] 实现 `SearchController`
- [x] 实现 `SearchOrchestrator`
- [x] 在 TFIM / LiH 中提供多策略 demo

### Phase 3：协议固化、Warm-start 与构造式原型（MOSTLY DONE）

1. 评估与日志协议
   - [x] `doc/evaluation_protocol.md`
   - [x] `doc/logging_spec.md`
   - [x] `schema_version 1.2`
2. Typed schema
   - [x] `core/model/schemas.py`
   - [x] `tests/test_schemas.py`
3. Warm-start
   - [x] `core/warmstart/config_mapper.py`
   - [x] `core/warmstart/ansatz_mapper.py`
   - [x] 参数映射测试与基准
4. 构造式策略
   - [x] `core/generator/adapt.py` 原型
   - [ ] 补齐与 GA / MultiDim 同等级的实验入口与对比报告
5. 优化器抽象
   - [x] evaluator 层已支持统一优化器描述与评估对象

### Phase 4：外层自动研究循环（PARTIAL）

- [x] `core/research/runtime.py` 可恢复运行
- [x] session pointer 写入 `artifacts/state/`
- [x] 单 Agent runtime 已拆分为 `agent / policy / executor / interpreter / session / memory_store`
- [x] `ResearchSession` 已以结构化 decision/run 对象作为主写入路径
- [ ] `ResearchSession` 与主 `run.json / index.jsonl` 的跨文件语义继续对齐
- [ ] keep / discard、hypothesis memory 与跨策略经验继续丰富

### Phase 5：大规模与迁移验证（FUTURE）

- [ ] 强化 100-qubit TFIM scaling workflow
- [ ] 建立更明确的 multi-fidelity benchmark 约定
- [ ] 探索跨任务迁移与结构先验复用

## 5. 当前优先级

1. 把 ADAPT 原型提升为标准实验入口，并形成与 GA / MultiDim 的公平对比。
2. 继续减少 research runtime 与 legacy 文件视图之间的语义重复。
3. 持续保持主文档和实验目录说明与代码行为一致。

## 6. 验收标准

- M1：主文档只引用现存文件、脚本和目录。
- M2：实验日志字段、schema 版本和产物路径与实现一致。
- M3：标准搜索策略至少包括 GA、Grid，ADAPT 具备可复现实验入口。
- M4：外层研究循环的 session schema 与主日志体系完成对齐。

本计划已按 2026-03-29 仓库状态更新。
