# Agent-VQE 编排协议 (Orchestration Protocol)

> **版本**: 1.0 (P1)  
> **目标**: 明确 `SearchOrchestrator` 与 `SearchController` 之间的行为契约，确保自动搜索过程的可解释性、鲁棒性与一致性。

---

## 1. 核心契约 (Behavioral Contract)

### 1.1 策略切换条件 (Strategy Switching)
- **无改进判定**: 当 `consecutive_no_improvement` 达到 `no_improvement_limit` 时，触发 `on_strategy_switch` 信号。
- **改进阈值**: 能量下降少于 `improvement_threshold` (默认 1e-4) 即视为无进展。
- **奥卡姆剃刀**: 如果能量差异在阈值内，但 `num_params` 减少，则视为有效改进。

### 1.2 故障恢复与重试 (Failure Recovery)
- **重试上限**: 单个配置在训练中如果连续失败超过 3 次，跳过该配置。
- **空间缩减**: 当全局 `consecutive_failures` 达到 `failure_limit` 时，触发 `on_space_reduction` 信号，建议缩小搜索空间或切换到更保守的策略。

### 1.3 Resume 语义 (Resumption)
- **检查点**: 每一轮实验产出的 `results.jsonl` 应包含完整的 `config` 和 `final_params`。
- **断点续跑**: 理想情况下，Orchestrator 应支持加载 `results.jsonl` 跳过已运行的配置。

---

## 2. Metric 对齐协议 (Metric Alignment)

### 2.1 统计口径
- **最优能量**: 指在本策略周期内观测到的最小能量。
- **归一化预算**: 所有的策略必须共享同一个 `SearchController` 实例，以计算总步数和总时间。

### 2.2 结果判定
- 在多策略竞争中，最终排名按以下顺序：
    1. `val_energy` (主要指标)
    2. `num_params` (次要指标，复杂度约束)
    3. `training_seconds` (效率指标)

---

## 3. 下一步增强计划

- [ ] 实现 `CheckpointedOrchestrator` 支持长实验恢复。
- [ ] 为不同策略（GA, Grid, Bayesian）提供统一的 `MetricScanner`。
