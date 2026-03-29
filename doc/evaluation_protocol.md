# Agent-VQE 实验评估协议（Evaluation Protocol）

> 版本：1.2  
> 状态：当前执行约定  
> 更新时间：2026-03-29

## 1. 目标

确保不同搜索策略、不同 ansatz 结构和不同运行入口之间的比较口径尽量一致，并与当前代码实现保持一致。

## 2. 预算约定

仓库当前没有为所有脚本强制一个完全统一的固定步数；预算由具体入口脚本和多保真阶段共同决定。

当前已落地的常见默认值：

- `experiments/tfim/run.py`: 默认 `max_steps=1500`
- `experiments/lih/run.py`: 默认 `max_steps=800`
- `core/evaluator/api.py` 的 promotion 约定：
  - `medium`: `max_steps=150`
  - `full`: `max_steps=500`
- `core/generator/adapt.py` 原型中，quick 评估常用 `max_steps=50`

因此，正式比较时必须明确记录：

- 使用的入口脚本
- `max_steps`
- 学习率或优化器配置
- `trials` 或 `n_seeds`

## 3. 排名原则

默认按以下优先级比较：

1. `energy_error` 或 `val_energy`
2. `num_params`
3. `two_qubit_gates`
4. `runtime_sec`

当精度处于同一量级时，优先保留更简单的结构。

## 4. 多种子与稳定性

- 搜索阶段可先用较少 trial 或单 seed 进行筛选。
- 验证阶段建议使用入口脚本的 `--trials` 做重复运行。
- 结论性对比至少应记录：
  - 最优值
  - 均值或中位数
  - 方差或标准差中的至少一种

## 5. 审计要求

每次正式实验至少要能恢复以下信息：

- 使用的 config 或 `config_path_used`
- `optimizer_spec`
- `metrics`
- `git_info`
- `runtime_env`
- `artifact_paths`

这些字段当前由 `core/evaluator/report.py` 统一写入 `run.json`，并摘要追加到 `index.jsonl`。

## 6. 例外情况

以下场景允许预算不一致，但必须在报告或备注中注明：

- quick / medium / full 多保真晋级
- LiH 几何扫描
- 100-qubit TFIM scaling
- baseline 或 smoke run

## 7. 当前缺口

- 还没有一个仓库级的“统一正式 benchmark matrix”脚本自动锁定所有预算。
- `ResearchSession` 与主 `run.json / index.jsonl` 之间仍有字段层面的重复与偏差。
