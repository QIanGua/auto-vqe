# 自动量子物理学家 (Auto-VQE)

这是一个让 LLM 自动探索量子线路结构（Ansatz）的实验。我们的目标是逼近一维横场伊辛模型 (TFIM) 的基态能量。

## 实验约束 (The Refutation Rules)

- **不可变领域**: `environment.py` 包含客观物理定律和评估约束。**绝对禁止修改**。
- **可变领域**: `ansatz.py` 是你的游乐场。你可以修改：
    - `create_circuit` 中的量子门组合（例如引入 $R_z$, $CZ$, $Mølmer-Sørensen$ 门，或者改变纠缠层拓扑）。
    - 优化器策略（例如添加动量 Momentum、使用 Adam 优化逻辑，或者调节学习率）。

## 奥卡姆剃刀原则 (Simplicity Criterion)

在能量误差（`energy_error`）相近的情况下，参数数量（`num_params`）越少、线路深度越浅越好。如果一个改动使得能量下降了极小的幅度，却让线路复杂了一倍，这个改动应当被丢弃。

## 目标

运行 `python ansatz.py`。你的唯一核心优化目标是：**最小化 `val_energy`**（使其无限逼近 `exact_energy`，即最小化 `energy_error`）。

## 实验循环 (The Loop)

1. **查看**当前代码状态。
2. **直接修改** `ansatz.py` 提出一个新的线路猜想。
3. **运行实验**：`python ansatz.py > run.log 2>&1`
4. **读取结果**：`grep "^val_energy:\|^num_params:\|^energy_error:" run.log`
5. **记录结果** 到 `results.tsv` (格式: `commit/tag, val_energy, num_params, 状态(keep/discard), 变更描述`)。
6. **评估决策**：
    - 如果 `val_energy` 显著降低，或在能量不变的情况下参数/线路显著简化，则保留代码 (`keep`)。
    - 如果能量变差或系统崩溃，则回退代码 (`discard` -> `git reset --hard`)。
7. **永不停歇**，直到人类介入。