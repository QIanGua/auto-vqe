# 自动量子物理学家 (Auto-VQE)

这是一个让 LLM 自动探索量子线路结构（Ansatz）的实验。我们的目标是逼近一维横场伊辛模型 (TFIM)、LiH 等量子体系的基态能量。

## 实验约束 (The Refutation Rules)

- **不可变领域 (Objective Reality)**  
  - `core/base_env.py` 中定义的 `QuantumEnvironment` 抽象类。  
  - `experiments/tfim/env.py`、`experiments/lih/env.py` 中给出的哈密顿量与精确能量。  
  这些文件代表真实物理世界的规则，**绝对禁止修改**。
- **可变领域 (The Conjecture)**  
  - `experiments/tfim/run.py`、`experiments/lih/run.py` 中的 ansatz 构造函数 `create_circuit`。  
  - `experiments/*/search.py` 中的 ansatz 搜索配置。  
  你可以修改：
    - `create_circuit` 中的量子门组合（例如引入 $R_z$, $CZ$, $Mølmer-Sørensen$ 门，或者改变纠缠层拓扑）。
    - 优化器策略（例如添加动量 Momentum、使用 Adam 优化逻辑，或者调节学习率）。

## 奥卡姆剃刀原则 (Simplicity Criterion)

在能量误差（`energy_error`）相近的情况下，参数数量（`num_params`）越少、线路深度越浅越好。如果一个改动使得能量下降了极小的幅度，却让线路复杂了一倍，这个改动应当被丢弃。

**Pareto 占优条件**（量化版奥卡姆剃刀）：  
方案 A 优于方案 B，当且仅当以下任一条件成立：
1. `A.energy_error < B.energy_error` **且** `A.num_params ≤ B.num_params`
2. `A.energy_error ≤ B.energy_error` **且** `A.num_params < B.num_params`

不满足 Pareto 占优的方案之间按能量误差优先排序。

## 目标

运行对应体系的实验脚本，你的唯一核心优化目标是：  
**最小化 `val_energy`**（使其无限逼近 `exact_energy`，即最小化 `energy_error`）。

- TFIM: `uv run python experiments/tfim/run.py`
- LiH: `uv run python experiments/lih/run.py`

## 搜索策略指导

### 推荐搜索维度（按优先级排列）
1. **层数 (Depth)**: 从 1 层逐步增加，寻找最小有效深度
2. **门类型 (Gate Set)**: RY, RX+RY, RZ+RX+RZ (Euler), CNOT, RZZ, CZ 等
3. **纠缠拓扑 (Entanglement Topology)**: 线性 (Linear), 环形 (Ring), 砖墙式 (Brick-Layer), 全连接 (All-to-All)
4. **参数策略 (Parameter Strategy)**: 独立参数, 参数绑定 (Tying), QAOA 式对称绑定
5. **初始化 (Initialization)**: 随机, Hadamard 基, Hartree-Fock 初始态

### 推荐搜索顺序
1. 先用 `search.py` 扫描不同层数，确定最小有效深度
2. 在最小深度上尝试不同的门类型组合
3. 对最优门类型进行拓扑优化
4. 最后尝试参数绑定策略以进一步精简

## 实验循环 (The Loop)

1. **查看**当前代码状态  
   - 阅读 `core/engine.py` 了解通用 VQE 流程。  
   - 阅读 `experiments/<system>/env.py` 理解哈密顿量结构。  
   - 阅读 `experiments/<system>/run.py` 理解当前 ansatz 设计。
2. **修改猜想空间**  
   - 直接修改 `experiments/<system>/run.py` 中的 `create_circuit` 或 ansatz 配置，提出新的线路猜想。
3. **运行实验**  
   - 执行：`uv run python experiments/<system>/run.py`  
   - 日志自动写入 `experiments/<system>/vqe_*.log`
   - 收敛曲线自动保存为 `convergence_*.png`
4. **读取结果**  
   - 在日志和终端中查看 `val_energy`, `num_params`, `energy_error`, `actual_steps` 等指标。  
   - 汇总结果会被自动追加到 `experiments/<system>/results.tsv`（含时间戳）。
5. **记录与标注**  
   - 在 `results.tsv` 中注明：`timestamp, exp_name, val_energy, energy_error, num_params, actual_steps, training_sec, comment`。
   - 使用一致的命名规范：`{System}_{SeqID}_{ShortName}`，如 `TFIM_001_Baseline`。
6. **评估决策**  
   - 如果新方案 Pareto 占优于当前最优，则保留代码 (`keep`)。  
   - 如果能量变差或系统崩溃，则回退代码 (`discard`)。
   - 可使用 `experiment_guard` 上下文管理器自动保护 `run.py`。
7. **永不停歇**，直到人类介入。
