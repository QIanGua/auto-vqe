# Agent-VQE 实验评估协议 (Evaluation Protocol)

> **版本**: 1.0 (P0)  
> **状态**: 草案  
> **目标**: 确保不同搜索策略、不同 Ansatz 结构之间的比较是公平、可复现且具有统计显著性的。

---

## 1. 训练预算 (Training Budget)

所有参与正式对比的策略必须遵循统一的预算约束，除非实验目的另有说明。

### 1.1 迭代次数 (VQE Steps)
- **默认步数**: 500 Steps。
- **早停规则 (Early Stopping)**:
  - 窗口大小: 50 Steps。
  - 相对阈值: $10^{-8}$。
  - 如果在窗口内能量改进小于阈值，则视为收敛。

### 1.2 时间约束
- **最大运行时间**: 每轮 `vqe_train` 不得超过 300 秒（在标准环境指纹设备下）。

---

## 2. 统计显著性 (Statistical Significance)

### 2.1 多种子验证 (Multi-Seed)
- **初筛阶段**: 可使用单 Seed。
- **正式验证 (Full Evaluation)**: 必须在 5 个随机种子（建议: [42, 123, 2024, 777, 999]）下进行独立实验。
- **记录指标**: 均值 ($Mean$)、中位数 ($Median$)、标准差 ($Std$)。

### 2.2 初始参数处理
- **Random Initialization**: 必须记录随机分布类型。
- **Warm-start (参数复用)**: 必须在日志中显式标记为 `warm_start: true`。

---

## 3. 排名指标 (Ranking Metrics)

评估一个 Ansatz 优劣的综合打分公式（奥卡姆剃刀）：

$$ Score = W_{energy} \cdot \log_{10}(\Delta E) + W_{params} \cdot N_{params} + W_{depth} \cdot Depth $$

> 其中 $\Delta E$ 是能量误差，$N_{params}$ 是参数量。默认权重比例应优先保证能量精度，其次是参数极简化。

---

## 4. 实验复现与审计

### 4.1 环境指纹
每次运行必须捕获以下信息：
- `code_commit_sha`
- `dependency_versions` (tensorcircuit, torch, pennylane 等)
- `device_info` (CPU/GPU 型号)

### 4.2 显式配置
- 最终结果必须明确记录所使用的 `config.json` 路径，禁止仅通过隐式文件名进行识别。
