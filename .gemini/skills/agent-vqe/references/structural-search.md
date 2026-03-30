# 结构搜索与策略选择 (Structural Search)

本文档详细说明 Agent-VQE 的四种 Ansatz 结构搜索策略及其使用方法。

## 1. 四种搜索策略概览

| 策略 | 文件位置 | 定位 | 适用场景 |
|-----|---------|------|---------|
| **GA** | `core/generator/ga.py` | 遗传算法，全局探索 | 搜索空间大、初始阶段、需要广泛探索 |
| **MultiDim** | `core/generator/grid.py` | 多维网格搜索 | 搜索空间可控、需要系统性覆盖 |
| **ADAPT** | `core/generator/adapt.py` | 自适应梯度选算符 | 分子体系、精细结构增长 |
| **Qubit-ADAPT** | `core/generator/adapt.py` | 量子比特级自适应 | 分子体系、更细粒度的算符选择 |
| **Baselines** | `baselines/` | 成熟的标准 Ansatz 库 | 作为基准测试、研究起点或对比参照 |

### 策略选择决策树

```
需要搜索或使用什么样的 Ansatz？
├── 从零开始探索 → GA (广覆盖，可能找到意料之外的好结构)
├── 已知大致范围 → MultiDim (系统性枚举最优组合)
├── 分子体系，想自动增长线路 → ADAPT (梯度驱动的算符选择)
├── 需要更紧凑的分子 Ansatz → Qubit-ADAPT (单/双量子比特算符层级)
└── 需要标准化学基准 → Baselines (Givens, k-UpCCGSD, QUCC, UCCSD)
```

## 2. 搜索空间维度

所有维度定义在 `core/model/schemas.py` 的 `SearchSpaceSpec` 中：

| 维度 | 类型 | 示例值 | 说明 |
|-----|------|-------|------|
| `layers` | `List[int]` | `[1, 2, 3, 4, 5]` | Ansatz 线路深度 |
| `entanglement` | `List[str]` | `["linear", "ring", "brick", "full"]` | 纠缠拓扑 |
| `single_qubit_gates` | `List[List[str]]` | `[["ry"], ["rx", "ry", "rz"]]` | 单比特门集合 |
| `two_qubit_gate` | `List[str]` | `["cnot", "cz", "rzz", "rxx_ryy_rzz"]` | 双比特门类型 |
| `init_state` | `List[str]` | `["zero", "hadamard", "hf"]` | 初始态制备 |
| `param_strategy` | `List[str]` | `["independent", "tied"]` | 参数共享策略 |

## 3. GA 搜索

### CLI 命令

```bash
# TFIM 体系
uv run python experiments/tfim/run.py search ga

# LiH 体系
uv run python experiments/lih/run.py search ga

# 使用 just
just ga-tfim
just ga-lih
```

### GA 参数（在 ExperimentManifest 中配置）

| 参数 | LiH 默认值 | 说明 |
|-----|-----------|------|
| `pop_size` | 10 | 每代种群大小 |
| `generations` | 6 | 迭代代数 |
| `mutation_rate` | 0.4 | 变异概率 |
| `elite_count` | 2 | 精英保留数 |
| `trials_per_config` | 2 | 每个配置的验证次数 |
| `max_steps` | 600 | 单次优化最大步数 |

### GA 工作流

1. 在搜索空间中随机生成初始种群（`pop_size` 个配置）
2. 使用 `build_ansatz()` 编译每个配置为量子线路
3. 使用 `vqe_train()` 评估适应度（`energy_error`）
4. 选择→交叉→变异生成下一代
5. 保留精英（`elite_count`）直接进入下一代
6. 重复 `generations` 轮并返回最优个体

## 4. MultiDim 搜索

### CLI 命令

```bash
uv run python experiments/tfim/run.py search multidim
uv run python experiments/lih/run.py search multidim

just multidim-tfim
just multidim-lih
```

### MultiDim 工作流

1. 将搜索空间各维度的可选值做笛卡尔积（或按序扫描）
2. 逐个评估每种组合
3. 记录所有结果并排序
4. 输出 Pareto 最优前沿

> MultiDim 适合搜索空间较小（<100 种组合）且需要完整覆盖的场景。

## 5. ADAPT-VQE 搜索

### CLI 命令

```bash
uv run python experiments/lih/run.py search adapt
uv run python experiments/lih/run.py search qubit_adapt
```

### ADAPT 特有配置

```python
# 在 SearchSpec.search_config 中指定
{
    "gradient_epsilon": 1e-3,       # 数值梯度步长
    "gradient_tol": 1e-4,           # 梯度阈值（低于此值停止增长）
    "max_adapt_steps": 6,           # 最大增长步数
    "pool_config": {                # 算符池配置
        "init_state": "hf",
        "hf_qubits": [0, 1],
        "occupied_orbitals": [0, 1],
        "virtual_orbitals": [2, 3],
        "include_singles": True,
        "include_doubles": True,
    }
}
```

### Qubit-ADAPT 特有配置

```python
{
    "gradient_epsilon": 1e-3,
    "gradient_tol": 1e-4,
    "max_adapt_steps": 6,
    "max_body": 2,                  # 最大算符体数
    "include_single_qubit": True,   # 是否包含单比特算符
}
```

### ADAPT 工作流

1. 构建算符池（Pauli exponential operators）
2. 计算所有池内算符对当前态的梯度
3. 选择梯度最大的算符追加到 Ansatz
4. 重新优化全部参数
5. 重复直到梯度低于阈值或达到最大步数

## 6. 标准化学基准 (Standard Chemistry Baselines)

除了动态搜索外，框架在 `baselines/` 目录中集成了多种成熟的量子化学 Ansatz 实现。这些实现均遵循 `AnsatzSpec` 架构，使用 `tensorcircuit` 编写，适合作为研究的起点。

| 基准 Ansatz | 文件位置 | 特点 |
|------------|---------|------|
| **Givens** | `baselines/givens.py` | 高效 Givens 旋转，门数量极少 (双激发 30 门)，粒子数守恒 |
| **k-UpCCGSD**| `baselines/kupccgsd.py` | 分层广义配对单一双激发，支持强相关系统，表达能力强 |
| **QUCC** | `baselines/qucc.py` | Qubit Unitary Coupled Cluster，门电路分解紧凑 |
| **UCCSD** | `baselines/uccsd.py` | 标准一阶 Trotter 展开的 Jordan-Wigner 映射 UCCSD |

### 获取基准 Ansatz
通过各模块的 `build_ansatz(env, config)` 函数获取 `AnsatzSpec`：

```python
from baselines import givens, kupccgsd, qucc, uccsd

# 示例：获取 LiH 的 Givens 基准
spec = givens.build_ansatz(env, {"layers": 1})
```

## 7. 多策略编排 (Orchestration)

### CLI 命令

```bash
uv run python experiments/tfim/run.py auto
uv run python experiments/lih/run.py auto
```

### 编排工作流

编排由 `SearchOrchestrator`（`core/orchestration/controller.py`）管理：

1. 按 `OrchestrationSpec` 定义的 `phases` 序列执行
2. 每个 phase 可以是不同策略（GA 先探索，MultiDim 再精搜）
3. `SearchController` 控制预算与停止：
   - `max_runs`: 最大运行次数
   - `no_improvement_limit`: 连续无改进次数上限
   - `failure_limit`: 连续失败次数上限
4. 各 phase 之间通过 promotion 机制传递最优配置

## 7. 结果比较与 Pareto 改进判据

根据 `doc/evaluation_protocol.md` 和 `program.md` 的约定：

### 排名优先级

1. `energy_error` (越低越好)
2. `num_params` (越少越好)
3. `two_qubit_gates` (越少越好)
4. `runtime_sec` (越短越好)

### Pareto 改进

当且仅当满足以下条件之一，新候选被视为 Pareto 改进：

- **能量改进**：`new.energy_error < best.energy_error * 0.95` (至少 5% 改进)
- **效率改进**：`new.energy_error <= best.energy_error * 1.05` 且 `new.num_params < best.num_params * 0.9` (相近精度但参数量显著减少)

### 奥卡姆剃刀原则

在误差相近时，优先选择更简单的方案：参数更少、线路更浅、双比特门更少。

## 8. 推荐搜索顺序

来自 `program.md` 的建议：

1. **先搜索层数** — 确认最小有效深度
2. **再搜索门集和门类型** — 单比特门集合 + 双比特门类型
3. **再比较纠缠拓扑** — linear / ring / brick / full
4. **最后调参数细节** — 参数绑定、初始化、warm-start

## 9. 查看搜索结果

搜索完成后，最优配置通常保存在：

```text
experiments/<system>/presets/ga.json          # GA 搜索产出
experiments/<system>/presets/multidim.json    # MultiDim 搜索产出
experiments/<system>/best_config.json         # 全局最优（若有）
```

使用最优配置进行验证：

```bash
uv run python experiments/lih/run.py --config experiments/lih/presets/ga.json --trials 5
```
