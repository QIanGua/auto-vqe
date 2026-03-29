# 自动量子物理学家（Auto-VQE）

这是面向 Agent 的实验执行手册。目标是在不修改物理问题定义的前提下，自动探索更好的 ansatz 结构。

## 1. 不可变领域（Objective Reality）

以下内容代表物理问题本身，默认不应修改：

- `core/foundation/base_env.py` 中定义的 `QuantumEnvironment`
- `experiments/tfim/env.py`
- `experiments/lih/env.py`

## 2. 可变领域（The Conjecture）

以下内容属于可搜索、可改进的猜想空间：

- `core/representation/compiler.py`
- `experiments/*/ga/search.py`
- `experiments/*/multidim/search.py`
- `experiments/*/orchestration/auto_search.py`
- `core/generator/` 中的策略实现
- `core/warmstart/` 中的参数继承规则

推荐优先修改结构化 config，而不是重写底层物理环境。当前常用维度包括：

- `layers`
- `single_qubit_gates`
- `two_qubit_gate`
- `entanglement`
- `init_state`
- `param_strategy`

## 3. 核心目标

核心优化目标是让 `val_energy` 尽可能逼近 `exact_energy`，同时遵守复杂度约束。

常用入口：

- TFIM：`uv run python experiments/tfim/run.py`
- LiH：`uv run python experiments/lih/run.py`

显式指定配置时可使用：

```bash
uv run python experiments/tfim/run.py --config experiments/tfim/ga/best_config_ga.json --trials 5
uv run python experiments/lih/run.py --config experiments/lih/multidim/best_config_multidim.json --trials 2
```

## 4. 奥卡姆剃刀原则

在误差相近时，优先选择更简单的方案：

- 参数更少
- 线路更浅
- 双比特门更少

可按 Pareto 视角判断：

1. `A.energy_error < B.energy_error` 且 `A.num_params <= B.num_params`
2. `A.energy_error <= B.energy_error` 且 `A.num_params < B.num_params`

## 5. 推荐搜索顺序

1. 先搜索层数，确认最小有效深度。
2. 再搜索单比特门集合与双比特门类型。
3. 再比较纠缠拓扑。
4. 最后再尝试参数绑定、初始化和 warm-start 细节。

## 6. 推荐工作流

### 方式一：GA 搜索

```bash
uv run python experiments/tfim/ga/search.py
uv run python experiments/lih/ga/search.py
```

### 方式二：多维网格搜索

```bash
uv run python experiments/tfim/multidim/search.py
uv run python experiments/lih/multidim/search.py
```

### 方式三：多策略编排

```bash
uv run python experiments/tfim/orchestration/auto_search.py
uv run python experiments/lih/orchestration/auto_search.py
```

### 方式四：可恢复外层研究循环

```bash
uv run python core/research/runtime.py --dir experiments/lih --strategy ga --target 1e-6 --max 100
```

当前这个入口已经走结构化 Agent runtime，而不是单纯 shell 脚本循环。实际运行时链路是：

```text
runtime.py
  -> ResearchAgent
  -> PolicyEngine
  -> ExperimentExecutor
  -> ResultInterpreter
  -> ResearchSession / ResearchMemoryStore
```

## 7. 单轮实验闭环

1. 阅读当前环境、默认 config 和输出目录。
2. 修改猜想空间，优先调整 config 或搜索范围。
3. 执行搜索或验证。
4. 读取 `val_energy`、`energy_error`、`num_params`、`actual_steps` 等指标。
5. 根据 Pareto 改进决定 keep / discard。

## 8. 当前输出约定

常见输出包括：

- `experiment.log`
- `results.tsv`
- `results.jsonl`
- `report_*.md`
- `convergence_*.png`
- `circuit_*.png`
- `circuit_*.json`

注意：

- LiH 的默认运行产物集中到 `experiments/lih/artifacts/runs/`
- 长周期 autoresearch session 在 `experiments/<system>/artifacts/runs/autoresearch/`
- `experiments/<system>/results.tsv` 会保留系统级轻量汇总

## 9. 预算与停止规则

- 搜索策略应通过 `SearchController` 管理预算，而不是无限制尝试。
- 多策略流程可通过 `SearchOrchestrator` 基于“无改进 / 连续失败”等信号切换。
- 若改进极小但复杂度显著上升，应倾向 discard。

## 10. 当前注意事项

- `AdaptVQEStrategy` 已存在于 `core/generator/adapt.py`，但还不是标准实验入口。
- `results.jsonl` 当前 schema 为 `1.2`。
- Agent runtime 当前主真相源是 `DecisionRecord + RunBundle + research_memory.json`。
- `autoresearch.jsonl` 与 `autoresearch.md` 仍保留为 session 兼容视图。
