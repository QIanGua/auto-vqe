# 架构设计与扩展开发 (Architecture & Extension)

本文档说明 Agent-VQE 的分层架构、核心 Schema 关系，以及如何扩展新物理体系、新搜索策略和新 FailureHandler。

## 1. 四层架构

```
┌─────────────────────────────────────────────┐
│            Control Layer (研究控制层)         │
│  runtime.py → ResearchAgent → PolicyEngine  │
│  → ExperimentExecutor → ResultInterpreter   │
│  → ResearchSession → ResearchMemoryStore    │
├─────────────────────────────────────────────┤
│          Evaluation Layer (评估层)           │
│  evaluator/training.py    — VQE 优化循环     │
│  evaluator/api.py         — 评估包装         │
│  evaluator/report.py      — run.json 审计    │
│  evaluator/logging_utils.py — events.jsonl  │
├─────────────────────────────────────────────┤
│            Search Layer (搜索层)             │
│  generator/ga.py          — 遗传算法         │
│  generator/grid.py        — 网格搜索         │
│  generator/adapt.py       — ADAPT 系列       │
│  orchestration/controller — 编排与预算       │
├─────────────────────────────────────────────┤
│          Reality Layer (物理现实层)           │
│  foundation/base_env.py   — QuantumEnvironment│
│  experiments/tfim/env.py  — TFIM 环境        │
│  experiments/lih/env.py   — LiH 环境         │
│  ⚠️ 这一层不应被 Agent 修改                  │
└─────────────────────────────────────────────┘
```

## 2. 核心 Schema 关系

```
AnsatzSpec                    # 统一 Ansatz 结构表示
  ├── config: dict            # GA/Grid 兼容的结构化参数
  ├── blocks: [BlockSpec | OperatorSpec]  # 层级化结构
  └── growth_history: [StructureEdit]     # 生长历史

CandidateSpec                 # 搜索候选
  ├── ansatz: AnsatzSpec
  ├── proposed_by: str        # 提出者 (ga / adapt / ...)
  └── warm_start_from: str | None

EvaluationSpec → EvaluationResult  # 评估配置 → 评估结果

--- Research Layer ---

HypothesisSpec → ActionSpec → RunBundle → DecisionRecord → ResearchMemory
```

### Schema 文件位置

| Schema | 文件 |
|--------|------|
| `AnsatzSpec`, `CandidateSpec`, `EvaluationSpec`, `EvaluationResult` | `core/model/schemas.py` |
| `BlockSpec`, `OperatorSpec`, `StructureEdit`, `WarmStartPlan` | `core/model/schemas.py` |
| `SearchSpaceSpec`, `OptimizerSpec`, `HardwareConstraintSpec` | `core/model/schemas.py` |
| `HypothesisSpec`, `ActionSpec`, `DecisionRecord`, `RunBundle`, `ResearchMemory` | `core/model/research_schemas.py` |

## 3. 添加新物理体系

### 步骤概览

1. 创建 `experiments/<new_system>/` 目录
2. 实现 `env.py` — 继承 `QuantumEnvironment`
3. 实现 `run.py` — 使用 `ExperimentManifest` 声明
4. 创建 `presets/` 目录放置搜索配置

### 3.1 实现 env.py

```python
# experiments/new_system/env.py
import tensorcircuit as tc
from core.foundation.base_env import QuantumEnvironment

class NewSystemEnvironment(QuantumEnvironment):
    def __init__(self, **kwargs):
        self._n_qubits = 4  # 你的量子比特数
        self._hamiltonian = self._build_hamiltonian()
        self._exact_energy = self._compute_exact_energy()

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @property
    def exact_energy(self) -> float:
        return self._exact_energy

    def compute_energy(self, circuit) -> float:
        # 计算给定线路的期望能量
        ...

ENV = NewSystemEnvironment()
```

### 3.2 实现 run.py

```python
# experiments/new_system/run.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.shared import (
    ExperimentManifest, SearchSpec, BaselineSpec,
    run_config_experiment, run_search_experiment,
)
from core.representation.compiler import build_ansatz

SYSTEM_DIR = os.path.dirname(__file__)
RUNS_DIR = os.path.join(SYSTEM_DIR, "artifacts", "runs")

def load_env():
    from experiments.new_system.env import ENV
    return ENV

def build_circuit(config):
    return build_ansatz(config, load_env().n_qubits)

MANIFEST = ExperimentManifest(
    name="new_system",
    system_dir=SYSTEM_DIR,
    runs_dir=RUNS_DIR,
    fallback_config={
        "layers": 2,
        "single_qubit_gates": ["ry", "rz"],
        "two_qubit_gate": "cnot",
        "entanglement": "linear",
    },
    config_priority=("presets/ga.json",),
    run_result_label="NewSystem_Phase1",
    load_env=load_env,
    build_circuit=build_circuit,
    searches={
        "ga": SearchSpec(
            kind="ga",
            dimensions={
                "layers": [1, 2, 3, 4],
                "single_qubit_gates": [["ry"], ["ry", "rz"]],
                "two_qubit_gate": ["cnot", "rzz"],
                "entanglement": ["linear", "ring"],
            },
            base_exp_name="NewSystem_GA",
            run_slug="new_system_ga_search",
            pop_size=8,
            generations=4,
        ),
    },
)

# 标准 CLI 解析（参考 experiments/lih/run.py 的 build_parser / main）
```

### 3.3 注册到 Agent Runtime

Agent runtime 会自动发现 `run.py` 中的 `MANIFEST.searches`，无需额外注册。

```bash
# 直接使用
uv run python core/research/runtime.py --dir experiments/new_system --strategy ga --max 50
```

## 4. 添加新搜索策略

### 4.1 实现策略类

```python
# core/generator/my_strategy.py
from core.generator.base import SearchStrategy
from core.model.schemas import AnsatzSpec, CandidateSpec, EvaluationResult

class MyStrategy(SearchStrategy):
    def __init__(self, search_config: dict, **kwargs):
        super().__init__()
        self.config = search_config

    def generate_candidates(self, **kwargs) -> list[CandidateSpec]:
        # 生成候选 Ansatz 列表
        ...

    def update(self, results: list[EvaluationResult]) -> None:
        # 根据评估结果更新内部状态
        ...

    def best_candidate(self) -> CandidateSpec | None:
        # 返回当前最优候选
        ...
```

### 4.2 在 ExperimentManifest 中注册

在 `run.py` 的 `MANIFEST.searches` 中添加新策略：

```python
searches={
    "my_strategy": SearchSpec(
        kind="my_strategy",
        dimensions={...},
        base_exp_name="MyStrategy_Search",
        run_slug="my_strategy_search",
        # 策略特有参数
    ),
}
```

## 5. 扩展 FailureHandler

### 5.1 实现新的 Handler

```python
# 可以放在 core/research/policy.py 或新文件中
from core.research.policy import FailureHandler

class AdaptSaturationHandler(FailureHandler):
    """Handle ADAPT algorithm saturation."""

    def can_handle(self, failure_type, memory, controller):
        return failure_type == "adapt_saturation"

    def handle(self, engine, memory, controller, hypothesis, system_dir):
        # 当 ADAPT 饱和时的处理逻辑
        alt_strats = engine.get_alternative_strategies(memory)
        if alt_strats:
            return engine.build_switch_action(
                hypothesis, system_dir, alt_strats[0],
                rationale="ADAPT has saturated; switching to exploratory strategy."
            )
        return None
```

### 5.2 注册到 PolicyEngine

```python
from core.research.policy import PolicyEngine

engine = PolicyEngine("ga")
engine.handlers.append(AdaptSaturationHandler())
```

或修改 `PolicyEngine.__init__` 添加到默认链中。

## 6. Warm-start 机制

Warm-start 用于在结构变化时迁移参数，避免从零开始优化。

### 两级映射

| 层级 | 文件 | 说明 |
|-----|------|------|
| Config 级 | `core/warmstart/config_mapper.py` | 配置维度变化时的参数索引映射 |
| Ansatz 级 | `core/warmstart/ansatz_mapper.py` | Block/Operator 结构变化时的参数继承 |

### WarmStartPlan

```python
class WarmStartPlan(BaseModel):
    old_param_count: int
    new_param_count: int
    reused_indices: list[tuple[int, int]]    # (旧索引, 新索引) 映射
    initialized_indices: list[int]           # 新参数的初始化位置
    dropped_indices: list[int]               # 被丢弃的旧参数位置
    init_strategy: "zeros"|"small_random"|"copy_neighbor"|"block_default"
```

详细规则见 `doc/parameter_mapping.md`。

## 7. 实验日志 Schema (v1.2)

当前 `run.json` 使用 schema v1.2，详细字段参考 `doc/logging_spec.md`。

关键设计：
- `run.json` 是单次运行的主审计真相源
- `events.jsonl` 记录过程中间事件，不替代最终结论
- `index.jsonl` 是体系级轻量索引，用于快速检索
- `ResearchSession` 有独立的日志语义（`autoresearch.jsonl`），与上述不应混淆

## 8. 代码组织约定

| 约定 | 说明 |
|-----|------|
| 模块边界 | 每个 `core/<subpackage>/` 有自己的 `__init__.py` |
| 环境隔离 | `QuantumEnvironment` 及 `env.py` 不应被搜索/Agent 修改 |
| 统一模型 | 所有组件共用 `core/model/schemas.py` 中的数据模型 |
| 实验隔离 | 每个体系有独立的 `experiments/<system>/` 目录 |
| 标准基准 | 所有成熟 Ansatz 集中在 `baselines/` 模块，便于对比 |
| 共享逻辑 | 跨体系执行逻辑放在 `experiments/shared.py` |
| 产物路径 | 所有运行产物放入 `artifacts/runs/` |
