# Agent Runtime Guide

> 更新时间：2026-03-29
> 适用范围：当前仓库中的单 Agent 外层研究循环

## 1. 现在的 Agent 是什么

当前项目中的 Agent 不是聊天外壳，而是 `core/research/` 下的一套研究运行时。

核心链路是：

```text
ResearchAgent
  -> PolicyEngine
  -> ExperimentExecutor
  -> ResultInterpreter
  -> ResearchSession / ResearchMemoryStore
```

对应文件：

- `core/research/agent.py`
- `core/research/policy.py`
- `core/research/executor.py`
- `core/research/interpreter.py`
- `core/research/session.py`
- `core/research/memory_store.py`
- `core/research/runtime.py`

## 2. Agent 当前负责什么

Agent 当前负责的事情：

- 读取研究记忆 `research_memory.json`
- 形成一轮 hypothesis
- 选择本轮 action
- 调用实验入口执行搜索
- 把结果解释成 `DecisionRecord`
- 更新 `dead_ends`、`strategy_stats`、`best_energy_error`
- 把状态写回 session 目录

Agent 当前不负责的事情：

- 不修改物理环境定义
- 不直接重写底层优化器
- 不做完全自由生成的结构搜索

## 3. 研究循环如何启动

当前推荐入口：

```bash
uv run python core/research/runtime.py --dir experiments/lih --strategy ga --target 1e-6 --max 100
uv run python core/research/runtime.py --dir experiments/lih --strategy multidim --target 1e-6 --max 100
uv run python core/research/runtime.py --dir experiments/tfim --strategy ga --target 1e-6 --max 50
```

这里直接使用 `core/research/runtime.py` 作为正式入口。

## 4. Session 目录里有什么

每个 autoresearch session 当前会在：

```text
experiments/<system>/artifacts/runs/autoresearch/<timestamp>_<strategy>_autoresearch/
```

生成这些文件：

- `driver.log`
- `autoresearch.jsonl`
- `autoresearch.md`
- `research_memory.json`
- `best_config_snapshot.json`（在 keep 路径上出现）

另外，resume pointer 写在：

```text
experiments/<system>/artifacts/state/current_autoresearch_<strategy>_session
```

## 5. 运行时的主数据对象

当前 Agent runtime 主对象：

- `HypothesisSpec`
- `ActionSpec`
- `RunBundle`
- `DecisionRecord`
- `ResearchMemory`

定义位置：

- `core/model/research_schemas.py`

主真相源：

- `DecisionRecord + RunBundle`

兼容视图：

- `autoresearch.jsonl`
- `autoresearch.md`

## 6. 如果你想直接从代码里使用 Agent

最常见的两种方式：

### 6.1 跑完整循环

```python
from core.research.runtime import start_driver_with_strategy

start_driver_with_strategy(
    "experiments/lih",
    strategy="ga",
    target_error=1e-6,
    max_loops=100,
)
```

### 6.2 跑单轮结构化 step

```python
from core.research.agent import create_default_research_agent
from core.research.session import ResearchSession

session = ResearchSession("experiments/lih", "experiments/lih/artifacts/runs/autoresearch/manual_session")
agent = create_default_research_agent(
    system_dir="experiments/lih",
    strategy="ga",
    session=session,
)
success, metrics, run = agent.execute_structured_step(iteration=1)
```

## 7. 当前推荐使用方式

如果你的目标是“跑项目”：

- 先用 `experiments/*/run.py` 做验证
- 再用 `experiments/*/ga/search.py` 或 `multidim/search.py` 做结构搜索
- 最后用 `core/research/runtime.py` 跑可恢复 Agent 循环

如果你的目标是“改 Agent”：

- 优先看 `core/research/policy.py`
- 再看 `core/research/interpreter.py`
- 需要改 session 语义时看 `core/research/session.py` 和 `core/research/memory_store.py`

## 8. 当前局限

- `PolicyEngine` 目前仍然是规则驱动，不是 LLM 驱动
- `ExperimentExecutor` 仍主要兼容现有 shell/search 入口，而不是全量纯 Python 实验 API
- `AdaptVQEStrategy` 已有原型，但还不是与 GA / MultiDim 并列的标准外层研究策略
