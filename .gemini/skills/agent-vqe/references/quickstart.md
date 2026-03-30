# 快速上手 (Quick Start)

本文档指引你从零开始安装、运行和理解 Agent-VQE 的基本工作流。

## 1. 环境要求

| 依赖 | 版本要求 | 说明 |
|-----|---------|------|
| Python | 3.10 ~ 3.12 | 不兼容 3.13+ |
| uv | 最新版 | 包管理与虚拟环境工具 |
| just | 最新版 | 任务运行器 (可选但推荐) |
| TensorCircuit | ≥0.12, <0.16 | 量子线路模拟后端 |
| PyTorch | ≥2.1, <3.0 | 参数优化后端 |
| NumPy | ≥1.24, <2.0 | **注意：不兼容 NumPy 2.x** |
| Qiskit | ≥1.0, <2.0 | 量子线路表示与绘图 |
| PySCF | ≥2.12 | 分子计算（LiH 体系需要） |
| OpenFermion | ≥1.7 | 费米子-量子比特映射 |

## 2. 安装

```bash
# 进入项目根目录后
uv sync
```

> `uv sync` 会根据 `pyproject.toml` 和 `uv.lock` 自动创建 `.venv` 并安装全部依赖。

## 3. 首次验证（Smoke Test）

以下命令只查看帮助信息，不启动真实计算：

```bash
uv run python experiments/tfim/run.py --help
uv run python experiments/lih/run.py --help
uv run python core/research/runtime.py --help
uv run python core/molecular/generate.py --list
```

## 4. 运行第一个实验

### 4.1 TFIM (Transverse-Field Ising Model)

```bash
# 使用默认最优配置运行 3 个 trial
uv run python experiments/tfim/run.py --trials 3

# 或使用 just 快捷命令
just tfim
```

### 4.2 LiH (Lithium Hydride)

```bash
# 使用默认最优配置运行 2 个 trial
uv run python experiments/lih/run.py --trials 2

# 或使用 just 快捷命令
just lih
```

### 4.3 指定配置运行

```bash
uv run python experiments/tfim/run.py --config experiments/tfim/presets/ga.json --trials 5
uv run python experiments/lih/run.py --config experiments/lih/presets/multidim.json --trials 2
```

## 5. 理解输出产物

每次运行会在 `experiments/<system>/artifacts/runs/` 下创建一个时间戳目录：

```text
experiments/lih/artifacts/runs/20260329_120000_lih_vqe/
├── run.json              # 主审计记录（schema v1.2）
├── events.jsonl          # 过程事件流（append-only）
├── run.log               # 文本日志
└── config_snapshot.json  # 使用的 Ansatz 配置快照（若有）
```

### 5.1 run.json 核心字段

| 字段 | 说明 |
|-----|------|
| `metrics.val_energy` | VQE 最终优化能量 |
| `metrics.exact_energy` | 精确参考能量 |
| `metrics.energy_error` | 能量误差 `|val_energy - exact_energy|` |
| `metrics.num_params` | 参数量 |
| `metrics.two_qubit_gates` | 双量子比特门数量 |
| `metrics.runtime_sec` | 运行时间（秒） |
| `metrics.actual_steps` | 实际优化步数 |
| `ansatz_spec` | Ansatz 结构化描述 |
| `git_info` | Git commit + dirty 状态 |
| `runtime_env` | Python/OS/依赖版本快照 |

### 5.2 系统级索引

```text
experiments/<system>/artifacts/index.jsonl
```

这是一个 append-only 的轻量索引，每次运行自动追加一条摘要。

## 6. 运行测试

```bash
# 快速测试（排除 slow 标记的测试）
just quick-test

# 全量测试
just test-all

# 或直接使用 pytest
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run pytest -m "not slow" tests/
```

## 7. Justfile 命令速查

| 命令 | 说明 |
|-----|------|
| `just` | 显示所有可用命令 |
| `just setup` | `uv sync` 安装依赖 |
| `just lih` | 运行 LiH 验证实验 |
| `just tfim` | 运行 TFIM 验证实验 |
| `just ga-lih` | LiH GA 结构搜索 |
| `just ga-tfim` | TFIM GA 结构搜索 |
| `just multidim-lih` | LiH 多维网格搜索 |
| `just multidim-tfim` | TFIM 多维网格搜索 |
| `just baseline-lih` | LiH UCCSD Baseline |
| `just baseline-tfim` | TFIM Baseline |
| `just quick-test` | 快速回归测试 |
| `just test-all` | 全量测试 |
| `just benchmark` | 参数映射基准测试 |
| `just clean-py` | 清理 Python 缓存 |
| `just clean-experiments` | 清理实验产物 |
| `just clean` | 全量清理 |

## 8. 配置优先级

验证实验加载配置的优先级：

1. 显式 `--config` 参数
2. `presets/ga.json`
3. `presets/multidim.json`
4. 脚本内置 `fallback_config`

## 9. 下一步

| 想做什么 | 去哪里 |
|---------|--------|
| 进行结构搜索 | `references/structural-search.md` |
| 运行 Agent 自主研究 | `references/agent-research-loop.md` |
| 生成分子数据 | `references/molecular-hamiltonian.md` |
| 理解架构设计 | `references/architecture-extension.md` |
