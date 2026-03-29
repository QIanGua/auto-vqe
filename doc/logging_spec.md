# Agent-VQE 实验日志记录规范

> 当前实现版本：`schema_version = 1.2`  
> 更新时间：2026-03-29

## 1. 目标

让实验结果具备可追溯性、可比较性和可审计性，并与当前 `core/evaluator/report.py` 的真实写入字段保持一致。

## 2. 记录文件

每次运行通常会产生两类记录：

- `results.tsv`
  - 单次运行目录中的轻量摘要
  - 某些入口会同步追加到 `experiments/<system>/results.tsv`
- `results.jsonl`
  - 写在单次运行目录中
  - 每行是一条结构化实验记录

## 3. `results.jsonl` 核心字段

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `schema_version` | `string` | 当前为 `1.2` |
| `experiment_id` | `string` | UUID |
| `timestamp` | `string` | 记录写入时间 |
| `system` | `string` | `tfim` / `lih` 等 |
| `exp_name` | `string` | 实验名 |
| `seed` | `int \| null` | 当前 trial 的种子 |
| `n_qubits` | `int` | 量子比特数 |
| `ansatz_spec` | `dict` | 当前 ansatz 的结构化描述 |
| `optimizer_spec` | `dict \| null` | 优化器配置 |
| `measurement_spec` | `dict` | 观测量与 exact energy |
| `metrics` | `dict` | 运行指标 |
| `decision` | `string \| null` | keep / discard 等 |
| `parent_experiment` | `string \| null` | 父实验 ID |
| `change_summary` | `string \| null` | 变更摘要 |
| `comment` | `string \| null` | 补充备注 |
| `config_path_used` | `string \| null` | 实际使用配置路径 |
| `git_info` | `dict` | commit / dirty / diff_hash |
| `git_diff` | `string \| null` | 工作区 patch |
| `runtime_env` | `dict` | Python / OS / 关键依赖版本 |
| `artifact_paths` | `dict` | 报告与可视化产物路径 |

## 4. `metrics` 常见字段

当前实现常见字段包括：

- `val_energy`
- `exact_energy`
- `energy_error`
- `num_params`
- `two_qubit_gates`
- `runtime_sec`
- `actual_steps`

## 5. `artifact_paths` 当前实现

典型字段：

- `report_md`
- `circuit_png`
- `convergence_png`
- `circuit_json`

## 6. 目录约定

运行目录通常由 `prepare_experiment_dir(...)` 生成，为时间戳目录。例如：

- `experiments/lih/artifacts/runs/20260329_120000_lih_vqe/`
- `experiments/lih/artifacts/runs/20260329_120000_lih_geom_scan/`
- `experiments/tfim/20260329_120000_tfim_vqe/`

对应的 `results.jsonl` 就位于该运行目录中。

## 7. 注意事项

- `results.tsv` 是轻量摘要，不是完整审计来源。
- 严格审计、后处理和数据集化应优先使用 `results.jsonl`。
- `ResearchSession` 仍有独立日志语义，不应与本规范混为一谈。
