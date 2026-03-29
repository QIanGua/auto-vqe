# Report Regeneration Guide

> 更新时间：2026-03-29
> 适用范围：已有实验运行目录的补图、补报告场景

## 1. 为什么需要这个工具

当前项目已经把这些重文件改成了默认不生成：

- `report_*.md`
- `circuit_*.png`
- `convergence_*.png`
- `circuit_*.json`

这样做的好处是：

- 大规模实验目录更轻
- 仓库里不默认堆积大量二进制产物
- 搜索阶段不会因为渲染额外文件而拖慢主流程

但写实验报告、汇报材料或论文时，我们仍然经常需要把这些图补出来。

## 2. 当前工具

统一入口：

- `core/evaluator/render_report.py`

基本用法：

```bash
uv run python core/evaluator/render_report.py --run-dir experiments/lih/artifacts/runs/<timestamp>_lih_vqe
```

只补 Markdown，不补图片：

```bash
uv run python core/evaluator/render_report.py --run-dir experiments/lih/artifacts/runs/<timestamp>_lih_vqe --markdown-only
```

如果是老运行目录，没有保存 `report_context.json`，则可以允许工具重算一次来恢复 `final_params`：

```bash
uv run python core/evaluator/render_report.py --run-dir experiments/lih/artifacts/runs/<timestamp>_lih_vqe --recompute-if-missing
```

## 3. 当前工作方式

新运行目录现在会额外保存一个轻量上下文文件：

- `report_context.json`

它里面包含：

- `final_params`
- `energy_history`

因此对新的运行目录，补图通常不需要重跑优化，只需要：

```text
run.json + report_context.json + ansatz_spec
  -> rebuild circuit
  -> regenerate markdown/assets
```

## 4. 对老运行目录怎么办

旧运行目录可能只有：

- `run.json`
- `config_snapshot.json`
- `run.log`

而没有 `report_context.json`。

这时工具会先报一个明确错误，提示需要：

- `--recompute-if-missing`

加上这个参数后，它会：

1. 根据 `run.json` 中的 `system` 反查对应 manifest
2. 用 `ansatz_spec` 重建 `create_circuit_fn`
3. 用该系统环境重跑一轮优化
4. 拿新的 `final_params` / `energy_history` 生成报告与图像

说明：

- 对旧运行目录，这种方式生成的图是“补算得到的图”
- 如果原始运行没有保存完整参数轨迹，就无法做到完全零重算恢复

## 5. 适合怎么用

推荐工作流：

### 5.1 大规模实验阶段

- 默认不开 `render_markdown`
- 默认不开 `render_assets`
- 只保留轻量指标和运行记录

### 5.2 准备写报告时

- 先挑出要引用的 run 目录
- 对这些 run 单独执行 `render_report.py`
- 只对少数关键实验补图补报告

这样既不会让实验主流程臃肿，也不会丢掉后续写报告的能力。

## 6. 当前边界

当前工具支持的系统来自 manifest 入口：

- `experiments/tfim/run.py`
- `experiments/lih/run.py`

也就是说，当前补图工具默认支持：

- `tfim`
- `lih`

如果后面新增别的系统，只需要让它也有对应 manifest，并在工具里注册即可。
