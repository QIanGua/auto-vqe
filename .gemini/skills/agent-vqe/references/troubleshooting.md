# 故障排除与常见问题 (Troubleshooting)

本文档汇总 Agent-VQE 使用中的常见问题及解决方案。

## 1. 环境安装问题

### 1.1 NumPy 2.x 不兼容

**症状**：`ImportError` 或 `AttributeError`，提示 NumPy API 变更。

**原因**：项目依赖 `numpy>=1.24,<2.0`，不兼容 NumPy 2.x。

**解决**：
```bash
# 确保使用 uv 管理依赖（会自动锁定版本）
uv sync

# 验证版本
uv run python -c "import numpy; print(numpy.__version__)"
```

### 1.2 PySCF 安装失败

**症状**：`uv sync` 时 PySCF 编译失败。

**解决**：
```bash
# macOS 需要先安装 cmake
brew install cmake

# 重新安装
uv sync
```

### 1.3 TensorCircuit 版本冲突

**症状**：`ImportError: cannot import name 'xxx' from 'tensorcircuit'`

**原因**：TensorCircuit 版本必须在 `>=0.12.0,<0.16.0` 范围内。

**解决**：
```bash
uv sync  # 会使用 uv.lock 中锁定的精确版本
```

### 1.4 Python 版本不兼容

**症状**：`requires-python = ">=3.10,<3.13"` 导致安装拒绝。

**解决**：
```bash
# 使用 uv 安装兼容版本
uv python install 3.12
uv venv --python 3.12
uv sync
```

### 1.5 TensorCircuit Gates 属性缺失

**症状**：`AttributeError: module 'tensorcircuit.gates' has no attribute 'z'`

**原因**：不同 TensorCircuit 版本的 gate 访问方式不同。

**解决**：使用 `tc.gates.zgate()` 或 `tc.gates._z_matrix` 替代直接 `tc.gates.z`。参考 `experiments/*/env.py` 中的实际用法。

## 2. 运行时问题

### 2.1 实验输出目录找不到

**症状**：脚本报 FileNotFoundError 或未产生预期产物。

**原因**：`artifacts/runs/` 目录需要自动创建。

**排查**：
```bash
# 确认目录存在
ls experiments/lih/artifacts/runs/
ls experiments/tfim/artifacts/runs/

# 若不存在，运行一次实验自动生成
uv run python experiments/lih/run.py --trials 1
```

### 2.2 Agent 研究循环无法恢复

**症状**：`runtime.py` 每次都创建新的 session 而不是恢复。

**排查**：
```bash
# 检查 resume pointer
cat experiments/lih/artifacts/state/current_autoresearch_ga_session

# 验证指向的目录是否存在
ls -la $(cat experiments/lih/artifacts/state/current_autoresearch_ga_session)
```

**修复**：
```bash
# 手动设置 resume pointer 到正确的 session 目录
echo "/absolute/path/to/session_dir" > experiments/lih/artifacts/state/current_autoresearch_ga_session
```

### 2.3 Agent 立即停止且输出 "Target already reached"

**症状**：
```
*** Target already reached in prior runs. Best Energy Error: X.XXe-XX
```

**原因**：之前的运行已达到目标误差。

**解决**：
```bash
# 降低目标误差
uv run python core/research/runtime.py --dir experiments/lih --strategy ga --target 1e-8 --max 100

# 或清除之前的 session 状态
rm experiments/lih/artifacts/state/current_autoresearch_ga_session
```

### 2.4 Agent "Resume point exceeds max_loops"

**症状**：
```
*** Resume point iteration X exceeds max_loops=Y. Nothing to do.
```

**原因**：之前的运行已超过最大轮数。

**解决**：
```bash
# 增大 max_loops
uv run python core/research/runtime.py --dir experiments/lih --strategy ga --max 200

# 或重新开始一个 session
rm experiments/lih/artifacts/state/current_autoresearch_ga_session
```

## 3. 搜索策略问题

### 3.1 GA 搜索运行很慢

**可能原因**：
- `pop_size` 过大或 `generations` 过多
- `max_steps` 过高，单次评估耗时长
- `trials_per_config` 增加了重复评估次数

**缓解策略**：
- 先用较小预算（`pop_size=6, generations=3, max_steps=300`）做预筛
- 再用大预算验证 top 配置

### 3.2 ADAPT 策略梯度精度问题

**症状**：ADAPT 选择的算符看起来不合理，或过早停止。

**排查**：
- 检查 `gradient_epsilon` 是否太大（数值梯度步长）
- 检查 `gradient_tol` 是否太高（过早截断）
- 确认 `pool_config` 中的轨道配置正确

```python
# 调整示例
search_config = {
    "gradient_epsilon": 1e-4,   # 缩小步长提高精度
    "gradient_tol": 1e-5,       # 降低阈值延迟截断
    "max_adapt_steps": 10,      # 允许更多增长步
}
```

### 3.3 MultiDim 搜索组合爆炸

**症状**：搜索空间过大导致运行时间不可接受。

**解决**：
- 减少每个维度的选项数
- 先用小搜索空间定位关键维度
- 固定不敏感的维度，只搜索关键维度

## 4. 缓存与清理

### 4.1 清理 Python 编译缓存

```bash
just clean-py
```

这会清理：`__pycache__`, `.pytest_cache`, `*.pyc`, `.DS_Store`, `.coverage` 等。

### 4.2 清理实验数据

```bash
# ⚠️ 这会删除所有运行结果！
just clean-experiments
```

这会清理：`artifacts/runs/`, `artifacts/cache/`, `index.jsonl`, 所有运行 JSON/日志等。

### 4.3 全量清理

```bash
just clean
```

## 5. 测试相关

### 5.1 单独运行某个测试文件

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run pytest tests/test_research_agent.py -v
```

### 5.2 跳过慢测试

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run pytest -m "not slow" tests/
```

### 5.3 运行带覆盖率的测试

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run pytest --cov=core tests/
```

## 6. 常见 Python 导入问题

### 6.1 `ModuleNotFoundError: No module named 'core'`

**原因**：Python 路径未包含项目根目录。

**解决**：
```bash
# 方式 1：设置 PYTHONPATH
PYTHONPATH=. uv run python your_script.py

# 方式 2：使用项目入口脚本（它们已内置 sys.path 修正）
uv run python experiments/lih/run.py
uv run python core/research/runtime.py
```

### 6.2 循环导入

**排查**：检查是否在类型注解中使用了应该放在 `TYPE_CHECKING` 块中的导入。

## 7. 预算与停止控制

### SearchController 参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `max_runs` | 配置依赖 | 最大运行总次数 |
| `max_wall_clock_seconds` | None | 最大墙钟时间 |
| `no_improvement_limit` | 3 | 连续无改进触发动作 |
| `failure_limit` | 3 | 连续失败触发停止 |

### 调整预算

通过 CLI 或配置调整：
```bash
# 减少最大轮数
uv run python core/research/runtime.py --dir experiments/lih --strategy ga --max 20

# 收紧目标
uv run python core/research/runtime.py --dir experiments/lih --strategy ga --target 1e-8
```

## 8. 数据完整性

### 验证 run.json

```bash
# 检查某个运行的完整性
uv run python -c "
import json
with open('experiments/lih/artifacts/runs/<run_dir>/run.json') as f:
    r = json.load(f)
print(f'Schema: {r[\"schema_version\"]}')
print(f'Energy Error: {r[\"metrics\"][\"energy_error\"]}')
print(f'Git: {r[\"git_info\"][\"commit\"]}')
"
```

### 重新生成报告

```bash
uv run python core/evaluator/render_report.py --run-dir experiments/lih/artifacts/runs/<timestamp>_lih_vqe
uv run python core/evaluator/render_report.py --run-dir experiments/lih/artifacts/runs/<timestamp>_lih_vqe --markdown-only
uv run python core/evaluator/render_report.py --run-dir experiments/lih/artifacts/runs/<timestamp>_lih_vqe --recompute-if-missing
```
