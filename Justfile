# Justfile for VQE Experiments

# 默认显示任务列表
default:
    @just --list

# 安装 git hooks 到本地
install-hooks:
    @echo "Installing pre-push hook..."
    @printf "#!/bin/bash\necho 'Running pre-push tests...'\njust test || { echo 'Tests failed! Push aborted.'; exit 1; }\n" > .git/hooks/pre-push
    @chmod +x .git/hooks/pre-push
    @echo "Hooks installed successfully."

# 安装依赖
setup:
    uv sync

# 运行 LiH (Lithium Hydride) 实验
lih trials="2":
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/lih/run.py --trials {{trials}}

# 运行 TFIM (Transverse Field Ising Model) 实验
tfim trials="3":
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/tfim/run.py --trials {{trials}}

# 启动 LiH 结构搜索 (GA)
ga-lih:
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/lih/run.py search ga

# 启动 TFIM 结构搜索 (GA)
ga-tfim:
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/tfim/run.py search ga

# 启动 LiH 多维网格搜索 (MultiDim)
multidim-lih:
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/lih/run.py search multidim

# 启动 TFIM 多维网格搜索 (MultiDim)
multidim-tfim:
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/tfim/run.py search multidim

# 运行 LiH Baseline / Benchmark
baseline-lih:
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/lih/run.py baseline

# 运行 TFIM Baseline / Benchmark
baseline-tfim:
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/tfim/run.py baseline

# 运行默认快速回归（不包含 slow 测试）
quick-test:
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run pytest -m "not slow" tests/

# 向后兼容：`just test` 默认等价于快速回归
test:
    just quick-test

# 运行所有单元测试，包含所有大型计算如 MPS
test-all:
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run pytest tests/

# 运行参数映射基准测试 (Warm-start Verification)
benchmark:
    PYTHONDONTWRITEBYTECODE=1 uv run python tests/benchmark_mapping.py

# --- 清理与维护 ---

# 清理 Python 编译缓存和系统垃圾 (全项目范围)
clean-py:
    @echo "Cleaning Python cache files..."
    @find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ipynb_checkpoints" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -exec rm -rf {} +
    @find . -type f \( -name ".DS_Store" -o -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" -o -name ".coverage" -o -name "coverage.xml" \) -delete
    @# 如果安装了 pyclean，也可以备选运行: pyclean . --debris
    @echo "Python cache cleaned."

# 清理实验产生的临时物与实验记录
clean-experiments:
    @echo "Cleaning experimental data and logs..."
    @find experiments -type f \( -name "*.png" -o -name "*.jsonl" -o -name "*.tsv" -o -name "*.log" -o -name "*.patch" -o -name "report_*.md" -o -name "circuit_*.json" -o -name "run.json" -o -name "config_snapshot.json" -o -name "index.jsonl" \) -delete
    @find experiments -type d -name "[0-9]*_*" -exec rm -rf {} +
    @find experiments -type d -name "audit" -exec rm -rf {} +
    @find experiments -type d -path "*/artifacts/runs" -exec rm -rf {} +
    @find experiments -type d -path "*/artifacts/cache" -exec rm -rf {} +
    @find experiments -type f -path "*/artifacts/state/current_autoresearch_*" -delete
    @rm -rf tmp_test_research_driver_session
    @echo "Experimental data cleaned."

# 合并所有清理任务
clean: clean-py clean-experiments
