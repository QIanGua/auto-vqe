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
    uv run python experiments/lih/run.py --trials {{trials}}

# 运行 TFIM (Transverse Field Ising Model) 实验
tfim trials="3":
    uv run python experiments/tfim/run.py --trials {{trials}}

# 启动 LiH 结构搜索 (GA)
ga-lih:
    uv run python experiments/lih/ga/search.py

# 启动 TFIM 结构搜索 (GA)
ga-tfim:
    uv run python experiments/tfim/ga/search.py

# 启动 LiH 多维网格搜索 (MultiDim)
multidim-lih:
    uv run python experiments/lih/multidim/search.py

# 启动 TFIM 多维网格搜索 (MultiDim)
multidim-tfim:
    uv run python experiments/tfim/multidim/search.py

# 运行 LiH Baseline / Benchmark
baseline-lih:
    uv run python experiments/lih/baseline/baseline_run.py

# 运行 TFIM Baseline / Benchmark
baseline-tfim:
    uv run python experiments/tfim/baseline/baseline_run.py

# 运行默认快速回归（不包含 slow 测试）
quick-test:
    PYTHONPATH=. uv run pytest -m "not slow" tests/

# 向后兼容：`just test` 默认等价于快速回归
test:
    just quick-test

# 运行所有单元测试，包含所有大型计算如 MPS
test-all:
    PYTHONPATH=. uv run pytest tests/

# 运行参数映射基准测试 (Warm-start Verification)
benchmark:
    uv run python tests/benchmark_mapping.py

# 清理实验产生的可再生产物与缓存，保留手工整理的 reports
clean:
    find experiments -type f \( -name "*.png" -o -name "*.jsonl" -o -name "*.tsv" -o -name "*.log" -o -name "*.patch" -o -name "report_*.md" -o -name "circuit_*.json" \) -delete
    find experiments -type d -name "[0-9]*_*" -exec rm -rf {} +
    find experiments -type d -name "audit" -exec rm -rf {} +
    find experiments -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ipynb_checkpoints" \) -exec rm -rf {} +
    find experiments -type f \( -name ".DS_Store" -o -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete
    find experiments -type d -path "*/artifacts/runs" -exec rm -rf {} +
    find experiments -type d -path "*/artifacts/cache" -exec rm -rf {} +
    find experiments -type f -path "*/artifacts/state/current_autoresearch_*" -delete
