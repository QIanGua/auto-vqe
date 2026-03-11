# Justfile for VQE Experiments

# 默认显示任务列表
default:
    @just --list

# 安装依赖
setup:
    uv sync

# 运行 LiH (Lithium Hydride) 实验
lih trials="2":
    uv run python experiments/lih/run.py

# 运行 TFIM (Transverse Field Ising Model) 实验
tfim trials="3":
    uv run python experiments/tfim/run.py

# 启动 LiH 结构搜索 (GA)
ga-lih:
    uv run python experiments/lih/ga_search.py

# 启动 TFIM 结构搜索 (GA)
ga-tfim:
    uv run python experiments/tfim/ga_search.py

# 启动 LiH 多维网格搜索 (MultiDim)
multidim-lih:
    uv run python experiments/lih/multidim/multidim_search.py

# 启动 TFIM 多维网格搜索 (MultiDim)
multidim-tfim:
    uv run python experiments/tfim/multidim/multidim_search.py

# 运行 LiH Baseline / Benchmark
baseline-lih:
    uv run python experiments/lih/baseline/baseline_run.py

# 运行 TFIM Baseline / Benchmark
baseline-tfim:
    uv run python experiments/tfim/baseline/baseline_run.py

# 运行所有单元测试 (Schema, Orchestration, Parameter Mapping)
test:
    uv run pytest tests/

# 运行参数映射基准测试 (Warm-start Verification)
benchmark:
    uv run python tests/benchmark_mapping.py

# 清理实验产生的临时文件 (日志, 结果, 报告, 图片, 审计产物)
clean:
    find experiments -type f \( -name "*.png" -o -name "*.jsonl" -o -name "*.tsv" -o -name "*.log" -o -name "*.patch" -o -name "report_*.md" -o -name "circuit_*.json" \) -delete
    find experiments -type d -name "[0-9]*_*" -exec rm -rf {} +
    find experiments -type d -name "audit" -exec rm -rf {} +
