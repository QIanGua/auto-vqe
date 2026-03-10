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

# 启动 LiH 结构搜索
search-lih:
    uv run python experiments/lih/search.py

# 启动 TFIM 结构搜索
search-tfim:
    uv run python experiments/tfim/search.py

# 清理实验产生的临时文件 (日志, 结果, 报告, 图片)
clean:
    rm -f experiments/lih/*.log experiments/lih/*.json experiments/lih/*.png experiments/lih/*.md experiments/lih/results.tsv
    rm -f experiments/tfim/*.log experiments/tfim/*.json experiments/tfim/*.png experiments/tfim/*.md experiments/tfim/results.tsv
