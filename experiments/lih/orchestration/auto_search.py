"""
Auto-Search Orchestration (LiH)

演示如何使用 SearchOrchestrator 编排多个搜索策略：
1. GASearchStrategy (粗搜：在广阔的 ansatz 空间中寻找潜力结构)
2. GridSearchStrategy (精扫：在最有潜力的结构附近进行细致验证)
"""

import argparse
import os
import sys

# 将项目根目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "runs")

# LiH 特有的线路构造包装：确保包含必要的 HF 占据轨道配置
def make_lih_circuit_fn(config):
    from core.representation.compiler import build_ansatz
    from experiments.lih.env import ENV

    n_qubits = ENV.n_qubits
    hf_qubits = [0, 1] # LiH 在 STO-3G 下 active space 常规设置
    if config.get("init_state") == "hf" and "hf_qubits" not in config:
        config = {**config, "hf_qubits": hf_qubits}
    return build_ansatz(config, n_qubits)

def run_auto_search():
    from core.orchestration.controller import SearchController, SearchOrchestrator
    from core.evaluator.api import prepare_experiment_dir
    from core.generator.ga import GASearchStrategy
    from core.generator.grid import GridSearchStrategy
    from core.representation.search_space import generate_config_grid
    from experiments.lih.env import ENV

    exp_dir = prepare_experiment_dir(RUNS_DIR, "lih_auto_search")
    
    # 1. 初始化控制器：设置全局预算和停止规则
    controller = SearchController(
        max_runs=15,             # 总实验轮次限制 (LiH 耗时较长，初始设为 15)
        no_improvement_limit=3,  # 连续 3 轮无改进则触发策略切换
        failure_limit=3,         # 连续 3 轮失败则报错
        logger=None
    )

    # 2. 阶段一：GA 遗传算法 (粗筛探索)
    ga_dimensions = {
        "init_state": ["hf"],
        "layers": [2, 3, 4],
        "single_qubit_gates": [["ry"], ["ry", "rz"]],
        "two_qubit_gate": ["cnot", "rzz"],
        "entanglement": ["linear", "ring"],
    }
    
    ga_strategy = GASearchStrategy(
        env=ENV,
        make_circuit_fn=make_lih_circuit_fn,
        dimensions=ga_dimensions,
        pop_size=6,
        generations=2,
        exp_dir=exp_dir,
        base_exp_name="AutoSearch_LiH_Phase1_GA",
        controller=controller
    )

    # 3. 阶段二：Grid 网格扫描 (精扫验证)
    # 在第二阶段，我们可以针对某些已知表现良好的配置进行更深度的网格扫描
    grid_dimensions = {
        "init_state": ["hf"],
        "layers": [2, 3],
        "single_qubit_gates": [["ry", "rz"]],
        "two_qubit_gate": ["rzz", "rxx_ryy_rzz"], # 尝试更复杂的门集
        "entanglement": ["linear", "full"],
    }
    config_list = generate_config_grid(grid_dimensions)
    
    grid_strategy = GridSearchStrategy(
        env=ENV,
        make_create_circuit_fn=make_lih_circuit_fn,
        config_list=config_list,
        exp_dir=exp_dir,
        base_exp_name="AutoSearch_LiH_Phase2_Grid",
        trials_per_config=1,
        max_steps=600,
        controller=controller
    )

    # 4. 驱动编排器执行
    orchestrator = SearchOrchestrator(
        generators=[ga_strategy, grid_strategy],
        controller=controller
    )

    results = orchestrator.run()
    print(f"\nAuto-Search Completed. Executed {len(results)} strategies.")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run LiH orchestration auto-search demo.")


def main() -> None:
    build_parser().parse_args()
    run_auto_search()

if __name__ == "__main__":
    main()
