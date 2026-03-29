"""
Auto-Search Orchestration Demo (TFIM)

演示如何使用 SearchOrchestrator 编排多个搜索策略：
1. GASearchStrategy (粗搜)
2. GridSearchStrategy (精扫最优结果附近的邻域)
"""

import argparse
import os
import sys

# 将项目根目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

def run_auto_search():
    from core.orchestration.controller import SearchController, SearchOrchestrator
    from core.evaluator.api import prepare_experiment_dir
    from core.generator.ga import GASearchStrategy
    from core.generator.grid import GridSearchStrategy
    from core.representation.compiler import build_ansatz
    from core.representation.search_space import generate_config_grid
    from experiments.tfim.env import ENV

    base_dir = os.path.dirname(os.path.dirname(__file__))
    exp_dir = prepare_experiment_dir(base_dir, "tfim_auto_search")
    
    # 共享控制器：共享预算
    controller = SearchController(
        max_runs=20, # 总共最多跑 20 轮实验
        no_improvement_limit=5, # 连续 5 轮没改进就换策略
        logger=None
    )

    # 1. 配置 GA 策略 (粗搜)
    ga_dimensions = {
        "layers": [1, 2, 3],
        "single_qubit_gates": [["ry"], ["ry", "rz"]],
        "two_qubit_gate": ["cnot", "rzz"],
        "entanglement": ["linear"],
    }
    
    ga_strategy = GASearchStrategy(
        env=ENV,
        make_circuit_fn=lambda cfg: build_ansatz(cfg, ENV.n_qubits),
        dimensions=ga_dimensions,
        pop_size=6,
        generations=3,
        exp_dir=exp_dir,
        base_exp_name="AutoSearch_Phase1_GA",
        controller=controller
    )

    # 2. 配置 Grid 策略 (精扫)
    # 假设我们想在 GA 之后，对一些特定的层数做更深度的扫描
    grid_dimensions = {
        "layers": [1, 2, 3, 4],
        "single_qubit_gates": [["ry", "rz"]],
        "two_qubit_gate": ["rzz"],
        "entanglement": ["linear", "brick"],
    }
    config_list = generate_config_grid(grid_dimensions)
    
    grid_strategy = GridSearchStrategy(
        env=ENV,
        make_create_circuit_fn=lambda cfg: build_ansatz(cfg, ENV.n_qubits),
        config_list=config_list,
        exp_dir=exp_dir,
        base_exp_name="AutoSearch_Phase2_Grid",
        trials_per_config=1,
        max_steps=400,
        controller=controller
    )

    # 3. 交给编排器运行
    orchestrator = SearchOrchestrator(
        generators=[ga_strategy, grid_strategy],
        controller=controller
    )

    results = orchestrator.run()
    print(f"\nAuto-Search Completed. Executed {len(results)} strategies.")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run TFIM orchestration auto-search demo.")


def main() -> None:
    build_parser().parse_args()
    run_auto_search()

if __name__ == "__main__":
    main()
