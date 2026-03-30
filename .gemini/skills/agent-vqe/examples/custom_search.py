"""
Agent-VQE 自定义搜索示例

展示如何以编程方式配置并运行自定义的 Ansatz 结构搜索。
"""
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


def custom_ga_search_tfim():
    """使用自定义参数运行 TFIM GA 搜索."""
    from core.generator.ga import GASearch
    from core.evaluator.training import vqe_train
    from core.representation.compiler import build_ansatz
    from experiments.tfim.env import ENV

    print("=" * 60)
    print("自定义 GA 搜索 (TFIM)")
    print("=" * 60)

    # 定义自定义搜索空间（比默认更小，用于快速演示）
    search_space = {
        "layers": [1, 2, 3],
        "single_qubit_gates": [["ry"], ["ry", "rz"]],
        "two_qubit_gate": ["cnot", "rzz"],
        "entanglement": ["linear", "ring"],
    }

    print(f"搜索空间: {json.dumps(search_space, indent=2)}")

    # 初始化 GA 搜索
    ga = GASearch(
        search_space=search_space,
        pop_size=6,
        mutation_rate=0.3,
        elite_count=2,
    )

    # 适应度评估函数
    def evaluate_config(config: dict) -> float:
        """返回 energy_error 作为适应度（越低越好）."""
        try:
            create_circuit, num_params = build_ansatz(config, ENV.n_qubits)

            def compute_energy_fn(params):
                c, _ = create_circuit(params)
                return ENV.compute_energy(c)

            results = vqe_train(
                create_circuit_fn=create_circuit,
                compute_energy_fn=compute_energy_fn,
                n_qubits=ENV.n_qubits,
                exact_energy=ENV.exact_energy,
                num_params=num_params,
                max_steps=200,  # 快速评估
                lr=0.05,
            )
            return results.get("energy_error", float("inf"))
        except Exception as e:
            print(f"  评估失败: {e}")
            return float("inf")

    # 运行 GA
    best_config = None
    best_error = float("inf")

    generations = 3
    for gen in range(generations):
        population = ga.get_population()
        print(f"\n--- Generation {gen + 1}/{generations} (pop_size={len(population)}) ---")

        fitness_scores = []
        for i, config in enumerate(population):
            error = evaluate_config(config)
            fitness_scores.append(error)
            if error < best_error:
                best_error = error
                best_config = config.copy()
            print(f"  个体 {i + 1}: error={error:.4e}, config={config}")

        ga.evolve(fitness_scores)

    print(f"\n{'=' * 60}")
    print(f"最优配置: {json.dumps(best_config, indent=2)}")
    print(f"最优误差: {best_error:.4e}")
    print(f"{'=' * 60}")

    return best_config, best_error


def compare_configs():
    """对比两个 Ansatz 配置的性能."""
    from core.representation.compiler import build_ansatz
    from core.evaluator.training import vqe_train
    from experiments.tfim.env import ENV

    configs = [
        {"name": "simple", "layers": 1, "single_qubit_gates": ["ry"], "two_qubit_gate": "cnot", "entanglement": "linear"},
        {"name": "complex", "layers": 3, "single_qubit_gates": ["ry", "rz"], "two_qubit_gate": "rzz", "entanglement": "ring"},
    ]

    print("=" * 60)
    print("Ansatz 配置对比 (TFIM)")
    print("=" * 60)

    for cfg in configs:
        name = cfg.pop("name")
        create_circuit, num_params = build_ansatz(cfg, ENV.n_qubits)

        def compute_energy_fn(params):
            c, _ = create_circuit(params)
            return ENV.compute_energy(c)

        results = vqe_train(
            create_circuit_fn=create_circuit,
            compute_energy_fn=compute_energy_fn,
            n_qubits=ENV.n_qubits,
            exact_energy=ENV.exact_energy,
            num_params=num_params,
            max_steps=500,
            lr=0.05,
        )

        print(f"\n[{name}]")
        print(f"  Config:       {cfg}")
        print(f"  Num params:   {num_params}")
        print(f"  Energy error: {results['energy_error']:.4e}")
        print(f"  Val energy:   {results['val_energy']:.8f}")
        print(f"  Steps:        {results['actual_steps']}")
        print(f"  Time:         {results['runtime_sec']:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent-VQE 自定义搜索示例")
    parser.add_argument("--mode", choices=["ga", "compare"], default="compare")
    args = parser.parse_args()

    if args.mode == "ga":
        custom_ga_search_tfim()
    else:
        compare_configs()
