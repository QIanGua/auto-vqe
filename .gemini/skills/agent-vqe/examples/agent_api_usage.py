"""
Agent-VQE Agent API 使用示例

展示如何以编程方式使用 ResearchAgent 的各个组件。
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


def demo_memory_store():
    """演示 ResearchMemoryStore 的使用."""
    from core.research.memory_store import ResearchMemoryStore
    from core.model.research_schemas import (
        ActionSpec,
        DecisionRecord,
        HypothesisSpec,
        ResearchMemory,
        RunBundle,
    )

    print("=" * 60)
    print("ResearchMemoryStore 演示")
    print("=" * 60)

    # 使用临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchMemoryStore("experiments/lih", state_dir=tmpdir)

        # 1. 加载默认记忆
        memory = store.load()
        print(f"初始记忆:")
        print(f"  System: {memory.system}")
        print(f"  Objective: {memory.objective}")
        print(f"  Best error: {memory.best_energy_error}")
        print(f"  Dead ends: {memory.dead_ends}")

        # 2. 模拟一次 keep 决策
        action = ActionSpec(
            action_id="test-action-001",
            hypothesis_id="test-hyp-001",
            system_dir="experiments/lih",
            action_type="run_strategy",
            strategy_name="ga",
            fidelity="quick",
        )
        run = RunBundle(
            action=action,
            metrics={
                "energy_error": 0.005,
                "num_params": 12,
                "val_energy": -7.87,
                "exact_energy": -7.88,
                "runtime_sec": 15.3,
                "actual_steps": 200,
            },
            success=True,
            selected_config_path="experiments/lih/presets/ga.json",
        )
        decision = DecisionRecord(
            decision_id="test-action-001-decision",
            iteration=1,
            hypothesis_id="test-hyp-001",
            action_id="test-action-001",
            decision="keep",
            summary="Energy: 5.00e-03. Established the first baseline.",
            evidence_for=["No prior accepted result."],
            confidence=0.75,
            selected_config_path="experiments/lih/presets/ga.json",
        )

        # 3. 追加决策到 JSONL
        store.append_decision(decision, run)
        print(f"\n追加决策到 JSONL ✓")

        # 4. 应用决策更新记忆
        memory = store.apply_decision_to_memory(
            memory, decision, run,
            objective="优化 LiH 的 Ansatz 结构",
            strategy_name="ga",
        )
        store.save(memory)
        print(f"更新后记忆:")
        print(f"  Best error: {memory.best_energy_error}")
        print(f"  Best config: {memory.best_config_path}")
        print(f"  Strategy stats: {memory.strategy_stats}")

        # 5. 查看生成的文件
        print(f"\n生成的文件:")
        for f in os.listdir(tmpdir):
            path = os.path.join(tmpdir, f)
            size = os.path.getsize(path)
            print(f"  {f}: {size} bytes")

        # 6. 查看 Markdown 渲染
        md = store.render_markdown(memory)
        print(f"\n--- autoresearch.md 预览 ---")
        print(md[:500])
        print("...")


def demo_policy_engine():
    """演示 PolicyEngine 的假设生成和动作选择."""
    from core.research.policy import PolicyEngine
    from core.model.research_schemas import ResearchMemory
    from core.orchestration.controller import SearchController

    print("\n" + "=" * 60)
    print("PolicyEngine 演示")
    print("=" * 60)

    engine = PolicyEngine("ga", available_strategies=("ga", "multidim", "adapt", "qubit_adapt"))
    controller = SearchController()

    # 场景 1：首次运行（空记忆）
    memory = ResearchMemory(system="lih", objective="Optimize LiH ansatz.")

    hypothesis = engine.propose_hypothesis(
        memory, controller,
        iteration=1, system_dir="experiments/lih",
    )
    print(f"\n[场景 1: 首次运行]")
    print(f"  假设: {hypothesis.statement}")
    print(f"  搜索区域: {hypothesis.search_region}")

    action = engine.plan_next_action(
        memory, controller,
        hypothesis=hypothesis, system_dir="experiments/lih",
    )
    print(f"  动作类型: {action.action_type}")
    print(f"  策略: {action.strategy_name}")
    print(f"  精度: {action.fidelity}")

    # 场景 2：连续无改进后的策略切换
    memory.best_energy_error = 0.01
    memory.strategy_stats = {"ga": {"runs": 5, "keeps": 1, "discards": 4, "promotions": 0}}
    controller.consecutive_no_improvement = 4

    hypothesis2 = engine.propose_hypothesis(
        memory, controller,
        iteration=6, system_dir="experiments/lih",
    )
    action2 = engine.plan_next_action(
        memory, controller,
        hypothesis=hypothesis2, system_dir="experiments/lih",
    )
    print(f"\n[场景 2: 连续无改进]")
    print(f"  假设: {hypothesis2.statement}")
    print(f"  动作类型: {action2.action_type}")
    print(f"  目标策略: {action2.strategy_name}")
    print(f"  理由: {action2.rationale}")

    # 场景 3：查看替代策略排序
    alternatives = engine.get_alternative_strategies(memory)
    print(f"\n[替代策略排序]")
    for i, s in enumerate(alternatives, 1):
        stats = memory.strategy_stats.get(s, {})
        print(f"  {i}. {s} (runs={stats.get('runs', 0)}, discards={stats.get('discards', 0)})")


def demo_result_interpreter():
    """演示 ResultInterpreter 的失败分类."""
    from core.research.interpreter import ResultInterpreter
    from core.model.research_schemas import (
        ActionSpec, HypothesisSpec, ResearchMemory, RunBundle,
    )

    print("\n" + "=" * 60)
    print("ResultInterpreter 失败分类演示")
    print("=" * 60)

    interpreter = ResultInterpreter()

    hypothesis = HypothesisSpec(
        hypothesis_id="demo-hyp",
        system="lih",
        objective="Test",
        statement="Test hypothesis",
    )
    action = ActionSpec(
        action_id="demo-action",
        hypothesis_id="demo-hyp",
        system_dir="experiments/lih",
        action_type="run_strategy",
        strategy_name="ga",
    )

    # 场景列表
    scenarios = [
        ("执行失败", RunBundle(action=action, metrics={}, success=False, error_message="Timeout")),
        ("梯度塌缩", RunBundle(action=action, metrics={"energy_error": 0.01, "num_params": 12, "gradient_norm": 1e-10}, success=True)),
        ("参数效率低", RunBundle(action=action, metrics={"energy_error": 0.009, "num_params": 20}, success=True)),
        ("首个结果 (keep)", RunBundle(action=action, metrics={"energy_error": 0.005, "num_params": 12}, success=True)),
    ]

    for name, run in scenarios:
        memory = ResearchMemory(system="lih", objective="Test")
        if name in ("参数效率低",):
            memory.best_energy_error = 0.01
            memory.best_num_params = 12

        decision = interpreter.interpret_run(
            iteration=1, memory=memory, hypothesis=hypothesis, run=run,
        )
        print(f"\n[{name}]")
        print(f"  决策:     {decision.decision}")
        print(f"  失败类型: {decision.failure_type or 'None'}")
        print(f"  摘要:     {decision.summary}")
        print(f"  置信度:   {decision.confidence}")
        if decision.followup_actions:
            print(f"  后续:     {decision.followup_actions}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent-VQE Agent API 使用示例")
    parser.add_argument(
        "--demo",
        choices=["memory", "policy", "interpreter", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.demo in ("memory", "all"):
        demo_memory_store()
    if args.demo in ("policy", "all"):
        demo_policy_engine()
    if args.demo in ("interpreter", "all"):
        demo_result_interpreter()
