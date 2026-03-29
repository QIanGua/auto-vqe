import numpy as np
from typing import Any

from core.representation.compiler import build_circuit_from_ansatz
from core.model.schemas import AnsatzSpec, BlockSpec, OperatorSpec, WarmStartPlan


class ParameterMapper:
    """
    负责在新旧 AnsatzSpec 之间建立参数映射关系，实现 Warm-start。
    """

    def build_plan(
        self,
        old_ansatz: AnsatzSpec,
        new_ansatz: AnsatzSpec,
        old_params: np.ndarray,
        init_strategy: str = "zeros",
    ) -> WarmStartPlan:
        """
        对比新旧结构，生成参数重用计划。
        """
        _, old_count = build_circuit_from_ansatz(old_ansatz)
        _, new_count = build_circuit_from_ansatz(new_ansatz)

        reused = []
        initialized = []
        dropped = []
        min_blocks = min(len(old_ansatz.blocks), len(new_ansatz.blocks))

        old_idx = 0
        new_idx = 0

        for i in range(min_blocks):
            old_b = old_ansatz.blocks[i]
            new_b = new_ansatz.blocks[i]

            old_p_count = self._get_item_param_count(old_b)
            new_p_count = self._get_item_param_count(new_b)

            if type(old_b) == type(new_b) and old_p_count == new_p_count:
                for j in range(old_p_count):
                    reused.append((old_idx + j, new_idx + j))
                old_idx += old_p_count
                new_idx += new_p_count
            else:
                break

        for k in range(new_idx, new_count):
            initialized.append(k)

        for k in range(old_idx, old_count):
            dropped.append(k)

        return WarmStartPlan(
            old_param_count=old_count,
            new_param_count=new_count,
            reused_indices=reused,
            initialized_indices=initialized,
            dropped_indices=dropped,
            init_strategy=init_strategy,  # type: ignore[arg-type]
        )

    def apply_plan(
        self,
        plan: WarmStartPlan,
        old_params: np.ndarray,
    ) -> np.ndarray:
        """
        根据计划生成新的初始参数向量。
        """
        new_params = np.zeros(plan.new_param_count)

        for src, dst in plan.reused_indices:
            new_params[dst] = old_params[src]

        if plan.init_strategy == "small_random":
            for idx in plan.initialized_indices:
                new_params[idx] = np.random.normal(0, 0.01)
        elif plan.init_strategy == "zeros":
            for idx in plan.initialized_indices:
                new_params[idx] = 0.0

        return new_params

    def _get_item_param_count(self, item: Any) -> int:
        if isinstance(item, BlockSpec):
            return item.params_per_repeat * item.repetitions
        if isinstance(item, OperatorSpec):
            return 1
        return 0
