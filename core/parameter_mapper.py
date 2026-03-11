import numpy as np
from typing import Dict, Any, List, Optional
from core.schemas import AnsatzSpec, BlockSpec, OperatorSpec, WarmStartPlan

class ParameterMapper:
    """
    负责在新旧 AnsatzSpec 之间建立参数映射关系，实现 Warm-start。
    """
    
    def build_plan(
        self,
        old_ansatz: AnsatzSpec,
        new_ansatz: AnsatzSpec,
        old_params: np.ndarray,
        init_strategy: str = "zeros"
    ) -> WarmStartPlan:
        """
        对比新旧结构，生成参数重用计划。
        """
        # 计算新旧参数总量
        # 注意: 这里需要与 circuit_factory.build_circuit_from_ansatz 的计数逻辑一致
        from core.circuit_factory import build_circuit_from_ansatz
        _, old_count = build_circuit_from_ansatz(old_ansatz)
        _, new_count = build_circuit_from_ansatz(new_ansatz)
        
        reused = []
        initialized = []
        dropped = []
        
        # 简单策略: 比较 blocks 列表
        # 如果新结构只是在末尾增加了 block，则前段参数完全复用
        # 这是一个 MVP 实现，后续可根据 block name/id 做更精细的 diff
        
        # TODO: 实现更通用的 diff 算法
        # 目前只处理 append 场景：
        min_blocks = min(len(old_ansatz.blocks), len(new_ansatz.blocks))
        
        old_idx = 0
        new_idx = 0
        
        # 假设前 min_blocks 个 block 没变
        for i in range(min_blocks):
            old_b = old_ansatz.blocks[i]
            new_b = new_ansatz.blocks[i]
            
            # 这里简单判断类型和参数量是否一致
            old_p_count = self._get_item_param_count(old_b)
            new_p_count = self._get_item_param_count(new_b)
            
            if type(old_b) == type(new_b) and old_p_count == new_p_count:
                # 记录映射
                for j in range(old_p_count):
                    reused.append((old_idx + j, new_idx + j))
                old_idx += old_p_count
                new_idx += new_p_count
            else:
                # 结构发生冲突，停止简单复用
                break
        
        # 剩余的新参数需要初始化
        for k in range(new_idx, new_count):
            initialized.append(k)
            
        # 剩余的旧参数被丢弃
        for k in range(old_idx, old_count):
            dropped.append(k)
            
        return WarmStartPlan(
            old_param_count=old_count,
            new_param_count=new_count,
            reused_indices=reused,
            initialized_indices=initialized,
            dropped_indices=dropped,
            init_strategy=init_strategy # type: ignore
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
        
        # 1. 复用旧参数
        for src, dst in plan.reused_indices:
            new_params[dst] = old_params[src]
            
        # 2. 对新增参数应用初始化策略
        if plan.init_strategy == "small_random":
            for idx in plan.initialized_indices:
                new_params[idx] = np.random.normal(0, 0.01)
        elif plan.init_strategy == "zeros":
            for idx in plan.initialized_indices:
                new_params[idx] = 0.0
        # 其他策略如 copy_neighbor 可在后续补齐
        
        return new_params

    def _get_item_param_count(self, item: Any) -> int:
        if isinstance(item, BlockSpec):
            return item.params_per_repeat * item.repetitions
        elif isinstance(item, OperatorSpec):
            return 1 # MVP 假设
        return 0
