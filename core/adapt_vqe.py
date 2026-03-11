import numpy as np
import logging
from typing import List, Optional, Literal, Dict, Any
from core.strategy_base import SearchStrategy
from core.schemas import (
    AnsatzSpec, CandidateSpec, EvaluationResult, EvaluationSpec, 
    OperatorPoolSpec, StructureEdit, StrategyCheckpoint
)

class AdaptVQEStrategy(SearchStrategy):
    """
    ADAPT-VQE 策略 MVP。
    按轮次从算符池中选择梯度最大（或能量下降最多）的算符添加到线路中。
    """
    
    def __init__(
        self,
        env: Any,
        operator_pool: OperatorPoolSpec,
        selection_mode: Literal["greedy", "beam"] = "greedy",
        beam_width: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        self.env = env
        self.operator_pool = operator_pool
        self.selection_mode = selection_mode
        self.beam_width = beam_width
        self.logger = logger or logging.getLogger(__name__)
        self.name = "AdaptVQE"

    def initialize(self, ctx: Any = None) -> StrategyCheckpoint:
        """
        初始化策略状态。
        """
        return StrategyCheckpoint(
            strategy_name=self.name,
            step_count=0,
            best_candidate_id=None,
            best_energy=None,
            internal_state={"current_step": 0}
        )

    def propose(
        self,
        state: StrategyCheckpoint,
        budget: int = 1,
    ) -> List[CandidateSpec]:
        """
        根据当前状态，提议一组候选结构（将池中每个算符分别尝试添加一次）。
        """
        # 这里 MVP 实现：获取当前最佳或初始结构
        # 假设 ctx 中包含当前最佳 ansatz
        current_ansatz = state.internal_state.get("current_ansatz")
        if current_ansatz is None:
            # 初始空线路
            current_ansatz = AnsatzSpec(
                name="adapt_start", 
                n_qubits=self.env.n_qubits,
                metadata={"strategy": "adapt"}
            )
            
        candidates = []
        for i, op in enumerate(self.operator_pool.operators):
            # 构造结构编辑指令
            edit = StructureEdit(
                edit_type="append_operator",
                payload={"operator": op.model_dump()},
                reason=f"ADAPT trial op {op.name}"
            )
            
            from core.circuit_factory import apply_structure_edit
            new_ansatz = apply_structure_edit(current_ansatz, edit)
            new_ansatz.name = f"adapt_s{state.step_count}_op{i}"
            
            cand = CandidateSpec(
                candidate_id=new_ansatz.name,
                parent_candidate_id=state.best_candidate_id,
                ansatz=new_ansatz,
                proposed_by=self.name,
                structure_edit=edit
            )
            candidates.append(cand)
            
            if len(candidates) >= budget:
                break
                
        return candidates

    def update(
        self,
        state: StrategyCheckpoint,
        results: List[EvaluationResult],
    ) -> StrategyCheckpoint:
        """
        根据评估结果更新状态（选择本轮最佳候选）。
        """
        if not results:
            return state
            
        # 简单贪心选择：取能量最低的
        best_res = min(results, key=lambda x: x.val_energy if x.val_energy is not None else float('inf'))
        
        new_state = state.model_copy(deep=True)
        new_state.step_count += 1
        
        if state.best_energy is None or best_res.val_energy < state.best_energy:
            new_state.best_energy = best_res.val_energy
            new_state.best_candidate_id = best_res.candidate_id
            
        # TODO: 更新 current_ansatz 供下一轮提议使用
        # 这需要从 results 关联回 ansatz，MVP 假设外部 orchestrator 处理
        return new_state

    def should_stop(self, state: StrategyCheckpoint) -> bool:
        """
        判断是否停止生长（能量收敛或达到最大步数）。
        """
        if state.step_count >= 10: 
            return True
        return False

    def run(self) -> Dict[str, Any]:
        """
        ADAPT-VQE 核心生长循环。
        """
        from core.engine import evaluate_candidate
        
        state = self.initialize()
        current_ansatz = AnsatzSpec(
            name="adapt_root", 
            n_qubits=self.env.n_qubits,
            config={"family": "adapt"}
        )
        state.internal_state["current_ansatz"] = current_ansatz

        while not self.should_stop(state) and self.controller.should_continue():
            self.logger.info(f"--- ADAPT Step {state.step_count} ---")
            
            # 1. 提议候选
            candidates = self.propose(state)
            if not candidates:
                break
                
            # 2. 评估候选 (MVP 使用 quick 模式计算梯度/能量)
            results = []
            eval_spec = EvaluationSpec(fidelity="quick", max_steps=50) # type: ignore
            for cand in candidates:
                res = evaluate_candidate(self.env, cand, eval_spec, logger=self.logger)
                results.append(res)
                self.controller.report_result({"val_energy": res.val_energy})
            
            # 3. 更新状态 (选择最佳算符)
            state = self.update(state, results)
            
            # 4. 持久化当前最佳结构供下一轮使用
            best_res = next(r for r in results if r.candidate_id == state.best_candidate_id)
            best_cand = next(c for c in candidates if c.candidate_id == best_res.candidate_id)
            state.internal_state["current_ansatz"] = best_cand.ansatz
            
            self.logger.info(f"Step {state.step_count} Best Energy: {state.best_energy:.6f}")

        return {
            "best_results": {"val_energy": state.best_energy},
            "best_config": state.internal_state["current_ansatz"].model_dump(),
            "ansatz_spec": state.internal_state["current_ansatz"]
        }
