import logging
from typing import Any, List, Literal, Optional

from core.evaluator.api import evaluate_candidate
from core.generator.base import GeneratorState, GeneratorStrategy
from core.representation.edits import apply_structure_edit
from core.model.schemas import (
    AnsatzSpec,
    CandidateSpec,
    EvaluationResult,
    EvaluationSpec,
    OperatorPoolSpec,
    StructureEdit,
)


class AdaptVQEStrategy(GeneratorStrategy):
    """
    ADAPT-VQE strategy expressed through the unified generator surface.
    """

    def __init__(
        self,
        env: Any,
        operator_pool: OperatorPoolSpec,
        selection_mode: Literal["greedy", "beam"] = "greedy",
        beam_width: int = 1,
        logger: Optional[logging.Logger] = None,
        controller: Optional[Any] = None,
    ):
        super().__init__(env=env, controller=controller, logger=logger, name="AdaptVQE")
        self.operator_pool = operator_pool
        self.selection_mode = selection_mode
        self.beam_width = beam_width

    def initialize(self) -> GeneratorState:
        state = GeneratorState(step_count=0)
        state.metadata["current_ansatz"] = AnsatzSpec(
            name="adapt_root",
            n_qubits=self.env.n_qubits,
            config={"family": "adapt"},
        )
        return state

    def propose(
        self,
        state: GeneratorState,
        budget: int = 1,
    ) -> List[CandidateSpec]:
        current_ansatz = state.metadata.get("current_ansatz")
        if current_ansatz is None:
            current_ansatz = AnsatzSpec(
                name="adapt_start",
                n_qubits=self.env.n_qubits,
                metadata={"strategy": "adapt"},
            )

        candidates = []
        for i, op in enumerate(self.operator_pool.operators):
            edit = StructureEdit(
                edit_type="append_operator",
                payload={"operator": op.model_dump()},
                reason=f"ADAPT trial op {op.name}",
            )
            new_ansatz = apply_structure_edit(current_ansatz, edit)
            new_ansatz.name = f"adapt_s{state.step_count}_op{i}"
            candidates.append(
                CandidateSpec(
                    candidate_id=new_ansatz.name,
                    parent_candidate_id=state.best_candidate_id,
                    ansatz=new_ansatz,
                    proposed_by=self.name,
                    structure_edit=edit,
                )
            )
            if len(candidates) >= budget:
                break
        return candidates

    def observe(
        self,
        state: GeneratorState,
        results: List[EvaluationResult],
    ) -> GeneratorState:
        if not results:
            return state

        best_res = min(
            results,
            key=lambda x: x.val_energy if x.val_energy is not None else float("inf"),
        )
        new_state = state.model_copy(deep=True)
        new_state.step_count += 1
        if new_state.best_score is None or (
            best_res.val_energy is not None and best_res.val_energy < new_state.best_score
        ):
            new_state.best_score = best_res.val_energy
            new_state.best_candidate_id = best_res.candidate_id
        return new_state

    def should_stop(self, state: GeneratorState) -> bool:
        return state.step_count >= 10

    def update(
        self,
        state: GeneratorState,
        results: List[EvaluationResult],
    ) -> GeneratorState:
        """Backward-compatible alias for the old ADAPT state update API."""
        return self.observe(state, results)

    def run(self):
        state = self.initialize()

        while not self.should_stop(state) and self.controller.should_continue():
            self.logger.info(f"--- ADAPT Step {state.step_count} ---")
            candidates = self.propose(state)
            if not candidates:
                break

            results = []
            eval_spec = EvaluationSpec(fidelity="quick", max_steps=50)  # type: ignore[arg-type]
            for cand in candidates:
                res = evaluate_candidate(self.env, cand, eval_spec, logger=self.logger)
                results.append(res)
                self.controller.report_result({"val_energy": res.val_energy})

            next_state = self.observe(state, results)
            best_res = next(
                (r for r in results if r.candidate_id == next_state.best_candidate_id),
                None,
            )
            if best_res is not None:
                best_cand = next(c for c in candidates if c.candidate_id == best_res.candidate_id)
                next_state.metadata["current_ansatz"] = best_cand.ansatz

            state = next_state
            if state.best_score is not None:
                self.logger.info(f"Step {state.step_count} Best Energy: {state.best_score:.6f}")

        current_ansatz = state.metadata["current_ansatz"]
        return {
            "best_results": {"val_energy": state.best_score},
            "best_config": current_ansatz.model_dump(),
            "ansatz_spec": current_ansatz,
        }
