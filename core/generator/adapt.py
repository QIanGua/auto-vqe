import logging
from itertools import combinations, product
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch

from baselines.uccsd import build_excitation_operator_pool
from core.evaluator.training import optimize_parameters
from core.generator.base import GeneratorState, GeneratorStrategy
from core.model.schemas import (
    AnsatzSpec,
    CandidateSpec,
    EvaluationResult,
    OperatorPoolSpec,
    OperatorSpec,
    OptimizerSpec,
    StructureEdit,
)
from core.representation.compiler import build_circuit_from_ansatz
from core.representation.edits import apply_structure_edit


def _operator_key(op: OperatorSpec) -> str:
    return f"{op.family}|{op.name}|{op.generator}|{tuple(op.support_qubits)}"


def _default_initial_config(env: Any) -> Dict[str, Any]:
    if getattr(env, "name", "").startswith("LiH"):
        n_qubits = getattr(env, "n_qubits", 4)
        return {
            "init_state": "hf",
            "hf_qubits": list(range(n_qubits // 2)),
        }
    return {"init_state": "zero"}


def build_qubit_adapt_pool(
    n_qubits: int,
    *,
    max_body: int = 2,
    include_single_qubit: bool = True,
) -> OperatorPoolSpec:
    """
    Build a small qubit-operator pool suitable for Qubit-ADAPT-VQE.

    The pool uses Pauli exponentials with X/Y support only, which is enough to
    produce non-trivial gradients from computational-basis reference states.
    """
    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")

    operators: List[OperatorSpec] = []
    if include_single_qubit:
        for qubit in range(n_qubits):
            operators.append(
                OperatorSpec(
                    name=f"Y{qubit}",
                    family="pauli_exp",
                    support_qubits=[qubit],
                    generator=f"Y{qubit}",
                    metadata={"paulis": ["Y"]},
                )
            )

    if max_body >= 2:
        for i, j in combinations(range(n_qubits), 2):
            for paulis in product(("X", "Y"), repeat=2):
                if all(p == "X" for p in paulis):
                    continue
                label = f"{paulis[0]}{i} {paulis[1]}{j}"
                operators.append(
                    OperatorSpec(
                        name=label.replace(" ", "_"),
                        family="pauli_exp",
                        support_qubits=[i, j],
                        generator=label,
                        metadata={"paulis": list(paulis)},
                    )
                )

    return OperatorPoolSpec(
        name="qubit_adapt_pool",
        operators=operators,
        metadata={"max_body": max_body, "include_single_qubit": include_single_qubit},
    )


def build_fermionic_adapt_pool(env: Any, config: Dict[str, Any] | None = None) -> OperatorPoolSpec:
    return build_excitation_operator_pool(env, config)


class AdaptVQEStrategy(GeneratorStrategy):
    """
    Real ADAPT-style ansatz growth based on operator-gradient ranking.

    The strategy supports both:
    - fermionic excitation pools (ADAPT-VQE)
    - qubit Pauli pools (Qubit-ADAPT-VQE)
    """

    def __init__(
        self,
        env: Any,
        operator_pool: OperatorPoolSpec,
        selection_mode: Literal["greedy", "beam"] = "greedy",
        beam_width: int = 1,
        gradient_epsilon: float = 1e-3,
        gradient_tol: float = 1e-4,
        max_adapt_steps: int = 10,
        optimizer_spec: Optional[OptimizerSpec] = None,
        initial_ansatz: Optional[AnsatzSpec] = None,
        logger: Optional[logging.Logger] = None,
        controller: Optional[Any] = None,
    ):
        super().__init__(env=env, controller=controller, logger=logger, name="AdaptVQE")
        self.operator_pool = operator_pool
        self.selection_mode = selection_mode
        self.beam_width = beam_width
        self.gradient_epsilon = gradient_epsilon
        self.gradient_tol = gradient_tol
        self.max_adapt_steps = max_adapt_steps
        self.optimizer_spec = optimizer_spec or OptimizerSpec(lr=0.05, max_steps=200)
        self.initial_ansatz = initial_ansatz

    def initialize(self) -> GeneratorState:
        state = GeneratorState(step_count=0)
        current_ansatz = self.initial_ansatz or AnsatzSpec(
            name="adapt_root",
            family="adapt",
            n_qubits=self.env.n_qubits,
            config=_default_initial_config(self.env),
            metadata={"strategy": "adapt"},
        )
        state.metadata["current_ansatz"] = current_ansatz
        _, param_count = build_circuit_from_ansatz(current_ansatz)
        state.metadata["current_params"] = np.zeros(param_count, dtype=np.float32)
        state.metadata["selected_operator_keys"] = []
        state.metadata["gradient_history"] = []
        return state

    def _energy_for_ansatz(self, ansatz: AnsatzSpec, params: np.ndarray) -> float:
        create_circuit_fn, _ = build_circuit_from_ansatz(ansatz)
        params_tensor = torch.tensor(params, dtype=torch.float32)
        with torch.no_grad():
            circuit, _ = create_circuit_fn(params_tensor)
            energy = self.env.compute_energy(circuit)
        return float(energy.item() if hasattr(energy, "item") else energy)

    def _optimize_ansatz(
        self,
        ansatz: AnsatzSpec,
        init_params: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        create_circuit_fn, num_params = build_circuit_from_ansatz(ansatz)

        if num_params == 0:
            final_params = np.zeros(0, dtype=np.float32)
            val_energy = self._energy_for_ansatz(ansatz, final_params)
            return {
                "val_energy": val_energy,
                "energy_error": abs(val_energy - self.env.exact_energy) if hasattr(self.env, "exact_energy") else None,
                "num_params": 0,
                "actual_steps": 0,
                "final_params": final_params,
                "energy_history": [val_energy],
            }

        return optimize_parameters(
            env=self.env,
            ansatz=ansatz,
            optimizer_spec=self.optimizer_spec,
            init_params=init_params,
            logger=None,
        )

    def _candidate_gradient(
        self,
        current_ansatz: AnsatzSpec,
        current_params: np.ndarray,
        op: OperatorSpec,
    ) -> float:
        edit = StructureEdit(
            edit_type="append_operator",
            payload={"operator": op.model_dump()},
            reason=f"ADAPT gradient probe for {op.name}",
        )
        candidate_ansatz = apply_structure_edit(current_ansatz, edit)

        plus_params = np.concatenate([current_params, np.array([self.gradient_epsilon], dtype=np.float32)])
        minus_params = np.concatenate([current_params, np.array([-self.gradient_epsilon], dtype=np.float32)])
        e_plus = self._energy_for_ansatz(candidate_ansatz, plus_params)
        e_minus = self._energy_for_ansatz(candidate_ansatz, minus_params)
        return (e_plus - e_minus) / (2.0 * self.gradient_epsilon)

    def propose(
        self,
        state: GeneratorState,
        budget: int = 1,
    ) -> List[CandidateSpec]:
        current_ansatz = state.metadata.get("current_ansatz")
        if current_ansatz is None:
            return []

        current_params = np.asarray(state.metadata.get("current_params", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        selected_keys = set(state.metadata.get("selected_operator_keys", []))

        ranked: List[tuple[float, OperatorSpec]] = []
        for op in self.operator_pool.operators:
            key = _operator_key(op)
            if key in selected_keys:
                continue
            gradient = self._candidate_gradient(current_ansatz, current_params, op)
            ranked.append((gradient, op))

        ranked.sort(key=lambda item: abs(item[0]), reverse=True)
        state.metadata["last_gradient_ranking"] = [
            {"operator": op.name, "gradient": grad, "abs_gradient": abs(grad)}
            for grad, op in ranked
        ]

        candidates = []
        for gradient, op in ranked[:budget]:
            edit = StructureEdit(
                edit_type="append_operator",
                payload={"operator": op.model_dump()},
                reason=f"ADAPT selected op {op.name} (grad={gradient:.3e})",
            )
            new_ansatz = apply_structure_edit(current_ansatz, edit)
            new_ansatz.name = f"adapt_s{state.step_count}_{op.name}"
            new_ansatz.metadata["selection_gradient"] = gradient
            new_ansatz.metadata["selection_abs_gradient"] = abs(gradient)
            candidates.append(
                CandidateSpec(
                    candidate_id=new_ansatz.name,
                    parent_candidate_id=state.best_candidate_id,
                    ansatz=new_ansatz,
                    proposed_by=self.name,
                    structure_edit=edit,
                    metadata={"selection_gradient": gradient, "operator_key": _operator_key(op)},
                )
            )
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
        if state.step_count >= self.max_adapt_steps:
            return True
        ranking = state.metadata.get("last_gradient_ranking") or []
        if ranking and float(ranking[0]["abs_gradient"]) < self.gradient_tol:
            return True
        return False

    def update(
        self,
        state: GeneratorState,
        results: List[EvaluationResult],
    ) -> GeneratorState:
        return self.observe(state, results)

    def run(self):
        state = self.initialize()

        current_ansatz = state.metadata["current_ansatz"]
        init_params = np.asarray(state.metadata.get("current_params", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        baseline = self._optimize_ansatz(current_ansatz, init_params)
        state.best_score = baseline["val_energy"]
        state.metadata["current_params"] = np.asarray(baseline["final_params"], dtype=np.float32)

        while not self.should_stop(state) and self.controller.should_continue():
            self.logger.info(f"--- ADAPT Step {state.step_count} ---")
            candidates = self.propose(state, budget=self.beam_width if self.selection_mode == "beam" else 1)
            ranking = state.metadata.get("last_gradient_ranking") or []
            if ranking:
                top = ranking[0]
                self.logger.info(
                    "Top gradient: %s grad=%.6e",
                    top["operator"],
                    top["gradient"],
                )
            if not candidates:
                break
            if ranking and float(ranking[0]["abs_gradient"]) < self.gradient_tol:
                self.logger.info("ADAPT stopping: max gradient below threshold %.3e", self.gradient_tol)
                break

            results: List[EvaluationResult] = []
            current_params = np.asarray(state.metadata["current_params"], dtype=np.float32)
            for cand in candidates:
                warm_params = np.concatenate([current_params, np.zeros(1, dtype=np.float32)])
                opt_results = self._optimize_ansatz(cand.ansatz, warm_params)
                gradient = float(cand.metadata.get("selection_gradient", 0.0))
                results.append(
                    EvaluationResult(
                        candidate_id=cand.candidate_id,
                        fidelity="full",
                        success=True,
                        val_energy=opt_results["val_energy"],
                        energy_error=opt_results.get("energy_error"),
                        num_params=int(opt_results["num_params"]),
                        two_qubit_gates=0,
                        runtime_sec=0.0,
                        actual_steps=int(opt_results["actual_steps"]),
                        proxy_score=abs(gradient),
                        artifacts={},
                    )
                )
                cand.metadata["optimized_params"] = np.asarray(opt_results["final_params"], dtype=np.float32).tolist()
                cand.metadata["optimized_energy"] = float(opt_results["val_energy"])
                self.controller.report_result(
                    {"val_energy": opt_results["val_energy"], "num_params": opt_results["num_params"]}
                )

            next_state = self.observe(state, results)
            best_res = next(
                (r for r in results if r.candidate_id == next_state.best_candidate_id),
                None,
            )
            if best_res is not None:
                best_cand = next(c for c in candidates if c.candidate_id == best_res.candidate_id)
                next_state.metadata["current_ansatz"] = best_cand.ansatz
                next_state.metadata["current_params"] = np.asarray(best_cand.metadata["optimized_params"], dtype=np.float32)
                selected_keys = list(next_state.metadata.get("selected_operator_keys", []))
                selected_keys.append(str(best_cand.metadata["operator_key"]))
                next_state.metadata["selected_operator_keys"] = selected_keys
                grad_hist = list(next_state.metadata.get("gradient_history", []))
                grad_hist.append(
                    {
                        "step": next_state.step_count,
                        "operator": best_cand.ansatz.blocks[-1].name,
                        "gradient": float(best_cand.metadata["selection_gradient"]),
                        "energy": float(best_res.val_energy) if best_res.val_energy is not None else None,
                    }
                )
                next_state.metadata["gradient_history"] = grad_hist

            state = next_state
            if state.best_score is not None:
                self.logger.info(f"Step {state.step_count} Best Energy: {state.best_score:.6f}")

        current_ansatz = state.metadata["current_ansatz"]
        return {
            "best_results": {
                "val_energy": state.best_score,
                "gradient_history": state.metadata.get("gradient_history", []),
            },
            "best_config": current_ansatz.model_dump(),
            "ansatz_spec": current_ansatz,
        }
