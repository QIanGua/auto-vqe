import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.model.schemas import CandidateSpec, EvaluationResult
from core.generator.strategy import SearchStrategy


class GeneratorState(BaseModel):
    """Shared mutable state for proposal-based generators."""

    step_count: int = 0
    best_candidate_id: Optional[str] = None
    best_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def best_energy(self) -> Optional[float]:
        return self.best_score

    @best_energy.setter
    def best_energy(self, value: Optional[float]) -> None:
        self.best_score = value

    @property
    def internal_state(self) -> Dict[str, Any]:
        return self.metadata


class GeneratorStrategy(SearchStrategy):
    """
    Unified generator contract.

    Concrete strategies may still implement custom batch loops, but they
    should expose the same propose/observe surface so the repo can gradually
    converge on a single Generator layer.
    """

    def __init__(
        self,
        env: Any,
        controller: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        name: str = "BaseGenerator",
    ):
        super().__init__(env=env, controller=controller, logger=logger, name=name)

    def initialize(self) -> GeneratorState:
        return GeneratorState()

    @abstractmethod
    def propose(
        self,
        state: GeneratorState,
        budget: int = 1,
    ) -> List[CandidateSpec]:
        """Propose the next batch of structure candidates."""

    @abstractmethod
    def observe(
        self,
        state: GeneratorState,
        results: List[EvaluationResult],
    ) -> GeneratorState:
        """Update internal state from evaluator results."""

    def should_stop(self, state: GeneratorState) -> bool:
        return False
