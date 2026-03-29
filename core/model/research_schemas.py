from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from core.model.schemas import EvaluationResult


class HypothesisSpec(BaseModel):
    """Structured research hypothesis for one iteration or branch."""

    hypothesis_id: str
    parent_hypothesis_id: Optional[str] = None
    system: str
    objective: str
    statement: str
    motivation: Optional[str] = None
    target_metric: str = "energy_error"
    expected_effect: Optional[str] = None
    search_region: Dict[str, Any] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    priority: int = 1
    created_by: str = "policy_engine"
    status: Literal["open", "tested", "accepted", "rejected"] = "open"


class ActionSpec(BaseModel):
    """Action chosen by the research policy for one step."""

    action_id: str
    hypothesis_id: str
    system_dir: str
    action_type: Literal[
        "run_strategy",
        "verify_config",
        "promote_candidate",
        "reduce_search_space",
        "switch_strategy",
        "stop_research",
    ]
    strategy_name: Optional[str] = None
    fidelity: Optional[Literal["quick", "medium", "full"]] = None
    budget: Dict[str, Any] = Field(default_factory=dict)
    config_path: Optional[str] = None
    candidate_ids: List[str] = Field(default_factory=list)
    search_space_patch: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None


class DecisionRecord(BaseModel):
    """Research-layer interpretation of one executed action."""

    decision_id: str
    iteration: int
    hypothesis_id: str
    action_id: str
    decision: Literal["keep", "discard", "refine", "promote", "stop"]
    summary: str
    evidence_for: List[str] = Field(default_factory=list)
    evidence_against: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    selected_config_path: Optional[str] = None
    selected_candidate_id: Optional[str] = None
    followup_actions: List[str] = Field(default_factory=list)


class ResearchMemory(BaseModel):
    """Aggregate research state persisted between iterations."""

    system: str
    objective: str
    best_energy_error: Optional[float] = None
    best_candidate_id: Optional[str] = None
    best_config_path: Optional[str] = None
    active_hypotheses: List[HypothesisSpec] = Field(default_factory=list)
    accepted_hypotheses: List[str] = Field(default_factory=list)
    rejected_hypotheses: List[str] = Field(default_factory=list)
    dead_ends: List[str] = Field(default_factory=list)
    transferable_insights: List[str] = Field(default_factory=list)
    strategy_stats: Dict[str, Any] = Field(default_factory=dict)
    last_decision: Optional[DecisionRecord] = None
    next_recommendations: List[str] = Field(default_factory=list)


class RunBundle(BaseModel):
    """Unified result bundle for a research action execution."""

    action: ActionSpec
    metrics: Dict[str, Any] = Field(default_factory=dict)
    candidate_results: List[EvaluationResult] = Field(default_factory=list)
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
    selected_config_path: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
