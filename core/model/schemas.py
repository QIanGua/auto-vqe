from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BlockSpec(BaseModel):
    """模块规范"""

    name: str
    family: str
    qubit_subset: Optional[List[int]] = None
    repetitions: int = 1
    params_per_repeat: int
    parameter_sharing: str = "none"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OperatorSpec(BaseModel):
    """单算符规范"""

    name: str
    family: str
    support_qubits: List[int]
    generator: Optional[str] = None
    cost_weight: float = 1.0
    symmetry_tags: List[str] = Field(default_factory=list)
    hardware_legal: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OperatorPoolSpec(BaseModel):
    """算符池规范"""

    name: str
    operators: List[OperatorSpec]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StructureEdit(BaseModel):
    """结构编辑描述"""

    edit_type: Literal[
        "append_block",
        "insert_block",
        "remove_block",
        "replace_block",
        "append_operator",
        "merge_blocks",
        "expand_qubit_subset",
    ]
    target_path: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None


class WarmStartPlan(BaseModel):
    """参数重用/Warm-start 计划"""

    old_param_count: int
    new_param_count: int
    reused_indices: List[tuple[int, int]] = Field(default_factory=list)
    initialized_indices: List[int] = Field(default_factory=list)
    dropped_indices: List[int] = Field(default_factory=list)
    init_strategy: Literal["zeros", "small_random", "copy_neighbor", "block_default"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationSpec(BaseModel):
    """评估配置"""

    fidelity: Literal["quick", "medium", "full"]
    max_steps: int
    max_wall_clock_sec: Optional[float] = None
    n_seeds: int = 1
    hamiltonian_term_budget: Optional[int] = None
    use_full_observable: bool = True
    enable_early_stop: bool = True
    optimizer_name: str = "adam"


class EvaluationResult(BaseModel):
    """评估结果"""

    candidate_id: str
    fidelity: str
    success: bool
    val_energy: Optional[float] = None
    energy_error: Optional[float] = None
    proxy_score: Optional[float] = None
    num_params: int
    two_qubit_gates: int
    runtime_sec: float
    actual_steps: int
    trainability_score: Optional[float] = None
    measurement_cost_est: Optional[float] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)
    error_message: Optional[str] = None


class AnsatzSpec(BaseModel):
    """Ansatz 结构规范，支持层级化 Block 和 Operator 构造"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Ansatz 唯一标识")
    family: str = Field("hea", description="类别 (hea, uccsd, hva, ga, blocks, etc.)")
    n_qubits: int = Field(..., gt=0, description="量子比特数")
    env_name: Optional[str] = Field(None, description="环境名称")
    create_circuit: Optional[Callable[..., Any]] = Field(None, description="可选线路构造器")
    num_params: int = Field(0, ge=0, description="显式参数量")
    config: Dict[str, Any] = Field(default_factory=dict, description="结构化具体参数 (GA/Grid 兼容)")
    blocks: List[Union[BlockSpec, OperatorSpec]] = Field(default_factory=list, description="有序结构列表")
    growth_history: List[StructureEdit] = Field(default_factory=list, description="生长历史")
    operator_pool_ref: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元信息")

    @field_validator("config")
    @classmethod
    def validate_config(cls, v: Dict[str, Any], info: Any) -> Dict[str, Any]:
        return v

    def to_logging_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude={"create_circuit"}, exclude_none=True)


class CandidateSpec(BaseModel):
    """搜索候选规范"""

    candidate_id: str
    parent_candidate_id: Optional[str] = None
    ansatz: AnsatzSpec
    proposed_by: str
    structure_edit: Optional[StructureEdit] = None
    warm_start_from: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StrategyCheckpoint(BaseModel):
    """策略状态存档"""

    strategy_name: str
    step_count: int
    best_candidate_id: Optional[str]
    best_energy: Optional[float]
    internal_state: Dict[str, Any] = Field(default_factory=dict)


class HardwareConstraintSpec(BaseModel):
    """硬件约束规范"""

    topology_type: Literal["line", "grid", "custom"]
    coupling_edges: List[tuple[int, int]]
    native_gate_set: List[str]
    max_depth: Optional[int] = None
    max_two_qubit_gates: Optional[int] = None
    routing_policy: str = "strict"


class SchedulerSpec(BaseModel):
    """优化器调度器规范"""

    type: str = Field("ReduceLROnPlateau")
    patience: int = Field(50)
    factor: float = Field(0.5)
    min_lr: float = Field(1e-5)
    mode: str = Field("min")


class OptimizerSpec(BaseModel):
    """优化器配置规范"""

    method: str = Field("Adam")
    lr: float = Field(0.01, gt=0)
    max_steps: int = Field(500, gt=0)
    tol: float = Field(1e-8)
    early_stop_window: int = Field(50, gt=0)
    early_stop_threshold: float = Field(1e-8, gt=0)
    grad_clip_norm: Optional[float] = Field(1.0)
    scheduler: Optional[SchedulerSpec] = Field(default_factory=SchedulerSpec)


class SearchSpaceSpec(BaseModel):
    """搜索空间配置规范"""

    layers: List[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    entanglement: List[str] = Field(default_factory=lambda: ["linear", "ring", "brick", "full"])
    single_qubit_gates_options: List[List[str]] = Field(
        default_factory=lambda: [["ry"], ["rx", "ry"], ["ry", "rz"], ["rx", "ry", "rz"]]
    )
    two_qubit_gate_options: List[str] = Field(default_factory=lambda: ["cnot", "cz", "rzz", "rxx_ryy_rzz"])
    init_state_options: List[str] = Field(default_factory=lambda: ["zero", "hadamard", "hf"])
    param_strategy_options: List[str] = Field(default_factory=lambda: ["independent", "tied"])
