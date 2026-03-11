from __future__ import annotations

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator

class AnsatzSpec(BaseModel):
    """Ansatz 结构规范"""
    name: str = Field(..., description="Ansatz 唯一标识")
    family: str = Field("hea", description="类别 (hea, uccsd, hva, ga, etc.)")
    n_qubits: int = Field(..., gt=0, description="量子比特数")
    config: Dict[str, Any] = Field(..., description="结构化具体参数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元信息")

    @field_validator("config")
    @classmethod
    def validate_config(cls, v: Dict[str, Any], info: Any) -> Dict[str, Any]:
        # Access family from the model values isn't directly available in same way in field_validator
        # but we can do basic structure check here. Full validation can be done in model_validator if needed.
        return v

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
    two_qubit_gate_options: List[str] = Field(
        default_factory=lambda: ["cnot", "cz", "rzz", "rxx_ryy_rzz"]
    )
    init_state_options: List[str] = Field(default_factory=lambda: ["zero", "hadamard", "hf"])
    param_strategy_options: List[str] = Field(default_factory=lambda: ["independent", "tied"])
