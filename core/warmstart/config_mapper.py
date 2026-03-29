from typing import Any, Dict

import torch
from torch import Tensor

from core.representation.compiler import count_params


class ParameterMapper:
    """
    参数映射基类：定义如何将旧 config 的参数映射到新 config。
    """

    def map(self, old_config: Dict[str, Any], old_params: Tensor, new_config: Dict[str, Any], n_qubits: int) -> Tensor:
        raise NotImplementedError


class IdentityMapper(ParameterMapper):
    """
    恒等映射器：
    1. 最大化保留：如果局部结构未变，保留原值。
    2. 局部性：结构扩展（如加层）时，原有参数位置保持不变。
    """

    def map(self, old_config: Dict[str, Any], old_params: Tensor, new_config: Dict[str, Any], n_qubits: int) -> Tensor:
        new_num_params = count_params(new_config, n_qubits)
        new_params = torch.zeros(new_num_params)

        old_layers = old_config.get("layers", 1)
        new_layers = new_config.get("layers", 1)
        keys_to_check = ["single_qubit_gates", "two_qubit_gate", "entanglement", "param_strategy"]
        configs_consistent = all(old_config.get(k) == new_config.get(k) for k in keys_to_check)

        if configs_consistent and new_layers >= old_layers:
            old_num_params = len(old_params)
            new_params[:old_num_params] = old_params
            return new_params

        limit = min(len(old_params), new_num_params)
        new_params[:limit] = old_params[:limit]
        return new_params


def get_mapper(name: str = "identity") -> ParameterMapper:
    if name == "identity":
        return IdentityMapper()
    raise ValueError(f"Unknown mapper: {name}")
