import torch
from torch import Tensor
from typing import Dict, Any, List, Optional
from core.circuit_factory import count_params, _count_two_qubit_params, _linear_pairs, _ring_pairs, _brick_pairs, _full_pairs

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
        
        # 简单逻辑：如果 layers 增加，且其他不变，直接填充。
        # 这里实现一个更通用的按层/按门填充逻辑需要对 circuit_factory 的 build 逻辑有更深入的理解或解耦。
        # 暂时实现一个针对层数增加的 Identity 映射。
        
        old_layers = old_config.get("layers", 1)
        new_layers = new_config.get("layers", 1)
        
        # 检查除 layers 外的配置是否一致
        keys_to_check = ["single_qubit_gates", "two_qubit_gate", "entanglement", "param_strategy"]
        configs_consistent = all(old_config.get(k) == new_config.get(k) for k in keys_to_check)
        
        if configs_consistent and new_layers >= old_layers:
            # 这种情况最为简单，旧参数直接复制到前半部分
            old_num_params = len(old_params)
            new_params[:old_num_params] = old_params
            # 剩余部分保持为 0 (或者小随机值，但按照协议建议新参数初始为 0 退化为等效结构)
            return new_params
        
        # 对于更复杂的变化（门更换、拓扑变化），暂时 fallback 到零填充或部分匹配。
        # todo: 实现更精细的门级映射逻辑
        limit = min(len(old_params), new_num_params)
        new_params[:limit] = old_params[:limit]
        return new_params

def get_mapper(name: str = "identity") -> ParameterMapper:
    if name == "identity":
        return IdentityMapper()
    raise ValueError(f"Unknown mapper: {name}")
