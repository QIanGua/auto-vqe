"""
通用线路工厂 (Circuit Factory)

将结构化 config dict 编译为 (create_circuit_fn, num_params)，
使 Agent 只需生成结构化配置而非 Python 代码。

Config Schema
-------------
layers            : int          = 1            参数化层数
single_qubit_gates: list[str]    = ["ry"]       单比特旋转门 (rx/ry/rz)
two_qubit_gate    : str          = "cnot"       纠缠门 (cnot/cz/rzz/rxx_ryy_rzz)
entanglement      : str          = "linear"     拓扑 (linear/ring/brick/full)
init_state        : str          = "zero"       初始态 (zero/hadamard/hf)
hf_qubits         : list[int]    = []           HF 翻转比特
param_strategy    : str          = "independent" 参数策略 (independent/tied)
"""

from __future__ import annotations

import itertools
from typing import Callable, Any

import tensorcircuit as tc


# ---------------------------------------------------------------------------
# 纠缠拓扑生成器
# ---------------------------------------------------------------------------

def _linear_pairs(n: int) -> list[tuple[int, int]]:
    """线性拓扑: (0,1), (1,2), ..., (n-2, n-1)"""
    return [(i, i + 1) for i in range(n - 1)]


def _ring_pairs(n: int) -> list[tuple[int, int]]:
    """环形拓扑: 线性 + (n-1, 0)"""
    return _linear_pairs(n) + [(n - 1, 0)]


def _brick_pairs(n: int, layer_idx: int) -> list[tuple[int, int]]:
    """砖墙式拓扑: 偶数层 (0,1),(2,3),..., 奇数层 (1,2),(3,4),..."""
    offset = layer_idx % 2
    return [(i, i + 1) for i in range(offset, n - 1, 2)]


def _full_pairs(n: int) -> list[tuple[int, int]]:
    """全连接拓扑: 所有 (i,j) 对"""
    return list(itertools.combinations(range(n), 2))


def get_pairs(entanglement: str, n_qubits: int, layer_idx: int = 0) -> list[tuple[int, int]]:
    """根据拓扑名获取比特对列表。"""
    if entanglement == "linear":
        return _linear_pairs(n_qubits)
    elif entanglement == "ring":
        return _ring_pairs(n_qubits)
    elif entanglement == "brick":
        return _brick_pairs(n_qubits, layer_idx)
    elif entanglement == "full":
        return _full_pairs(n_qubits)
    else:
        raise ValueError(f"Unknown entanglement topology: {entanglement}")


# ---------------------------------------------------------------------------
# 参数计数
# ---------------------------------------------------------------------------

def _count_two_qubit_params(gate: str) -> int:
    """返回一个纠缠门贡献的参数数量。"""
    if gate in ("cnot", "cz"):
        return 0
    elif gate == "rzz":
        return 1
    elif gate == "rxx_ryy_rzz":
        return 3
    else:
        raise ValueError(f"Unknown two-qubit gate: {gate}")


def count_params(config: dict, n_qubits: int) -> int:
    """根据 config 和 qubit 数量计算总参数量。"""
    layers = config.get("layers", 1)
    sq_gates = config.get("single_qubit_gates", ["ry"])
    tq_gate = config.get("two_qubit_gate", "cnot")
    entanglement = config.get("entanglement", "linear")
    param_strategy = config.get("param_strategy", "independent")

    # 单比特参数: 每层 len(sq_gates) * n_qubits
    sq_per_layer = len(sq_gates) * n_qubits

    # 双比特参数: 取决于拓扑产生的 pair 数 × 每门参数
    tq_param_per_pair = _count_two_qubit_params(tq_gate)

    if entanglement == "brick":
        # brick 拓扑每层 pair 数不同 (偶/奇), 需要分别计数
        total_tq = 0
        for l in range(layers):
            pairs = _brick_pairs(n_qubits, l)
            total_tq += len(pairs) * tq_param_per_pair
    else:
        pairs = get_pairs(entanglement, n_qubits)
        total_tq = layers * len(pairs) * tq_param_per_pair

    total_sq = layers * sq_per_layer

    if param_strategy == "tied":
        # 参数共享: 只有 1 层独立参数
        if entanglement == "brick":
            # tied + brick: 取第一层的参数量 (保守处理)
            pairs_0 = _brick_pairs(n_qubits, 0)
            one_layer = sq_per_layer + len(pairs_0) * tq_param_per_pair
        else:
            pairs = get_pairs(entanglement, n_qubits)
            one_layer = sq_per_layer + len(pairs) * tq_param_per_pair
        return one_layer
    else:
        return total_sq + total_tq


# ---------------------------------------------------------------------------
# 核心: build_ansatz
# ---------------------------------------------------------------------------

def build_ansatz(
    config: dict,
    n_qubits: int,
) -> tuple[Callable, int]:
    """
    将结构化 config dict 编译为 (create_circuit_fn, num_params)。

    Parameters
    ----------
    config : dict
        Ansatz 配置，字段见模块文档。
    n_qubits : int
        量子比特数量。

    Returns
    -------
    create_circuit_fn : Callable[[Tensor], Tuple[Circuit, int]]
        接受 params tensor，返回 (circuit, idx)。
    num_params : int
        此 ansatz 所需参数总量。
    """
    layers = config.get("layers", 1)
    sq_gates = config.get("single_qubit_gates", ["ry"])
    tq_gate = config.get("two_qubit_gate", "cnot")
    entanglement = config.get("entanglement", "linear")
    init_state = config.get("init_state", "zero")
    hf_qubits = config.get("hf_qubits", [])
    param_strategy = config.get("param_strategy", "independent")

    num_params = count_params(config, n_qubits)

    # 验证门名称合法性
    valid_sq = {"rx", "ry", "rz"}
    for g in sq_gates:
        if g not in valid_sq:
            raise ValueError(f"Unknown single-qubit gate: {g}. Valid: {valid_sq}")

    valid_tq = {"cnot", "cz", "rzz", "rxx_ryy_rzz"}
    if tq_gate not in valid_tq:
        raise ValueError(f"Unknown two-qubit gate: {tq_gate}. Valid: {valid_tq}")

    def create_circuit(params):
        c = tc.Circuit(n_qubits)

        # ---- 初始态 ----
        if init_state == "hadamard":
            for i in range(n_qubits):
                c.h(i)
        elif init_state == "hf":
            for q in hf_qubits:
                c.x(q)

        idx = 0

        for layer in range(layers):
            # 若 tied 策略，每层都从 idx=0 开始（复用同一组参数）
            if param_strategy == "tied":
                idx = 0

            # ---- 单比特旋转门 ----
            for i in range(n_qubits):
                for g in sq_gates:
                    gate_fn = getattr(c, g)
                    gate_fn(i, theta=params[idx])
                    idx += 1

            # ---- 纠缠门 ----
            pairs = get_pairs(entanglement, n_qubits, layer_idx=layer)
            for i, j in pairs:
                if tq_gate == "cnot":
                    c.cnot(i, j)
                elif tq_gate == "cz":
                    c.cz(i, j)
                elif tq_gate == "rzz":
                    c.rzz(i, j, theta=params[idx])
                    idx += 1
                elif tq_gate == "rxx_ryy_rzz":
                    c.rxx(i, j, theta=params[idx])
                    idx += 1
                    c.ryy(i, j, theta=params[idx])
                    idx += 1
                    c.rzz(i, j, theta=params[idx])
                    idx += 1

        # 实际使用的参数量
        actual_idx = idx if param_strategy != "tied" else num_params
        return c, actual_idx

    return create_circuit, num_params


# ---------------------------------------------------------------------------
# 辅助: config 文本描述
# ---------------------------------------------------------------------------

def config_to_str(config: dict) -> str:
    """将 config dict 压缩为一行可读字符串，用于日志和 results.tsv。"""
    parts = []
    for k in ["layers", "single_qubit_gates", "two_qubit_gate",
              "entanglement", "init_state", "hf_qubits", "param_strategy"]:
        if k in config:
            v = config[k]
            if isinstance(v, list):
                v = "+".join(str(x) for x in v)
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# 搜索空间定义与随机/变异逻辑
# ---------------------------------------------------------------------------

SEARCH_DIMENSIONS = {
    "layers": [1, 2, 3, 4, 5],
    "single_qubit_gates": [["ry"], ["rx", "ry"], ["ry", "rz"], ["rx", "ry", "rz"]],
    "two_qubit_gate": ["cnot", "cz", "rzz", "rxx_ryy_rzz"],
    "entanglement": ["linear", "ring", "brick", "full"],
    "init_state": ["zero", "hadamard", "hf"],
    "param_strategy": ["independent", "tied"],
}

def get_random_config(custom_dimensions: dict | None = None) -> dict:
    """随机生成一个配置。"""
    import random
    dims = custom_dimensions or SEARCH_DIMENSIONS
    config = {}
    for k, v in dims.items():
        config[k] = random.choice(v)
    return config

def mutate_config(config: dict, custom_dimensions: dict | None = None, mutation_rate: float = 0.3) -> dict:
    """
    对现有配置进行变异。
    
    Parameters
    ----------
    config : dict
        原始配置。
    custom_dimensions : dict, optional
        可变维度的取值范围。
    mutation_rate : float
        每个字段发生变异的概率。
    """
    import random
    dims = custom_dimensions or SEARCH_DIMENSIONS
    new_config = config.copy()
    
    for k, v in dims.items():
        if random.random() < mutation_rate:
            # 确保变异后的值与原值不同（如果可能）
            if len(v) > 1:
                options = [x for x in v if x != config.get(k)]
                new_config[k] = random.choice(options)
            else:
                new_config[k] = v[0]
                
    return new_config

def crossover_configs(config1: dict, config2: dict) -> dict:
    """对两个配置进行交叉（Crossover）。"""
    import random
    new_config = {}
    for k in set(config1.keys()) | set(config2.keys()):
        new_config[k] = random.choice([config1.get(k), config2.get(k)])
        if new_config[k] is None: # 处理一个有一方没有的 key
             new_config[k] = config1.get(k) or config2.get(k)
    return new_config

def generate_config_grid(
    dimensions: dict[str, list],
) -> list[dict]:
    """
    根据维度定义生成所有配置的笛卡尔积。
    """
    keys = list(dimensions.keys())
    values = list(dimensions.values())
    configs = []
    import itertools
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs
