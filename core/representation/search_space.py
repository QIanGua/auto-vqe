import itertools
import random


SEARCH_DIMENSIONS = {
    "layers": [1, 2, 3, 4, 5],
    "single_qubit_gates": [["ry"], ["rx", "ry"], ["ry", "rz"], ["rx", "ry", "rz"]],
    "two_qubit_gate": ["cnot", "cz", "rzz", "rxx_ryy_rzz"],
    "entanglement": ["linear", "ring", "brick", "full"],
    "init_state": ["zero", "hadamard", "hf"],
    "param_strategy": ["independent", "tied", "translational"],
}


def config_to_str(config: dict) -> str:
    parts = []
    for k in [
        "layers",
        "single_qubit_gates",
        "two_qubit_gate",
        "entanglement",
        "init_state",
        "hf_qubits",
        "param_strategy",
    ]:
        if k in config:
            v = config[k]
            if isinstance(v, list):
                v = "+".join(str(x) for x in v)
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def get_random_config(custom_dimensions: dict | None = None) -> dict:
    dims = custom_dimensions or SEARCH_DIMENSIONS
    return {k: random.choice(v) for k, v in dims.items()}


def mutate_config(config: dict, custom_dimensions: dict | None = None, mutation_rate: float = 0.3) -> dict:
    dims = custom_dimensions or SEARCH_DIMENSIONS
    new_config = config.copy()
    for k, v in dims.items():
        if random.random() < mutation_rate:
            if len(v) > 1:
                options = [x for x in v if x != config.get(k)]
                new_config[k] = random.choice(options)
            else:
                new_config[k] = v[0]
    return new_config


def crossover_configs(config1: dict, config2: dict) -> dict:
    new_config = {}
    for k in set(config1.keys()) | set(config2.keys()):
        new_config[k] = random.choice([config1.get(k), config2.get(k)])
        if new_config[k] is None:
            new_config[k] = config1.get(k) or config2.get(k)
    return new_config


def generate_config_grid(dimensions: dict[str, list]) -> list[dict]:
    keys = list(dimensions.keys())
    values = list(dimensions.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
