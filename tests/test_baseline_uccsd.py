import numpy as np

from baselines.uccsd import build_ansatz
from core.base_env import QuantumEnvironment


def test_minimal_uccsd_default_excitation_structure():
    env = QuantumEnvironment("LiH_test", 4, -1.0)

    spec = build_ansatz(env)

    assert spec.family == "uccsd"
    assert spec.name == "uccsd"
    assert spec.config["occupied_orbitals"] == [0, 1]
    assert spec.config["virtual_orbitals"] == [2, 3]
    assert spec.metadata["singles_count"] == 4
    assert spec.metadata["doubles_count"] == 1
    assert spec.metadata["excitation_count"] == 5
    assert spec.num_params == 5


def test_minimal_uccsd_circuit_builds_and_consumes_all_params():
    env = QuantumEnvironment("LiH_test", 4, -1.0)
    spec = build_ansatz(env, {"layers": 2})

    params = np.zeros(spec.num_params, dtype=np.float32)
    circuit, used = spec.create_circuit(params)

    assert used == spec.num_params
    assert circuit._nqubits == 4
    assert len(circuit.to_qir()) > 0


def test_minimal_uccsd_supports_singles_only_override():
    env = QuantumEnvironment("LiH_test", 4, -1.0)
    spec = build_ansatz(
        env,
        {
            "layers": 2,
            "include_doubles": False,
        },
    )

    assert spec.metadata["singles_count"] == 4
    assert spec.metadata["doubles_count"] == 0
    assert spec.num_params == 8
