import math

from baselines.adapt import build_ansatz as build_adapt
from baselines.adapt import build_operator_pool as build_adapt_pool
from baselines.hva import build_ansatz as build_hva
from baselines.qubit_adapt import build_ansatz as build_qubit_adapt
from baselines.qubit_adapt import build_operator_pool as build_qubit_adapt_pool
from experiments.tfim.env import TFIMEnvironment, get_tfim_reference_energy


def test_tfim_reference_energy_matches_known_4q_open_chain_value():
    env = TFIMEnvironment(n_qubits=4)
    assert abs(env.exact_energy - (-4.758770483143634)) < 1e-9
    assert abs(get_tfim_reference_energy(4) - env.exact_energy) < 1e-12


def test_tfim_large_scale_reference_is_marked_unavailable():
    env = TFIMEnvironment(n_qubits=100)
    assert math.isnan(env.exact_energy)
    assert env.reference_energy_kind == "unavailable"


def test_hva_defaults_follow_environment_boundary():
    env = TFIMEnvironment(n_qubits=4)
    spec = build_hva(env)
    assert spec.config["use_ring"] is False
    assert spec.metadata["boundary"] == "open"


def test_adapt_initializers_expose_real_search_entrypoints():
    env = TFIMEnvironment(n_qubits=4)

    adapt = build_adapt(env)
    assert adapt.metadata["research_status"] == "search_initialized"
    assert adapt.num_params == 0
    assert build_adapt_pool(env).operators

    qubit_adapt = build_qubit_adapt(env)
    assert qubit_adapt.metadata["research_status"] == "search_initialized"
    assert qubit_adapt.num_params == 0
    assert build_qubit_adapt_pool(env).operators
