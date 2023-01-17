import numpy as np
import pytest

from bombproofbasis.agents.A2C import A2C
from bombproofbasis.agents.utils import get_action_shape, get_obs_shape
from bombproofbasis.probe_environments import (
    ProbeEnv1,
    ProbeEnv2,
    ProbeEnv3,
    ProbeEnv4,
    ProbeEnv5,
)
from bombproofbasis.types import (
    A2CConfig,
    BufferConfig,
    LoggingConfig,
    NetworkConfig,
    ScalerConfig,
)

gamma = 0.5
eps = 1e-1


def get_config(env):
    action_shape = get_action_shape(env)
    obs_shape = get_obs_shape(env)
    critic_architecture = ["64", "relu", "32", "relu"]
    actor_architecture = ["64", "tanh", "32", "tanh"]
    # recurrent_architecture = ["4", "relu", "16", "relu", "LSTM(8*1)"]
    policy_network_config = NetworkConfig(
        learning_rate=1e-2,
        architecture=actor_architecture,
        input_shape=obs_shape,
        output_shape=action_shape,
        actor=True,
    )
    value_network_config = NetworkConfig(
        learning_rate=1e-2,
        architecture=critic_architecture,
        input_shape=obs_shape,
        output_shape=1,
        actor=False,
    )
    n_steps = 1
    buffer_size = 1
    buffer_config = BufferConfig(
        obs_shape=obs_shape,
        buffer_size=buffer_size,
        setting="n-step",
        n_steps=n_steps,
        gamma=gamma,
    )
    buffer_MC_config = BufferConfig(obs_shape=obs_shape, setting="MC", gamma=gamma)
    scaler_config = ScalerConfig(scale=True)
    A2C_MC_CONFIG = A2CConfig(
        environment=env,
        agent_type="A2C",
        policy_network=policy_network_config,
        value_network=value_network_config,
        scaler=scaler_config,
        buffer=buffer_MC_config,
        entropy_coeff=0.0,
    )
    A2C_TD_CONFIG = A2CConfig(
        environment=env,
        agent_type="A2C",
        policy_network=policy_network_config,
        value_network=value_network_config,
        scaler=scaler_config,
        buffer=buffer_config,
        entropy_coeff=0.0,
    )
    return A2C_MC_CONFIG, A2C_TD_CONFIG


def test_1():
    env = ProbeEnv1()
    td_config, MC_config = get_config(env)
    agent = A2C(MC_config, LoggingConfig(logging_output=None))
    agent.train(env, n_iter=500)
    assert agent.get_value(env.reset()[0]) == pytest.approx(1.0, rel=eps)

    agent = A2C(td_config, LoggingConfig(logging_output=None))
    agent.train(env, n_iter=500)
    assert agent.get_value(env.reset()[0]) == pytest.approx(1.0, rel=eps)


def test_2():
    env = ProbeEnv2()
    td_config, MC_config = get_config(env)
    agent = A2C(MC_config, LoggingConfig(logging_output=None))
    agent.train(env, n_iter=500)
    assert agent.get_value(np.array([-1])) == pytest.approx(-1, rel=eps)
    assert agent.get_value(np.array([1])) == pytest.approx(1, rel=eps)

    agent = A2C(td_config, LoggingConfig(logging_output=None))
    agent.train(env, n_iter=500)
    assert agent.get_value(np.array([-1])) == pytest.approx(-1, rel=eps)
    assert agent.get_value(np.array([1])) == pytest.approx(1, rel=eps)


def test_3():
    env = ProbeEnv3()
    td_config, MC_config = get_config(env)
    agent = A2C(MC_config, LoggingConfig(logging_output=None))
    agent.train(env, n_iter=500)
    assert agent.get_value(np.array([0])) == pytest.approx(gamma, rel=eps)
    assert agent.get_value(np.array([1])) == pytest.approx(1, rel=eps)

    agent = A2C(td_config, LoggingConfig(logging_output=None))
    agent.train(env, n_iter=500)
    assert agent.get_value(np.array([0])) == pytest.approx(gamma, rel=eps)
    assert agent.get_value(np.array([1])) == pytest.approx(1, rel=eps)


def test_4():
    env = ProbeEnv4()
    td_config, MC_config = get_config(env)
    agent = A2C(MC_config, LoggingConfig(logging_output=None))
    agent.train(env, n_iter=500)
    for _ in range(10):
        obs, _ = env.reset()
        assert agent.select_action(obs) == 0

    agent = A2C(td_config, LoggingConfig(logging_output=None))
    agent.train(env, n_iter=500)
    for _ in range(10):
        obs, _ = env.reset()
        assert agent.select_action(obs) == 0


def test_5():
    env = ProbeEnv5()
    td_config, MC_config = get_config(env)
    agent = A2C(MC_config, LoggingConfig(logging_output=None))
    agent.train(env, n_iter=500)
    for _ in range(10):
        assert agent.select_action(-1) == 0
        assert agent.select_action(1) == 1
    assert agent.get_value(-1) == pytest.approx(1, rel=eps)
    assert agent.get_value(1) == pytest.approx(1, rel=eps)

    agent = A2C(td_config, LoggingConfig(logging_output=None))
    agent.train(env, n_iter=500)
    for _ in range(10):
        assert agent.select_action(-1) == 0
        assert agent.select_action(1) == 1
    assert agent.get_value(-1) == pytest.approx(1, rel=eps)
    assert agent.get_value(1) == pytest.approx(1, rel=eps)
