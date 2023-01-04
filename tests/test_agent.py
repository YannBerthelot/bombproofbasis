import gym
import numpy as np
import pytest

from bombproofbasis.agents.agent import Agent
from bombproofbasis.agents.utils import get_action_shape
from bombproofbasis.types import AgentConfig, NetworkConfig, ScalerConfig


def test_agent():
    env = gym.make("CartPole-v1")
    policy_network_config = NetworkConfig(
        learning_rate=1e-3,
        architecture=["32", "relu", "64", "relu", "LSTM(128*3)"],
        input_shape=env.observation_space.shape[0],
        output_shape=get_action_shape(env),
        actor=True,
    )
    value_network_config = NetworkConfig(
        learning_rate=1e-3,
        architecture=["32", "relu", "64", "relu", "LSTM(128*3)"],
        input_shape=env.observation_space.shape[0],
        output_shape=get_action_shape(env),
        actor=False,
    )
    scaler_config = ScalerConfig(scale=True)
    agent_config = AgentConfig(
        learning_rate=1e-3,
        environment=env,
        agent_type="A2C",
        continous=False,
        policy_network=policy_network_config,
        value_network=value_network_config,
        scaler=scaler_config,
    )
    agent = Agent(agent_config)

    assert agent.env == env

    agent.env = gym.make("CartPole-v0")
    with pytest.raises(ValueError):
        agent.env = gym.make("MountainCar-v0")

    assert agent.obs_shape == env.observation_space.shape[0]

    with pytest.raises(NotImplementedError):
        agent.select_action(np.ones(agent.obs_shape))

    assert agent.action_shape == get_action_shape(env)

    with pytest.raises(NotImplementedError):
        agent.train(n_iter=10, env=env)
    with pytest.raises(NotImplementedError):
        agent.test(env=env, n_episodes=10, render=False)
    with pytest.raises(NotImplementedError):
        agent.save("./model")
    with pytest.raises(NotImplementedError):
        agent.load("./model")
