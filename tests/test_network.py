import gym
import numpy as np
import pytest

from bombproofbasis.agents.utils import get_action_shape
from bombproofbasis.network.network import BaseTorchNetwork
from bombproofbasis.types import NetworkConfig


def test_BaseTorchAgent():
    env = gym.make("CartPole-v1")
    input_shape = env.observation_space.shape[0]
    output_shape = get_action_shape(env)
    network_config = NetworkConfig(
        learning_rate=1e-3,
        architecture=["32", "relu", "64", "relu", "LSTM(128*3)"],
        input_shape=input_shape,
        output_shape=output_shape,
    )
    network = BaseTorchNetwork(config=network_config)

    with pytest.raises(NotImplementedError):
        network.select_action(np.ones(3))
    with pytest.raises(NotImplementedError):
        network.update_policy()
    with pytest.raises(NotImplementedError):
        network.save(path="some/path")
    with pytest.raises(NotImplementedError):
        network.load(path="some/path")
    assert isinstance(network.architecture, list)
