import gym
import pytest
import torch

from bombproofbasis.agents.utils import get_action_shape
from bombproofbasis.network.network import BaseTorchNetwork
from bombproofbasis.types import NetworkConfig


def test_BaseTorchNetwork():
    env = gym.make("CartPole-v1")
    input_shape = env.observation_space.shape[0]
    output_shape = get_action_shape(env)
    classic_architecture = ["32", "relu", "64", "relu", "32"]
    num_layers = 1
    hidden_size = 128
    recurrent_architecture = [
        "32",
        "relu",
        "64",
        "relu",
        "32",
        f"LSTM({hidden_size}*{num_layers})",
    ]
    seq_len = 1
    with pytest.raises(ValueError):
        # critic with output shape different from 1
        network_config = NetworkConfig(
            learning_rate=1e-3,
            architecture=classic_architecture,
            input_shape=input_shape,
            output_shape=2,
            actor=False,
        )
        network = BaseTorchNetwork(config=network_config)
    ## Test forward passes
    for i, architecture in enumerate((classic_architecture, recurrent_architecture)):
        network_config = NetworkConfig(
            learning_rate=1e-3,
            architecture=architecture,
            input_shape=input_shape,
            output_shape=output_shape,
        )
        network = BaseTorchNetwork(config=network_config)
        if i == 0:
            output = network(torch.ones(input_shape))
            assert output.view(-1).shape[0] == output_shape
        else:
            output = network(
                torch.ones(input_shape).view(seq_len, input_shape),
            )
            assert output.view(-1).shape[0] == output_shape
    ## More LSTM cells with more layers
    num_layers = 3
    hidden_size = 128
    recurrent_architecture = [
        "32",
        "relu",
        f"LSTM({hidden_size}*{num_layers})",
        "relu",
        "32",
        f"LSTM({hidden_size}*{num_layers})",
    ]
    network_config = NetworkConfig(
        learning_rate=1e-3,
        architecture=recurrent_architecture,
        input_shape=input_shape,
        output_shape=output_shape,
    )
    network = BaseTorchNetwork(config=network_config)
    output = network(
        torch.ones(input_shape).view(seq_len, input_shape),
    )
    assert output.view(-1).shape[0] == output_shape
