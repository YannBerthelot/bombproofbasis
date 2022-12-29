import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Beta

from bombproofbasis.network.utils import (
    LinearSchedule,
    compute_KL_divergence,
    get_device,
    get_network_from_architecture,
    t,
)


def test_t():
    old_tensor = np.ones(
        4,
    )
    new_tensor = t(old_tensor)
    assert isinstance(new_tensor, torch.Tensor)
    assert old_tensor.shape == new_tensor.shape


def test_get_network_from_architecture():
    input_shape = 16
    output_shape = 4
    network_architecture_1 = ["32", "64"]
    network_architecture_2 = ["32", "relu", "64", "tanh"]
    network_architecture_3 = ["48", "silu", "LSTM(128)", "LSTM(256*2)"]
    expected_network_1 = nn.Sequential(
        nn.Linear(input_shape, 32), nn.Linear(32, 64), nn.Linear(64, output_shape)
    )
    expected_network_2 = nn.Sequential(
        nn.Linear(input_shape, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.Tanh(),
        nn.Linear(64, output_shape),
        nn.Softmax(dim=-1),
    )
    expected_network_3 = nn.Sequential(
        nn.Linear(input_shape, 48),
        nn.SiLU(),
        nn.LSTM(
            48,
            128,
            1,
            batch_first=True,
        ),
        nn.LSTM(
            128,
            256,
            2,
            batch_first=True,
        ),
        nn.Linear(256, output_shape),
    )
    expected_networks = (expected_network_1, expected_network_2, expected_network_3)
    network_architectures = (
        network_architecture_1,
        network_architecture_2,
        network_architecture_3,
    )
    actors = (False, True, False)
    for expected_network, network_architecture, actor in zip(
        expected_networks, network_architectures, actors
    ):
        network = get_network_from_architecture(
            input_shape=input_shape,
            output_shape=output_shape,
            architecture=network_architecture,
            actor=actor,
            weight_init=False,
        )
        assert str(list(network.modules())[1:]) == str(
            list(expected_network.modules())[1:]
        )
    network = get_network_from_architecture(
        input_shape=input_shape,
        output_shape=output_shape,
        architecture=network_architecture_1,
        actor=True,
        weight_init=True,
    )
    network = get_network_from_architecture(
        input_shape=input_shape,
        output_shape=output_shape,
        architecture=network_architecture_1,
        actor=False,
        weight_init=True,
    )
    with pytest.raises(ValueError):
        # wrong archi length
        faulty_network = get_network_from_architecture(
            input_shape=input_shape,
            output_shape=output_shape,
            architecture=[],
            actor=True,
            weight_init=True,
        )
    with pytest.raises(ValueError):
        # wrong archi layer
        faulty_network = get_network_from_architecture(
            input_shape=input_shape,
            output_shape=output_shape,
            architecture=["RANDOMLAYER(128)"],
            actor=True,
            weight_init=True,
        )
    with pytest.raises(NotImplementedError):
        # wrong activation
        faulty_network = get_network_from_architecture(
            input_shape=input_shape,
            output_shape=output_shape,
            architecture=["random_activation"],
            actor=True,
            weight_init=True,
        )
        faulty_network


def test_compute_KL_divergence():
    beta_distrib = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
    assert compute_KL_divergence(beta_distrib, beta_distrib) == 0


def test_get_device():
    GPU = get_device("GPU")
    CPU = get_device("CPU")
    CPU
    GPU


def test_linear_schedule():
    linear_scheduler = LinearSchedule(1, 0, 100)
    assert linear_scheduler.transform(t=50) == 0.5
    assert linear_scheduler.transform(t=100) == 0
    assert linear_scheduler.transform(t=150) == 0
