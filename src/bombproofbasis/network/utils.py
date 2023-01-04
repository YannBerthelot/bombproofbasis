"""
Helper functions for the network module
"""
import re
import warnings
from typing import Any, List, Tuple

import numpy as np
import torch
from torch import nn


# helper function to convert numpy arrays to tensors
def t(x: np.ndarray) -> torch.Tensor:
    """
    Get torch.Tensor from a numpy array

    Args:
        x (np.ndarray): tensor to parse

    Returns:
        torch.Tensor: Transformed tensor
    """
    return torch.from_numpy(x)  # .float()


def init_weights(m: nn.Module, val: float = np.sqrt(2)) -> None:
    """
    Init the layer's weight using orthogonal init.

    Args:
        m (nn.Module): The layer to update
        val (float, optional): The init value. Defaults to np.sqrt(2).
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=val)
        m.bias.data.fill_(0)


def has_numbers(inputString: str) -> bool:
    """
    Checks wether a string has number inside or not

    Args:
        inputString (str): The string to be checked

    Returns:
        bool: Wether or not the string contains numbers
    """
    return any(char.isdigit() for char in inputString)


def get_activation_from_name(name: str) -> Any:
    """
    Returns the correct torch activation function based on name

    Args:
        name (str): The activation function name

    Raises:
        NotImplementedError: If the activation function is not implemented

    Returns:
        Any: The torch activation function
    """
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "tanh":
        return nn.Tanh()
    elif name.lower() in ("swish", "silu"):
        return nn.SiLU()
    raise NotImplementedError(f"No activation function like {name} implemented")


def get_LSTM_from_string(string: str, previous_shape: int) -> Tuple[nn.LSTM, int]:
    """
    Parse the LSTM architecture to the actual LSTM layer

    Args:
        string (str): The LSTM string representation
        previous_shape (int): The shape of the previous layer

    Returns:
        Tuple[nn.LSTM, int]: The LSTM layer and the number of neurons inside
    """
    LSTM_description = re.search(r"\((.*?)\)", string).group(1)
    if "*" in LSTM_description:
        nb_neurons = int(LSTM_description.split("*")[0])
        nb_layers = int(LSTM_description.split("*")[1])
    else:
        nb_neurons = int(LSTM_description)
        nb_layers = 1
    return (
        nn.LSTM(
            previous_shape,
            nb_neurons,
            nb_layers,
            batch_first=True,
        ),
        nb_neurons,
    )


def get_network_from_architecture(
    input_shape: int,
    output_shape: int,
    architecture: List[str],
    actor: bool,
    weight_init: bool = True,
) -> torch.nn.modules.container.Sequential:
    """
    Parses the architecture as a list of string towards the actual pytorch \
        network.

    Args:
        input_shape (int): The input shape of the network (observation size)
        output_shape (int): The output shape of the network (action size)
        architecture (List[str]): The list of string representation of the\
             architecture
        actor (bool): Wether or not this network is for an actor (if yes, add \
            a softmax at the end)
        weight_init (bool, optional): Wether or not to use orthogonal init \
            (square root of 2 for all layers, except the last one which will \
            be init to 0.01 for actors and 0.1 for critics). \
                cf : http://joschu.net/docs/nuts-and-bolts.pdf
            Defaults to True.

    Raises:
        ValueError: For unrecognized activations
        ValueError: For non-parsable architectures

    Returns:
        torch.nn.modules.container.Sequential: The actual network
    """
    if len(architecture) < 1:
        raise ValueError("Architecture is non-valid")
    architecture = [str(input_shape)] + architecture + [str(output_shape)]
    layers = []
    previous_shape = input_shape
    for i, element in enumerate(architecture):
        if element.isnumeric():
            nb_neurons = int(element)
            if i != 0:
                layers.append(nn.Linear(previous_shape, nb_neurons))
            previous_shape = nb_neurons
        elif has_numbers(element) and "LSTM" in element:
            LSTM_layer, previous_shape = get_LSTM_from_string(element, previous_shape)
            layers.append(LSTM_layer)
        elif has_numbers(element):
            raise ValueError(f"Unrecognized layer type {element}")
        else:
            activation = get_activation_from_name(name=element)
            layers.append(activation)
    if actor:
        layers.append(nn.Softmax(dim=-1))
    if weight_init:
        if actor:
            layers[-2].apply(lambda module: init_weights(m=module, val=(0.01)))
        else:
            layers[-1].apply(lambda module: init_weights(m=module, val=(1)))
        network = nn.Sequential(*layers)
        network.apply(init_weights)
    else:
        network = nn.Sequential(*layers)
    return network


def compute_KL_divergence(
    old_dist: torch.distributions.Distribution, dist: torch.distributions.Distribution
) -> float:
    """
    Compute the KL divergence between the old and new probability distribution\
         over (discrete) actions.

    Args:
        old_dist (torch.distributions.Distribution): The previous probability \
            distribution over actions
        dist (torch.distributions.Distribution): The new probability \
            distribution over actions

    Returns:
        float: the KL-divergence
    """
    return torch.distributions.kl_divergence(old_dist, dist).mean()


def get_device(device_name: str) -> torch.DeviceObjType:
    """
    Chose the right device for PyTorch. If no GPU is available, it will use CPU.

    Args:
        device_name (str): The device to use between "GPU" and "CPU"

    Returns:
        torch.DeviceObjType: The Torch.Device to use
    """
    if device_name == "GPU":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            warnings.warn("GPU not available, switching to CPU", UserWarning)
    else:
        device = torch.device("cpu")

    return device


class LinearSchedule:
    """
    Simple linear schedule for the evolution of a value over time
    """

    def __init__(self, start: float, end: float, t_max: int) -> None:
        """
        Init parameters

        Args:
            start (float): The initial value for the schedule
            end (float): The final value for the schedule
            t_max (int): How long before the end value is reached.
        """
        self.start = start
        self.end = end
        self.t_max = t_max
        self.step = (start - end) / t_max

    def transform(self, t: int) -> float:
        """
        Change the value according to the current timestep

        Args:
            t (int): The current timestep

        Returns:
            float: The new value after linear schedule
        """
        return max(self.end, self.start - self.step * t)
