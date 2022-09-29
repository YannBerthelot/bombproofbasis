"""
Define the base network classes
"""
from abc import abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

# PyTorch
from torch import nn

# Network creator tool
from bombproofbasis.network.utils import get_device, get_network_from_architecture
from bombproofbasis.types import NetworkConfig

ZERO = 1e-7


class BaseTorchNetwork(nn.Module):
    """
    Base class for networks of different agents/type

    Inherits from torch.nn base network Module
    """

    def __init__(self, config: NetworkConfig):
        """
         Creates the relevant attributes based on the config

        Args:
            config (NetworkConfig): Configuration of the network's\
                 architecture and activations
        """
        super().__init__()
        self.config = config
        self.network = get_network_from_architecture(
            input_shape=config.input_shape,
            output_shape=config.output_shape,
            architecture=config.architecture,
            actor=config.actor,
        )
        self.device = get_device(self.config.hardware)

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> Union[int, np.ndarray]:
        """
        Select the action based on the observation and the current state of \
            the network

        Actor-specific

        Args:
            obs (np.ndarray): The observation to consider

        Raises:
            NotImplementedError: To be implemented in each specific agent.

        Returns:
            Union[int, np.ndarray]: The action(s) (discrete or continuous)
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_obs(self, obs: np.ndarray) -> float:
        """
        Evaluate the observation given the current state of the value network

        Critic specific

        Args:
            obs (np.ndarray): The observation to consider

        Raises:
            NotImplementedError: To be implemented in each specific agent.

        Returns:
            float : the value of the observation
        """
        raise NotImplementedError

    @abstractmethod
    def update_policy(self):
        """
        Update the policy according to the agent's specific rules.

        Raises:
            NotImplementedError: To be implemented in each specific agent.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path):
        """
        Save the network(s)'s weights.

        Args:
            path (Path): The path in which the weights should be saved

        Raises:
            NotImplementedError: To be implemented in each specific agent.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path):
        """
        Load the network(s)'s weights.

        Args:
            path (Path): The path from which the weights should be loaded

        Raises:
            NotImplementedError: To be implemented in each specific agent.
        """
        raise NotImplementedError

    @property
    def architecture(self) -> list:
        """
        Returns:
            list: The network's architecture
        """
        return self.config.architecture


# class ActorCriticRecurrentNetworks(nn.Module):
#     """
#     TBRD
#     """

#     def __init__(
#         self,
#         state_dim,
#         action_dim,
#         architecture,
#         actor=True,
#         activation="relu",
#     ):
#         super().__init__()
#         self._action_dim = action_dim
#         self._state_dim = state_dim
#         self._architecture = architecture[1:-1].split(",")
#         self.actor = actor
#         self.activation = activation
#         self.network = self.init_layers()
#         print("actor" if actor else "critic", self.network)

#     @property
#     def architecture(self):
#         return self._architecture

#     @property
#     def state_dim(self):
#         return self._state_dim

#     @property
#     def action_dim(self):
#         return self._action_dim

#     @property
#     def hidden_dim(self):
#         return self._hidden_dim

#     @property
#     def num_layers(self):
#         return self._num_layers

#     def initialize_hidden_states(self):
#         hiddens = {}
#         for i, layer in enumerate(self.network):
#             if isinstance(layer, torch.nn.modules.rnn.LSTM):
#                 hiddens[i] = ActorCriticRecurrentNetworks.get_initial_states(
#                     hidden_size=layer.hidden_size, num_layers=layer.num_layers
#                 )
#         return hiddens

#     def init_layers(self) -> torch.nn.Sequential:
#         # Device to run computations on
#         self.device = "CPU"
#         output_size = self.action_dim if self.actor else 1
#         return get_network_from_architecture(
#             self.state_dim,
#             output_size,
#             self.architecture,
#             activation_function=self.activation,
#             mode="actor" if self.actor else "critic",
#         )

#     def forward(
#         self,
#         input: torch.Tensor,
#         hiddens: dict = None,
#     ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
#         """
#         Layers shared by the actor and the critic:
#         -Some FCs
#         -LSTM cell

#         Args:
#             state (torch.Tensor): State to be processed
#             hidden (Dict[torch.Tensor]): Hidden states of the LSTM

#         Returns:
#             Tuple[Torch.Tensor, Torch.Tensor]: The processed state and \
#                 the new hidden state of the LSTM
#         """
#         for i, layer in enumerate(self.network):
#             if isinstance(layer, torch.nn.modules.rnn.LSTM):
#                 input = input.view(-1, 1, layer.input_size)
#                 hiddens[i] = (hiddens[i][0].detach(), hiddens[i][1].detach())
#                 input, hiddens[i] = layer(input, hiddens[i])
#             else:
#                 input = layer(input)
#         return input, hiddens

#     @staticmethod
#     def get_initial_states(hidden_size, num_layers):
#         h_0, c_0 = None, None

#         h_0 = torch.zeros(
#             (
#                 num_layers,
#                 1,
#                 hidden_size,
#             ),
#             dtype=torch.float,
#         )

#         c_0 = torch.zeros(
#             (
#                 num_layers,
#                 1,
#                 hidden_size,
#             ),
#             dtype=torch.float,
#         )
#         return (h_0, c_0)
