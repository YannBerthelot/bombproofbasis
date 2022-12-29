"""
Define the base network classe
"""
from typing import Tuple, Union

import torch
from torch import nn

# Network creator tool
from bombproofbasis.network.utils import get_device, get_network_from_architecture
from bombproofbasis.types import NetworkConfig


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
        if not config.actor:
            if not config.output_shape == 1:
                raise ValueError(
                    f"Critic has an output shape different than one : {config.output_shape}"
                )
        self.device = get_device(self.config.hardware)
        self.recurrent = "LSTM" in str(self.network._modules)
        self.initialize_hidden_states()

    @staticmethod
    def get_initial_states(
        hidden_size: int, num_layers: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the initial (null) hidden states h_0 and c_0

        Args:
            hidden_size (int): The hidden size of the layer
            num_layers (int): The number of recurrent layers in the layer

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Both components of the \
                hidden state (h_0, c_0)
        """
        h_0, c_0 = None, None
        h_0 = torch.zeros(
            (
                num_layers,
                # batch_size,
                hidden_size,
            ),
            dtype=torch.float,
        )

        c_0 = torch.zeros(
            (
                num_layers,
                # batch_size,
                hidden_size,
            ),
            dtype=torch.float,
        )
        return (h_0, c_0)

    def initialize_hidden_states(self):
        """
        Initialize the hidden state(s) for the recurrent layer(s) \
            and wrap them in a dict
        """
        if self.recurrent:
            hiddens = {}
            for i, layer in enumerate(self.network):
                if isinstance(layer, nn.modules.rnn.LSTM):
                    hiddens[i] = self.get_initial_states(
                        hidden_size=layer.hidden_size, num_layers=layer.num_layers
                    )
            self.hiddens = hiddens
        else:
            self.hiddens = None

    def recurrent_forward(
        self,
        x: torch.Tensor,
        hiddens: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with at least one recurrent layer.

        Args:
            x (torch.Tensor): Input to be processed
            hidden (Dict[torch.Tensor]): Hidden state(s) of the LSTM layer(s) \
                in the following format {layer_index : (h_n, c_n)}

        Returns:
            Tuple[Torch.Tensor, Union[Type[None], dict]]: The processed input\
                and the new hidden state of the LSTM
        """
        new_hiddens = {}
        for i, layer in enumerate(self.network):
            if isinstance(layer, torch.nn.modules.rnn.LSTM):
                x, new_hidden = layer(
                    x,
                    (hiddens[i][0].detach(), hiddens[i][1].detach()),
                )
                new_hiddens[i] = new_hidden
                assert not (
                    (torch.equal(hiddens[i][0], new_hidden[0]))
                    and (torch.equal(hiddens[i][1], new_hidden[1]))
                )
            else:
                x = layer(x)
        return x, new_hiddens

    def forward(
        self, x: torch.Tensor, hiddens: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Wrapper function for forward calls depending on recurrency layers

        Args:
            x (torch.Tensor): Input to be processed
            # hiddens (torch.Tensor, optional): Hidden state(s) when using RNNs.\
            #   Defaults to None.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, \
                torch.Tensor]]]: either the output, or the output plus the \
                hidden states for recurrent networks.
        """
        if self.recurrent:
            x, hiddens = self.recurrent_forward(x, hiddens=hiddens)
            return x, hiddens
        return self.network(x)
