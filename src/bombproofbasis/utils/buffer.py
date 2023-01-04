"""
Buffer components
"""

from typing import Union

import numpy as np
import torch

from bombproofbasis.types import BufferConfig, BufferInternals, BufferStep


class RolloutBuffer:
    """
    Buffer for experience collection and value computation in online RL for \
        actor-critic type agents
    """

    def __init__(self, config: BufferConfig) -> None:
        """
        Init the buffer internals and configuration

        Args:
            config (BufferConfig): The buffer configuration \
                - setting : MC or n-step
                - gamma : The discount factor (between 0 and 1)
                - buffer_size : Number of experiences to collect before \
                    computing returns.
                - n_steps : The number of steps to consider for return \
                    computation

        Raises:
            ValueError: If buffer size is not strictly positive
            ValueError: If gamma is not beween 0 and 1
            ValueError: If the number of steps is not strictly positive
        """
        self.config = config
        if self.config.buffer_size < 1:
            raise ValueError("Buffer size must be strictly positive")
        if not (0 <= self.config.gamma and self.config.gamma <= 1):
            raise ValueError("Gamma must be between 0 and 1")
        if self.config.n_steps < 1:
            raise ValueError("Number of steps must be strictly positive")
        self.reset()

    def reset(self) -> None:
        """
        Set or reset the buffer internals to its original state.
        """
        if self.config.setting == "MC":
            buffer_size = 1000
        else:
            buffer_size = self.config.buffer_size  # + self.config.n_steps
        self.internals = BufferInternals(
            rewards=torch.zeros((buffer_size, 1)),
            dones=torch.zeros((buffer_size, 1)),
            values=torch.zeros((buffer_size, 1)),
            log_probs=torch.zeros((buffer_size, 1)),
            entropies=torch.zeros((buffer_size, 1)),
            returns=torch.zeros((buffer_size, 1)),
            advantages=torch.zeros((buffer_size, 1)),
            states=torch.zeros((buffer_size + 1, 1, self.config.obs_shape[0])),
            actions=torch.zeros((buffer_size, 1)),
            len=0,
        )

    @property
    def done(self) -> bool:
        """
        Returns:
            bool: Wether or not the episode has ended.
        """
        return max(self.internals.dones) == 1

    @property
    def full(self) -> bool:
        """
        Returns:
            bool: Wether or not the buffer has reached full capacity.
        """
        return (self.internals.len >= self.config.buffer_size) or self.done

    def add(self, step: BufferStep) -> None:
        """
        Add new elements into the buffer.

        Args:
        buffer_step (BufferStep) : The step to consider with :
                reward (float): Reward to add
                done (bool): Termination flag to add
                value (float): Value (from critic) to add
                log_prob (float): Log prob (from actor) to add
                entropy (float): Entropy (from actor) to add
                KL_divergence (float): KL-divergence (from actor) to add

        Raises:
            ValueError: If the buffer is full already.
        """
        if (
            self.full
            and self.config.setting != "MC"
            or (self.config.setting == "MC" and self.done)
        ):
            raise ValueError("Buffer is already full, cannot add anymore")
        self.internals.rewards[self.internals.len].copy_(step.reward)
        self.internals.states[self.internals.len + 1].copy_(self.obs2tensor(step.obs))
        self.internals.actions[self.internals.len].copy_(step.action)
        self.internals.log_probs[self.internals.len].copy_(step.log_prob)
        self.internals.values[self.internals.len].copy_(step.value)
        self.internals.dones[self.internals.len].copy_(step.done)
        self.internals.len += 1

    def _generate_buffer(self, shape: tuple) -> torch.Tensor:
        """
        Create a torch tensor of zeros of the relevant shape

        Args:
            shape (tuple): The desired shape

        Returns:
            torch.Tensor: The initialized buffer
        """
        return torch.zeros(shape)

    def compute_return(self, final_value: torch.Tensor) -> torch.Tensor:
        """
        Compute return based on the internals of the buffer and the final \
            value (critic value of the final state, which is not included in the buffer)

        Args:
            final_value (torch.Tensor): Value prediction from critic for the \
                final state of the rollout

        Returns:
            torch.Tensor: The list of returns as a tensor of single item tensors
        """
        r_discounted = self._generate_buffer((self.internals.len, self.config.n_envs))
        # Init return with final value (estimate of final reward if episode has not finished)
        R = self._generate_buffer((1, self.config.n_envs)).masked_scatter(
            (1 - self.internals.dones[-1]), final_value
        )

        for i in reversed(range(self.internals.len)):
            discounted_return = self.internals.rewards[i] + self.config.gamma * R
            mask = (
                1 - self.internals.dones[i]
            )  # wether or not to replace the current value by new one
            R = self._generate_buffer((1, self.config.n_envs)).masked_scatter(
                mask.bool(), discounted_return
            )
            r_discounted[i] = R

        return r_discounted

    @staticmethod
    def obs2tensor(obs: np.ndarray) -> torch.Tensor:
        """
        Convert an observation from gym env to a torch.Tensor

        Args:
            obs (np.ndarray): Observation

        Returns:
            torch.Tensor: Transformed observation
        """
        tensor = torch.from_numpy(obs.astype(np.float32))
        return tensor

    def get_state(self, step):
        """
        Returns the observation of index step as a cloned object,
        otherwise torch.nn.autograd cannot calculate the gradients
        (indexing is the culprit)
        :param step: index of the state
        :return:
        """
        return self.internals.states[step].clone()

    def compute_advantages(
        self, final_value: torch.Tensor
    ) -> Union[torch.Tensor, None]:
        """
        Wrapper to compute advantages in either MC or n-step fashion

        Args:
            final_value (torch.Tensor): The estimation of the value of the \
                final state encoutered by the critic.

        Raises:
            ValueError: The setting of the buffer is not handled (should be \
                either "MC" or "n-step")

        Returns:
            Union[torch.Tensor, None]: The advantages tensor or None if in \
                n-step setting there is not enough data in the buffer to compute\
                the n-step advantages (due to not having the n-step value)
        """
        if self.config.setting == "MC":
            return self.compute_advantages_MC(final_value)
        elif self.config.setting == "n-step":
            if self.internals.len >= self.config.n_steps:
                return self.compute_advantages_n_step(
                    final_value, n_steps=self.config.n_steps
                )
            return None
        else:
            raise ValueError(f"Unrecognized buffer setting : {self.config.setting}")

    def compute_advantages_MC(self, final_value: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages based on the buffer storage and the final value in \
            a Monte-Carlo fashion (full episode, no need to predict next state \
            value as actual return can be computed).

        Args:
            final_value (torch.Tensor): Final value from the critic

        Returns:
            torch.Tensor: Tensor of advantages (how good were the states when\
                 compared to expectations from the critic)
        """
        returns = self.compute_return(final_value)
        return returns - self.internals.values[: self.internals.len]

    def compute_advantages_n_step(
        self, final_value: torch.Tensor, n_steps: int = 1
    ) -> torch.Tensor:
        """
        Compute advantages based on the buffer storage and the final value.

        Args:
            final_value (torch.Tensor): Final value from the critic

        Returns:
            torch.Tensor: Tensor of advantages (how good were the states when\
                 compared to expectations from the critic)
        """
        returns = self.compute_return(final_value)
        next_values = self._generate_buffer(
            (self.internals.len - n_steps + 1, self.config.n_envs)
        )
        done = self.internals.dones.max() > 0
        next_values[:-1].copy_(
            self.internals.values[n_steps : (self.internals.len + 1 - int(done))]
        )
        next_values[-1].copy_(final_value)
        # print(
        #     "last",
        #     returns,
        #     next_values,
        #     self.internals.values[: self.internals.len - n_steps + 1],
        # )
        advantages = (
            returns[: self.internals.len - n_steps + 1]
            + (1 - self.internals.dones[: self.internals.len - n_steps + 1])
            * (self.config.gamma**n_steps)
            * next_values
            - self.internals.values[: self.internals.len - n_steps + 1]
        )
        return advantages

    def after_update(self):
        """
        Cleaning up buffers after a rollout is finished and
        copying the last state to index 0
        :return:
        """

        self.internals.states[0].copy_(self.internals.states[-1])
        self.internals.actions = self._generate_buffer(
            (self.config.buffer_size, self.config.n_envs)
        )
        self.internals.log_probs = self._generate_buffer(
            (self.config.buffer_size, self.config.n_envs)
        )
        self.internals.values = self._generate_buffer(
            (self.config.buffer_size, self.config.n_envs)
        )
        self.internals.len = 0
