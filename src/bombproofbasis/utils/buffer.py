"""
Buffer components
"""
from typing import Tuple, Union

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
            if not (self.config.buffer_size >= self.config.n_steps + 1) and (
                self.config.n_steps > 1
            ):
                raise ValueError(
                    f"Buffer size is not big enough for selected n-steps. \
                        Buffer size : {self.config.buffer_size}, n-steps : \
                        {self.config.n_steps}, buffer size should be at \
                        least {self.config.n_steps+1}"
                )
            buffer_size = self.config.buffer_size  # + self.config.n_steps
        self.internals = BufferInternals(
            rewards=torch.zeros((buffer_size, 1)),
            dones=torch.zeros((buffer_size, 1)),
            values=torch.zeros((buffer_size, 1)),
            log_probs=torch.zeros((buffer_size, 1)),
            entropies=torch.zeros((buffer_size, 1)),
            returns=torch.zeros((buffer_size, 1)),
            advantages=torch.zeros((buffer_size, 1)),
            states=torch.zeros((buffer_size + 1, 1, self.config.obs_shape)),
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

    def _generate_buffer(self, shape: Union[int, tuple]) -> torch.Tensor:
        """
        Create a torch tensor of zeros of the relevant shape

        Args:
            shape (tuple): The desired shape

        Returns:
            torch.Tensor: The initialized buffer
        """
        return torch.zeros(shape)

    def compute_return(self) -> torch.Tensor:
        """
        Wrapper function to compute return with right method.

        Args:
            final_value (torch.Tensor): Value of the final state according to the critic.

        Raises:
            ValueError: Unrecognized return computation setting

        Returns:
            torch.Tensor: The return
        """
        if self.config.setting == "MC":
            return self.compute_return_MC()
        elif self.config.setting == "n-step":
            if self.config.n_steps == 1:
                return self.compute_return_TD()
            elif self.config.buffer_size >= self.config.n_steps + 1:
                return self.compute_return_n_step()
            return None
        else:
            raise ValueError(f"Unrecognized buffer setting : {self.config.setting}")

    def compute_return_MC(self) -> torch.Tensor:
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
        R = self._generate_buffer((1, self.config.n_envs))

        for i in reversed(range(self.internals.len)):
            discounted_return = self.internals.rewards[i] + self.config.gamma * R
            R = self._generate_buffer((1, self.config.n_envs)).copy_(discounted_return)
            r_discounted[i] = R
        return r_discounted

    def compute_return_TD(self) -> torch.Tensor:
        """
        Compute return based on the internals of the buffer and the final \
            value (critic value of the final state, which is not included in the buffer)

        Args:
            final_value (torch.Tensor): Value prediction from critic for the \
                final state of the rollout

        Returns:
            torch.Tensor: The list of returns as a tensor of single item tensors
        """
        return self.internals.rewards

    def compute_return_n_step(self) -> torch.Tensor:
        """
        Compute return based on the internals of the buffer and the final \
            value (critic value of the final state, which is not included in the buffer)

        Args:
            final_value (torch.Tensor): Value prediction from critic for the \
                final state of the rollout

        Returns:
            torch.Tensor: The list of returns as a tensor of single item tensors
        """
        r_discounted = self._generate_buffer(
            (min(self.internals.len, self.config.n_steps), self.config.n_envs)
        )

        for i in range(min(self.internals.len, self.config.n_steps)):
            rewards = self.internals.rewards[
                i : min(self.internals.len, self.config.n_steps) + i
            ]
            dones = self.internals.dones[
                i : min(self.internals.len, self.config.n_steps) + i
            ]
            R = self._generate_buffer(self.config.n_envs)
            for j in reversed(range(len(rewards))):
                discounted_return = rewards[j] + self.config.gamma * R
                mask = 1 - dones[j]
                R = self._generate_buffer(self.config.n_envs).masked_scatter(
                    mask.bool(), discounted_return
                )
            r_discounted[i].copy_(R)
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
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
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
            return self.compute_advantages_MC()
        elif self.config.setting == "n-step":
            if self.config.n_steps == 1:
                return self.compute_advantages_TD(final_value)
            elif self.config.n_steps > 1:
                return self.compute_advantages_n_step(final_value)
            return None, None
        else:
            raise ValueError(f"Unrecognized buffer setting : {self.config.setting}")

    def compute_advantages_MC(self) -> torch.Tensor:
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
        returns = self.compute_return()
        return returns - self.internals.values[: self.internals.len], returns

    def compute_advantages_TD(
        self, final_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages based on the buffer storage and the final value.

        Args:
            final_value (torch.Tensor): Final value from the critic

        Returns:
            torch.Tensor: Tensor of advantages (how good were the states when\
                 compared to expectations from the critic)
        """
        returns = self.compute_return()[: self.internals.len]
        next_values = self._generate_buffer((self.internals.len, self.config.n_envs))
        next_values[:-1].copy_(
            self.internals.values[self.config.n_steps : self.internals.len]
        )
        next_values[-1].copy_(final_value)
        expected_value = (
            returns
            + (1 - self.internals.dones[: self.internals.len])
            * (self.config.gamma**self.config.n_steps)
            * next_values
        )
        advantages = expected_value - self.internals.values[: self.config.n_steps]
        return advantages, expected_value

    def compute_advantages_n_step(
        self, final_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages based on the buffer storage and the final value.

        Args:
            final_value (torch.Tensor): Final value from the critic

        Returns:
            torch.Tensor: Tensor of advantages (how good were the states when\
                 compared to expectations from the critic)
        """
        done = self.internals.dones.max() > 0
        # returns = [
        #     self.compute_return(
        #         self.internals.rewards[i : min(self.internals.len, n_steps) + i],
        #         self.internals.dones[i : min(self.internals.len, n_steps) + i],
        #     )
        #     for i in range(min(self.internals.len, n_steps))
        # ]
        returns = self.compute_return()
        next_values = self._generate_buffer(
            (
                self.internals.len
                - 1
                + int(done)
                - int(self.internals.dones[1:].max().item()),
                self.config.n_envs,
            )
        )
        next_values[:-1].copy_(self.internals.values[self.config.n_steps :])
        next_values[-1].copy_(final_value)
        expected_value = (
            returns
            + (1 - self.internals.dones[1:])
            * (self.config.gamma**self.config.n_steps)
            * next_values
        )
        advantages = (
            expected_value
            - self.internals.values[
                : self.internals.len - 1 + int(self.internals.dones[0])
            ]
        )
        return advantages, expected_value

    def after_update(self, agent=None):
        """
        Cleaning up buffers after a rollout is finished and
        copying the last state to index 0
        :return:
        """

        if self.config.n_steps > 1:
            step = BufferStep(
                reward=self.internals.rewards[-1].item(),
                obs=self.internals.states[-1].detach().numpy(),
                action=int(self.internals.actions[-1].detach().item()),
                log_prob=agent.networks.get_log_prob(
                    self.internals.states[-2].clone(),
                    self.internals.actions[-1].detach(),
                ),
                value=agent.networks.get_value(state=self.internals.states[-2].clone()),
                done=False,
            )
            self.internals.states[0].copy_(self.internals.states[-1])
            self.internals.values = self._generate_buffer(
                (self.config.buffer_size, self.config.n_envs)
            )
            self.internals.actions = self._generate_buffer(
                (self.config.buffer_size, self.config.n_envs)
            )
            self.internals.log_probs = self._generate_buffer(
                (self.config.buffer_size, self.config.n_envs)
            )

            self.internals.dones = self._generate_buffer(
                (self.config.buffer_size, self.config.n_envs)
            )

            self.internals.len = 0
            self.add(step)

        else:
            self.internals.states[0].copy_(self.internals.states[-1])
            self.internals.values = self._generate_buffer(
                (self.config.buffer_size, self.config.n_envs)
            )
            self.internals.actions = self._generate_buffer(
                (self.config.buffer_size, self.config.n_envs)
            )
            self.internals.log_probs = self._generate_buffer(
                (self.config.buffer_size, self.config.n_envs)
            )

            self.internals.dones = self._generate_buffer(
                (self.config.buffer_size, self.config.n_envs)
            )

            self.internals.len = 0
