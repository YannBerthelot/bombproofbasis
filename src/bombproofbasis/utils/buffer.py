"""
Buffer components
"""
from typing import Generator

import numpy as np
import numpy.typing as npt
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
        return (
            self.internals.len >= self.config.buffer_size + self.config.n_steps - 1
        ) or self.done

    def reset(self) -> None:
        """
        Set or reset the buffer internals to its original state.
        """
        buffer_size = self.config.buffer_size  # + self.config.n_steps
        self.internals = BufferInternals(
            rewards=np.zeros(buffer_size),
            dones=np.zeros(buffer_size),
            KL_divergences=np.zeros(buffer_size),
            values=torch.zeros(buffer_size),
            log_probs=torch.zeros(buffer_size),
            entropies=torch.zeros(buffer_size),
            len=0,
            returns=None,
            advantages=None,
        )

    def add(self, buffer_step: BufferStep) -> None:
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
        if self.full:
            raise ValueError("Buffer is already full, cannot add anymore")
        self.internals.dones[self.internals.len] = buffer_step.done
        self.internals.values[self.internals.len] = buffer_step.value
        self.internals.log_probs[self.internals.len] = buffer_step.log_prob
        self.internals.entropies[self.internals.len] = buffer_step.entropy
        self.internals.KL_divergences[self.internals.len] = buffer_step.KL_divergence
        self.internals.rewards[self.internals.len] = buffer_step.reward
        self.internals.len += 1

    @staticmethod
    def compute_return(rewards: npt.NDArray[np.float64], gamma: float) -> float:
        """
        Compute the return (cumulative discounted return considering \
            only n steps) for the given rewards

        Args:
            rewards (npt.NDArray[np.float64]): The n rewards to consider for \
                the n-step return
            gamma (float): The discount factor

        Returns:
            float: _description_
        """
        # return reduce(lambda a, b: gamma * a + b, reversed(rewards)) -> \
        # less efficient somehow
        n_step_return = 0
        for reward in reversed(rewards):
            n_step_return = reward + gamma * n_step_return
        return n_step_return

    @staticmethod
    def compute_MC_returns(rewards: npt.NDArray[np.float64], gamma: float) -> list:
        """
        Compute all returns in a Monte-Carlo fashion (need for the episode to \
            be finished) given the rewards for the episode.

        Args:
            rewards (npt.NDArray[np.float64]): The rewards for the whole episode.
            gamma (float): The discount factor (0 <= gamma <= 1)

        Returns:
            list: The returns for the whole episode
        """
        # return reduce(lambda a, b: gamma * a + b, reversed(rewards)) -> \
        # less efficient somehow
        MC_returns = []
        cum_return = 0
        for reward in reversed(rewards):
            cum_return = reward + gamma * cum_return
            MC_returns.append(cum_return)
        MC_returns = MC_returns[::-1]
        return MC_returns

    # @staticmethod
    # def compute_next_return(
    #     last_return: float, R_0: float, R_N: float, gamma: float, n_steps: int
    # ) -> float:
    #     """Compute next return based on previous return, and first and last \
    #         reward

    #     Args:
    #         last_return (float): The previous return value
    #         R_0 (float): The first (early) observed reward in the buffer
    #         R_N (float): The last (late) observed reward in the buffer
    #         gamma (float): The discount factor.
    #         n_steps (int): The number of steps to consider for return \
    #             computation

    #     Returns:
    #         float: The next value for return
    #     """
    #     return ((last_return - R_0) / gamma) + ((gamma**n_steps) * R_N)

    @staticmethod
    def compute_n_step_returns(
        rewards: npt.NDArray[np.float64],
        gamma: float,
        buffer_size: int,
        n_steps: int,
    ) -> list:
        """
        Compute all n-steps returns for the given rewards

        Args:
            rewards (npt.NDArray[np.float64]): All rewards to consider for \
                return computation
            gamma (float): The discount factor.
            buffer_size (int): The buffer size.
            n_steps (int): The number of steps to consider for return \
                computation.

        Returns:
            list: All returns for the given rewards.
        """
        returns = []
        for j in range(
            buffer_size
        ):  # only iterate for steps for which we have n rewards for full \
            # computation, the rest will be kept for future computations
            rewards_list = rewards[j : min(j + n_steps, len(rewards))]
            returns.append(RolloutBuffer.compute_return(rewards_list, gamma))
        return returns

    # @staticmethod
    # def compute_n_step_returns_iteratively(
    #     rewards: npt.NDArray[np.float64],
    #     gamma: float,
    #     buffer_size: int,
    #     n_steps: int,
    # ) -> np.ndarray:
    #     """
    #     Compute all returns in an iterative fashion. \
    #         UTILITY TO BE ASSESSED WHEN COMPARED TO compute_all_n_step_returns

    #     Args:
    #         rewards (npt.NDArray[np.float64]): The rewards to consider.
    #         gamma (float): The discount factor.
    #         buffer_size (int): The buffer size.
    #         n_steps (int): The number of steps to consider for return \
    #             computation.

    #     Returns:
    #         np.ndarray:  All returns for the given rewards.
    #     """

    #     returns = []

    #     # Compute initial return
    #     G_0 = RolloutBuffer.compute_return(rewards[0 : 0 + n_steps], gamma)
    #     returns.append(G_0)
    #     for i in range(buffer_size - 1):
    #         new_return = RolloutBuffer.compute_next_return(
    #             returns[i], rewards[i], rewards[i + n_steps], gamma, n_steps - 1
    #         )
    #         returns.append(new_return)
    #     return np.array(returns)

    @staticmethod
    def compute_n_step_advantages(
        returns: npt.NDArray[np.float64],
        values: torch.Tensor,
        dones: npt.NDArray[np.float64],
        gamma: float,
        n_steps: int,
    ) -> np.ndarray:
        """
        Compute advantages (How good was the action in that state when \
            compared to what I expected from the state) for the given \
                values in an n-step fashion.

        Args:
            returns (npt.NDArray[np.float64]): The returns to consider for \
                advantage computation.
            values (torch.Tensor): The values (from critic) to consider.
            dones (npt.NDArray[np.float64]): The termination flags to consider.
            gamma (float): The discount factor.
            n_steps (int): The number of steps to consider for return \
                computation.

        Raises:
            ValueError: If the number of steps is not positive.

        Returns:
            np.ndarray: The advantages.
        """

        # next_values = values[n_steps:]
        # next_values = np.append(next_values, last_val)
        if n_steps <= 0:
            raise ValueError(f"Invalid steps number : {n_steps}")
        return [
            returns[i]
            + (1 - max(dones[i : i + n_steps + 1]))
            * (gamma**n_steps)
            * values[i + n_steps]
            - values[i]
            for i in range(len(values) - n_steps)
        ]

    @staticmethod
    def compute_MC_advantages(
        returns: npt.NDArray[np.float64],
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute advantages in a Monte-Carlo fashion.

        Args:
            returns (npt.NDArray[np.float64]): The returns to consider
            values (torch.Tensor): The values (from critic) to consider

        Returns:
            torch.Tensor: The advantages.
        """
        return torch.sub(torch.tensor(returns), values)

    def update_advantages(self) -> None:
        """
        Computes the advantages based on values in buffer and the \
        buffer's config
        """
        # if (self.config.n_steps > 2) and fast:
        #     # Faster for high number of steps (complexity constant with n-steps)
        #     self.internals.returns = RolloutBuffer.compute_returns(
        #         self.internals.rewards,
        #         self.config.gamma,
        #         self.config.buffer_size,
        #         self.config.n_steps,
        #     )
        if self.config.setting == "MC":
            self.internals.returns = RolloutBuffer.compute_MC_returns(
                self.internals.rewards,
                self.config.gamma,
            )
        else:
            self.internals.returns = RolloutBuffer.compute_n_step_returns(
                self.internals.rewards,
                self.config.gamma,
                self.config.buffer_size,
                self.config.n_steps,
            )
        if self.config.setting == "MC":
            self.internals.advantages = RolloutBuffer.compute_MC_advantages(
                self.internals.returns,
                self.internals.values,
            )
            self.internals.advantages = self.internals.advantages[: self.internals.len]
            self.internals.log_probs = self.internals.log_probs[: self.internals.len]
        else:
            self.internals.advantages = RolloutBuffer.compute_n_step_advantages(
                self.internals.returns,
                self.internals.values,
                self.internals.dones,
                self.config.gamma,
                self.config.n_steps,
            )

    def show(self) -> None:
        """
        Print the different internal values.
        """
        print("REWARDS", self.internals.rewards)
        print("VALUES", self.internals.values)
        print("DONES", self.internals.dones)
        print("LOG PROBS", self.internals.log_probs)
        print("ENTROPIES", self.internals.entropies)
        print("KL_DIVERGENCES", self.internals.KL_divergences)
        print("ADVANTAGES", self.internals.advantages)

    def get_steps(self) -> tuple:
        """
        Return the relevant internals to be used for training:\
            -advantages
            -log probabilities of the actions
            -entropies
            -KL-divergences between policies


        Returns:
            tuple: The aforementionned internals
        """
        return (
            self.internals.advantages,
            self.internals.log_probs,
            self.internals.entropies,
            self.internals.KL_divergences,
        )

    def get_steps_generator(self) -> Generator:
        """
        Creates a generator for the steps (the relevant internal values)

        Yields:
            Generator: The steps generator
        """
        for advantage, log_prob, entropy, KL_div in zip(
            self.internals.advantages,
            self.internals.log_probs,
            self.internals.entropies,
            self.internals.KL_divergences,
        ):
            yield advantage, log_prob, entropy, KL_div

    # def clear(self) -> None:
    #     self.internals.rewards = self.internals.rewards[self.config.buffer_size :]
    #     self.internals.dones = self.internals.dones[self.config.buffer_size :]
    #     self.internals.KL_divergences = self.internals.KL_divergences[
    #         self.config.buffer_size :
    #     ]
    #     self.internals.values = self.internals.values[self.config.buffer_size :]
    #     self.internals.log_probs = self.internals.log_probs[self.config.buffer_size :]
    #     self.internals.entropies = self.internals.entropies[self.config.buffer_size :]

    def clean(self) -> None:
        """
        Remove the data that has already been processed from the buffer while \
            conserving data that will be used in next return computation.
        Seeting arrays to 0 and then modifying the slices is faster than \
            creating empty arrays and appending.
        """
        buffer_size = self.config.buffer_size + self.config.n_steps - 1
        old_rewards = self.internals.rewards[-self.config.n_steps :]
        self.internals.rewards = np.zeros(buffer_size)
        self.internals.rewards[: self.config.n_steps] = old_rewards
        self.internals.dones = np.zeros(buffer_size)
        self.internals.KL_divergences = np.zeros(buffer_size)
        self.internals.values = np.zeros(buffer_size)
        self.internals.log_probs = np.zeros(buffer_size)
        self.internals.entropies = np.zeros(buffer_size)
        self.internals.advantages = None
        self.internals.returns = None
        self.internals.len = self.config.n_steps
