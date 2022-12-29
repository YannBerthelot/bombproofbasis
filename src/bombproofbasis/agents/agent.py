"""
Base module for agents
"""

import os
from abc import abstractmethod
from datetime import date

import gym
import numpy as np

from bombproofbasis.agents.utils import get_action_shape
from bombproofbasis.types import AgentConfig
from bombproofbasis.utils.scalers import Scaler


class Agent:
    """
    Base class for agents
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Init the different internals.

        Args:
            config (AgentConfig): _description_
        """
        self._config = config
        self.scaler = (
            Scaler(config=self._config.scaler) if self._config.scaler.scale else None
        )

        # For logging purpose
        self.log_dir = self.create_dirs()
        self.best_episode_reward, self.episode = -np.inf, 1

    @property
    def env(self) -> gym.Env:
        """
        Returns:
            gym.Env: The current gym env
        """
        return self._config.environment

    @env.setter
    def env(self, new_env: gym.Env) -> None:
        """
        Set the new environment while checking for space dimensions

        Args:
            new_env (gym.Env): The new environment to be used

        Raises:
            ValueError: Prevents the change if the spaces don't match
        """
        if (new_env.observation_space == self.env.observation_space) & (
            new_env.action_space == self.env.action_space
        ):
            self._config.environment = new_env
        else:
            raise ValueError(
                "Environment spaces don't match. Check new \
                environment."
            )

    @property
    def obs_shape(self) -> int:
        """
        Returns:
            tuple: Wrapper for the obs space shape
        """
        return self.env.observation_space.shape[0]

    @property
    def action_shape(self) -> int:
        """
        Returns:
            int: Wrapper for the action space shape
        """
        return get_action_shape(self.env)

    @abstractmethod
    def select_action(self, observation: np.ndarray) -> None:
        """
        Abstract method for action selection to be implented in each agent.

        Args:
            observation (np.ndarray): The observation to consider for action selection

        Raises:
            NotImplementedError: To be implemented in the specific agent
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, env: gym.Env = None) -> None:
        """
        Abstract method for training agent to be implented in each agent.

        Args:
            env (gym.Env, optional): The environment to train on if different\
                 from the class one. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, env: gym.Env, nb_episodes: int, render: bool = False) -> None:
        """
        Abstract method for testing agent to be implented in each agent.

        Args:
            env (gym.Env): The environment to test on.
            nb_episodes (int): The number of episodes to test on
            render (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def save(self, name: str = "model") -> None:
        """
        Wrapper method for saving the network weights.

        Args:
            name (str, optional): Name of the save model file. Defaults to \
                "model".
        """
        raise NotImplementedError

    def load(self, name: str) -> None:
        """
        Wrapper method for loading the network weights.

        Args:
            name (str, optional): Name of the save model file.
        """
        raise NotImplementedError

    def create_dirs(self) -> str:
        """
        Create relevant dirs for the agent

        Returns:
            str: The path to use for the agent
        """
        today = date.today().strftime("%d-%m-%Y")
        os.makedirs(self._config.policy_network.model_path, exist_ok=True)
        return f"{self._config.log_path}/{self._config.environment}/{today}"

        # def save_if_best(
        #     self, best_episode_reward: float, reward_sum: float, logging: str, comment: str
        # ) -> wandb.Artifact:
        #     """
        #     Save the model's weights when it reaches a better performance

        #     Args:
        #         best_episode_reward (float): The best performance achieved so far
        #         reward_sum (float): The new performance
        #         logging (str): logging mode (in "wandb", "tensorboard" and "none")
        #         comment (str): comment for logging purposes

        #     Returns:
        #         wandb.Artifact: _description_
        #     """
        #     artifact = None
        #     if reward_sum >= best_episode_reward:
        #         best_episode_reward = reward_sum
        #         if logging == "wandb":
        #             wandb.run.summary["Train/best reward sum"] = reward_sum
        #             artifact = wandb.Artifact(f"{comment}_best", type="model")
        #         self.save(f"{comment}_best")
        #     return artifact

        # def early_stopping(self, reward_sum: float) -> bool:
        #     """
        #     Tells wether or not to stop the training now instead of waiting for \
        #         all timesteps to be completed.

        #     Args:
        #         reward_sum (float): The performance of the model

        #     Returns:
        #         bool: Wether or not to stop training.
        #     """
        #     if reward_sum == self.old_reward_sum:
        #         self.constant_reward_counter += 1
        #         if self.constant_reward_counter > self.config["GLOBAL"].getint(
        #             "early_stopping_steps"
        #         ):
        #             print(
        #                 f'Early stopping due to constant reward for\
        #  {self.config["GLOBAL"].getint("early_stopping_steps")} steps'
        #             )
        #             return True
        #     else:
        #         self.constant_reward_counter = 0
        #     return False

        # def episode_logging(self, reward_sum: float, actions_taken: dict) -> None:
        #     """_summary_

        #     Args:
        #         reward_sum (float): _description_
        #         actions_taken (dict): _description_
        #     """
        #     n_actions = sum(actions_taken.values())
        #     action_frequencies = {
        #         action: n / n_actions for action, n in actions_taken.items()
        #     }
        #     if self.config["GLOBAL"]["logging"].lower() == "wandb":
        #         for action, frequency in action_frequencies.items():
        #             wandb.log(
        #                 {
        #                     f"Actions/{action}": frequency,
        #                 },
        #                 step=self.t_global,
        #                 commit=False,
        #             )
        #         wandb.log(
        #             {
        #                 "Train/Episode_sum_of_rewards": reward_sum,
        #                 "Train/Episode": self.episode,
        #             },
        #             step=self.t_global,
        #             commit=True,
        #         )
        #     elif self.config["GLOBAL"]["logging"].lower() == "tensorboard":
        #         self.network.writer.add_scalar(
        #             "Reward/Episode_sum_of_rewards", reward_sum, self.episode
        #         )
