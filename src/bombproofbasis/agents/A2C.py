"""
A2C agent
"""
from pathlib import Path
from typing import Tuple

import gym
import numpy as np
import torch

# Base class for Agent
from bombproofbasis.agents.agent import Agent
from bombproofbasis.network.network import BaseTorchNetwork
from bombproofbasis.types import A2CConfig, BufferStep
from bombproofbasis.utils.buffer import RolloutBuffer

# from bombproofbasis.utils.normalize import SimpleStandardizer


class A2C(Agent):
    """
    Advantage Actor-Critic Agent

    Args:
        Agent : Base Agent class
    """

    def __init__(self, config: A2CConfig) -> None:
        """
        Init the A2C components:
        -rollout buffer
        -actor and critic networks

        Args:
            config (A2CConfig): _description_
        """
        super().__init__(config)

        self.rollout = RolloutBuffer(config.buffer)
        self.networks = A2CNetworks(config)
        self.t, self.t_global = 1, 1

    def save(self, folder: Path, name: str = "model"):
        """
        Save the current state of the model

        Args:
            folder (Path): the folder where to save the model
            name (str, optional): Model's name. Defaults to "model".
        """
        self.networks.save(folder=folder, name=name)

    def load(self, folder: Path, name: str = "model"):
        """
        Load the designated save of the model

        Args:
            folder (Path): the folder where to save the model
            name (str, optional): Model's name. Defaults to "model".
        """
        self.networks.load(folder=folder, name=name)

    def collect_rollout_episode(self, env: gym.Env) -> torch.Tensor:
        """
        Collect experience over a whole episode.

        Args:
            env (gym.Env): The environment to use for collection.

        Returns:
            torch.Tensor: The estimation of the final's state value by the \
                critic.
        """
        obs, _ = env.reset()
        self.rollout.internals.states[0].copy_(self.rollout.obs2tensor(obs))
        done, truncated = False, False
        t = 0
        self.networks.reset_hiddens()
        while not (done or truncated):
            state = self.rollout.get_state(t)
            action, log_prob = self.networks.select_action(state)
            value = self.networks.get_value(state=state)
            obs, reward, done, truncated, _ = env.step(action)
            print(reward)
            self.rollout.add(
                BufferStep(
                    reward=reward,
                    obs=obs,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    done=done,
                )
            )
            t += 1
        with torch.no_grad():
            final_value = self.networks.get_value(state=self.rollout.get_state(t))
        return final_value

    def update_policy(self, final_value: torch.Tensor):
        """
        Update the agent given the A2C update rule and update the \
            relevant other parameters

        Args:
            final_value (torch.Tensor): The estimation of the value, by the \
                critic, of the final state of the rollout.
        """
        self.networks.update(self.rollout, final_value)
        self.rollout.reset()

    def train(self, env: gym.Env, n_episodes: int) -> dict:
        cumulative_reward_per_episode = {}
        for episode in range(n_episodes):
            final_value = self.collect_rollout_episode(env)
            self.update_policy(final_value)
            cumulative_reward_per_episode[
                episode
            ] = self.rollout.internals.rewards.sum()
        return cumulative_reward_per_episode

    def test(self, env: gym.Env, n_episodes: int, render: bool = False) -> dict:
        cumulative_reward_per_episode = {}
        for episode in range(n_episodes):
            self.networks.reset_hiddens()
            cumulative_reward_per_episode[episode] = 0
            obs, _ = env.reset()
            done, truncated = False, False
            while not (done or truncated):
                action = self.select_action(obs)
                (
                    obs,
                    reward,
                    done,
                    truncated,
                    _,
                ) = env.step(action)
                if render:
                    env.render()
                cumulative_reward_per_episode[episode] += reward
        return cumulative_reward_per_episode

    def select_action(self, observation: np.ndarray) -> int:
        """
        Action selection wrapper from numpy to numpy without gradients
        See A2CNetworks's select action for one with gradients.
        Args:
            observation (np.ndarray): The observation for which to select \
                action

        Returns:
            np.ndarray: The selected action
        """
        observation = np.expand_dims(observation, axis=0)
        with torch.no_grad():
            probs = self.networks.actor(self.rollout.obs2tensor(observation))
            dist = torch.distributions.Categorical(probs=probs)  # gradient needed
            action = int(dist.sample().item())
        return action

    def get_value(self, observation: np.ndarray) -> float:
        """
        Get the value of a state given the current critic
        Does not compute any gradient, only used for logging/debugging.

        Args:
            observation (np.ndarray): The observation to evaluate

        Returns:
            float: The observation value
        """
        observation = np.expand_dims(observation, axis=0)
        with torch.no_grad():
            return self.networks.critic(self.rollout.obs2tensor(observation)).item()


# class Logger:
#     """
#     Logger for A2C agent. Used to handle the logging into wandb for the \
#         agent's training.
#     """

#     def __init__(self, config: LoggingConfig) -> None:
#         """Create the logger based on the config

#         Args:
#             config (LoggingConfig): Logger config
#         """
#         self.config = config

#     def log(self, timestep: int) -> None:
#         """
#         Logs the relevant values into wandb

#         Args:
#             timestep (int): The current timestep
#         """

# if timestep % self.config.logging_frequency == 0:

#     if self.config["GLOBAL"]["logging"].lower() == "tensorboard":
#         if self.writer:
#             self.writer.add_scalar(
#                 "Train/entropy loss", -entropy_loss, self.index
#             )
#             self.writer.add_scalar(
#                 "Train/leaarning rate",
#                 self.lr_scheduler.transform(self.index),
#                 self.index,
#             )
#             self.writer.add_scalar("Train/policy loss", actor_loss, self.index)
#             self.writer.add_scalar("Train/critic loss", critic_loss, self.index)
#             # self.writer.add_scalar(
#             #     "Train/explained variance", explained_variance, self.index
#             # )
#             # self.writer.add_scalar("Train/kl divergence", KL_divergence, self.index)
#         else:
#             warnings.warn("No Tensorboard writer available")
#     elif self.config["GLOBAL"]["logging"].lower() == "wandb":
#         wandb.log(
#             {
#                 "Train/entropy loss": -entropy_loss,
#                 "Train/actor loss": actor_loss,
#                 "Train/critic loss": critic_loss,
#                 # "Train/explained variance": explained_variance,
#                 # "Train/KL divergence": KL_divergence,
#                 "Train/learning rate": self.lr_scheduler.transform(self.index),
#             },
#             commit=False,
#         )


class A2CNetworks:
    """
    Network elements for A2C agent : the actor and critic network with their \
        connected methods
    """

    def __init__(self, config: A2CConfig) -> None:
        """
        Init the actor and critic networks according to config

        Args:
            config (A2CConfig): The Agent's config
        """
        super().__init__()
        self.config = config
        self.actor = BaseTorchNetwork(config.policy_network)
        self.critic = BaseTorchNetwork(config.value_network)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.policy_network.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.value_network.learning_rate,
        )

        # Init stuff
        self.index = 0
        self.old_dist = None

    def reset_hiddens(self):
        """
        Wrapper to init hidden states of networks
        """
        self.actor.init_hiddens()
        self.critic.init_hiddens()

    def select_action(
        self,
        observation: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """
        Select action based on observation (and hidden state if using a \
            reccurent network).
        Also returns relevant elements for updating the networks.

        Args:
            observation (np.ndarray): The observation of the environment to \
                consider
            hiddens (Dict[int, Tuple[torch.Tensor, torch.Tensor]]): \
                The hidden(s) state(s) of the actor network

        Returns:
            Tuple[ int, Dict[int, Tuple[torch.Tensor, torch.Tensor]],\
                 Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ]:\
                    The selected action, the new hidden(s) state(s), \
                        the relevant features for the A2C update \
                            (log_prob,entropy and KL div)
        """
        probs = self.actor(observation)
        dist = torch.distributions.Categorical(probs=probs)  # gradient needed
        action = dist.sample().detach()
        log_prob = dist.log_prob(action)  # gradient needed
        # entropy = dist.entropy().detach()  # gradient needed

        # KL_divergence = (
        #     compute_KL_divergence(self.old_dist, dist)
        #     if self.old_dist is not None
        #     else 0
        # )  # gradient needed

        return int(action.item()), log_prob

    def A2C_loss(
        self, log_prob: torch.Tensor, advantage: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss according to the A2C update rules for the critic and \
            actor

        Args:
            log_prob (torch.Tensor): The log-probability of the actions
            advantage (torch.Tensor): The advantages of the encoutered states

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The actor and critic losses
        """
        policy_loss = (-log_prob * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        return policy_loss, value_loss

    def update(self, buffer: RolloutBuffer, final_value: torch.Tensor) -> None:
        """
        Update the nework's weights according to A2C rule.

        Args:
            advantages (torch.Tensor): advantage to consider
            log_prob (torch.Tensor): log_prob to consider
        """

        # For logging purposes
        torch.autograd.set_detect_anomaly(True)

        self.index += 1
        actor_loss, critic_loss = self.A2C_loss(
            buffer.internals.log_probs[: buffer.internals.len],
            buffer.compute_advantages(final_value)[: buffer.internals.len],
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=False)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=False)
        self.critic_optimizer.step()
        # Logging
        # logger.log(self.index)

    def get_action_probabilities(self, state: torch.Tensor) -> np.ndarray:
        """
        Computes the policy pi(s, theta) for the given state s and for the \
            current policy parameters theta.
        Same as forward method with a clearer name for teaching purposes, but \
            as forward is a native method that needs to exist we keep both.
        Additionnaly this methods outputs np.array instead of torch.Tensor to \
            prevent the existence of pytorch stuff outside of network.py
        Shouldn't be used for update as the hidden state is not updated.
        Args:
            state (np.array): np.array representation of the state

        Returns:
            np.array: np.array representation of the action probabilities
        """

        return self.actor(state).detach().cpu().numpy()

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the state value for the given state s and for the current \
            policy parameters theta.
        Same as forward method with a clearer name for teaching purposes, but \
            as forward is a native method that needs to exist we keep both.
        Additionnaly this methods outputs np.array instead of torch.Tensor to \
            prevent the existence of pytorch stuff outside of network.py

        Args:
            state (np.array): np.array representation of the state

        Returns:
            np.array: np.array representation of the action probabilities
        """
        value = self.critic(state)

        return torch.squeeze(value)

    def save(self, folder: Path, name: str = "model") -> None:
        """
        Save the current model

        Args:
            name (str, optional): [Name of the model]. Defaults to "model".
        """
        torch.save(self.actor, f"{folder}/{name}_actor.pth")
        torch.save(self.critic, f"{folder}/{name}_critic.pth")

    def load(self, folder: Path, name: str = "model") -> None:
        """
        Load the designated model

        Args:
            name (str, optional): The model to be loaded (it should be in the \
                "models" folder). Defaults to "model".
        """
        print("Loading")
        self.actor = torch.load(f"{folder}/{name}_actor.pth")
        self.critic = torch.load(f"{folder}/{name}_critic.pth")

    # def fit_transform(self, x) -> torch.Tensor:
    #     self.scaler.partial_fit(x)
    #     if self.index > 2:
    #         return t(self.scaler.transform(x))

    # def gradient_clipping(self) -> None:
    #     clip_value = self.config["AGENT"].getfloat("gradient_clipping")
    #     if clip_value is not None:
    #         for optimizer in [self.actor_optimizer, self.critic_optimizer]:
    #             nn.utils.clip_grad_norm_(
    #                 [p for g in optimizer.param_groups for p in g["params"]],
    #                 clip_value,
    #             )
