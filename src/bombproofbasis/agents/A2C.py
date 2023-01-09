"""
A2C agent
"""
from pathlib import Path
from typing import Tuple, Union

import gym
import numpy as np
import torch
from pydantic import PositiveInt
from tqdm import tqdm

# Base class for Agent
from bombproofbasis.agents.agent import Agent
from bombproofbasis.network.network import BaseTorchNetwork
from bombproofbasis.types import A2CConfig, BufferStep, LoggingConfig
from bombproofbasis.utils.buffer import RolloutBuffer
from bombproofbasis.utils.logging import Logger

# from bombproofbasis.utils.normalize import SimpleStandardizer

WEIGHTS = False


class A2C(Agent):
    """
    Advantage Actor-Critic Agent

    Args:
        Agent : Base Agent class
    """

    def __init__(
        self, config: A2CConfig, log_config: LoggingConfig = LoggingConfig()
    ) -> None:
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
        self.episode_reward, self.episode_num = 0, 0
        self.logger = Logger(log_config)
        probs = torch.tensor(
            [
                1 / self.networks.actor.config.input_shape
                for _ in range(self.networks.actor.config.input_shape)
            ]
        )
        self.max_entropy = torch.distributions.Categorical(probs=probs).entropy().item()

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

    def collect_rollout(
        self, env: gym.Env, n_timesteps: PositiveInt
    ) -> Tuple[torch.Tensor, gym.Env, bool, torch.Tensor]:
        """
        Collect the experience according to the buffer's config until either \
            the buffer is full or the episode is finished.

        Args:
            env (gym.Env): The env to use for experience collection.
            n_timesteps (PositiveInt): The number of timesteps to interact for.

        Returns:
            Tuple[torch.Tensor, gym.Env, bool, torch.Tensor]:
                - The estimation of the value  by the critic of the last \
                    state encoutered
                - The stepped environment in its new state.
                - Wether or not the environment is finished (done or Truncated).
                - The cumulative entropy of the rollout.
        """
        episode_entropy = 0
        for step in range(n_timesteps):
            state = self.rollout.get_state(step)
            action, log_prob, entropy = self.networks.select_action(state)
            obs, reward, done, truncated, _ = env.step(action)
            self.episode_reward += reward
            episode_entropy += entropy / self.max_entropy
            self.t_global += 1
            finished = done or truncated
            self.rollout.add(
                BufferStep(
                    reward=reward,
                    obs=obs,
                    action=action,
                    log_prob=log_prob,
                    value=self.networks.get_value(state=state),
                    done=finished,
                )
            )
            if finished:
                self.episode_num += 1
                self.logger.log(
                    {
                        "Reward/Episode": self.episode_reward,
                        "Relative Entropy": episode_entropy / (step + 1),
                    },
                    self.episode_num,
                )
                self.logger.log_histograms(
                    self.rollout, self.networks, self.episode_num, weights=WEIGHTS
                )
                self.episode_reward = 0
                break
        with torch.no_grad():
            final_value = self.networks.get_value(
                state=self.rollout.get_state(step + 1)
            )
        return final_value, env, finished, episode_entropy

    def collect_episode(self, env: gym.Env) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.episode_reward = 0
        episode_entropy = 0
        while not (done or truncated):
            state = self.rollout.get_state(t)
            action, log_prob, entropy = self.networks.select_action(state)
            value = self.networks.get_value(state=state)
            obs, reward, done, truncated, _ = env.step(action)
            self.episode_reward += reward
            episode_entropy += entropy
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
            self.t_global += 1
        self.episode_num += 1
        self.logger.log({"Reward/Episode": self.episode_reward}, self.episode_num)
        self.logger.log_histograms(
            self.rollout, self.networks, self.episode_num, weights=WEIGHTS
        )
        with torch.no_grad():
            final_value = self.networks.get_value(state=self.rollout.get_state(t))
        return final_value, episode_entropy

    def update_policy(self, final_value: torch.Tensor, entropy: torch.Tensor):
        """
        Update the agent given the A2C update rule and update the \
            relevant other parameters

        Args:
            final_value (torch.Tensor): The estimation of the value, by the \
                critic, of the final state of the rollout.
        """
        actor_loss, critic_loss, advantages, targets = self.networks.update(
            self.rollout, final_value, entropy
        )
        self.logger.log(
            {"Loss/Actor": actor_loss, "Loss/Critic": critic_loss}, self.t_global
        )
        if advantages is not None:
            self.logger.add_histogram(
                "Histograms/Advantages", advantages, self.episode_num
            )
            self.logger.add_histogram(
                "Histograms/Value targets", targets, self.episode_num
            )

    def _train_n_step(self, env: gym.Env, n_updates: PositiveInt):
        """
        Train the agent for n_updates in an n-step fashion.

        Args:
            env (gym.Env): The env to use for training.
            n_updates (int): How much updates to do.
        """
        obs, _ = env.reset()
        self.rollout.internals.states[0].copy_(self.rollout.obs2tensor(obs))
        for _ in tqdm(range(n_updates)):
            final_value, env, terminated, entropy = self.collect_rollout(
                env, self.rollout.config.buffer_size
            )
            self.update_policy(final_value, entropy)
            self.rollout.after_update()
            if terminated:
                obs, _ = env.reset()
                self.rollout.reset()
                self.rollout.internals.states[0].copy_(self.rollout.obs2tensor(obs))

        env.close()

    def _train_MC(self, env: gym.Env, n_episodes: int):
        """
        Train the agent in an MC fashion.

        Args:
            env (gym.Env): The environment to use fo training.
            n_episodes (int): The number of episodes to train the agent for.

        """
        for _ in tqdm(range(n_episodes)):
            final_value, episode_entropy = self.collect_episode(env)
            self.update_policy(final_value, episode_entropy)
            self.rollout.reset()
        env.close()

    def train(
        self,
        env: gym.Env,
        n_iter: int,
    ):
        """
        Wrapper for training the agent according to the relevant method.

        Args:
            env (gym.Env): The environment to train the agent on.
            n_iter (int): the number of training units to train agent for :
                - episodes for "MC" setting
                - updates for "n-step" stting
        """
        if self.rollout.config.setting == "MC":
            self._train_MC(env, n_episodes=n_iter)
        elif self.rollout.config.setting == "n-step":
            self._train_n_step(env, n_updates=n_iter)
        else:
            raise ValueError(
                f"Urecognized buffer setting : {self.rollout.config.setting}"
            )

    def test(self, env: gym.Env, n_episodes: int, render: bool = False) -> dict:
        """
        Test/Evaluate the agent given its current state.

        Args:
            env (gym.Env): The environment to test the agent on.
            n_episodes (int): The number of episodes to test the agent.
            render (bool, optional): Wether or not to render the environment \
                while testing. Defaults to False.

        Returns:
            dict: Testing report for each episode.
        """
        cumulative_reward_per_episode = {}
        for episode in tqdm(range(n_episodes)):
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
        env.close()
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
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
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
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample().detach()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob, dist.entropy().mean()

    def A2C_loss(
        self, log_prob: torch.Tensor, advantage: torch.Tensor, entropy: torch.Tensor
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
        entropy_loss = self.config.entropy_coeff * entropy
        policy_loss = (-log_prob * advantage.detach()).mean() - entropy_loss
        value_loss = advantage.pow(2).mean()
        return policy_loss, value_loss

    def update(
        self, buffer: RolloutBuffer, final_value: torch.Tensor, entropy: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]:
        """
        Update the nework's weights according to A2C rule.

        Args:
            advantages (torch.Tensor): advantage to consider
            log_prob (torch.Tensor): log_prob to consider
        """

        # For logging purposes
        torch.autograd.set_detect_anomaly(True)
        self.index += 1
        n_steps = 0 if buffer.config.setting == "MC" else buffer.config.n_steps - 1
        advantages, target = buffer.compute_advantages(final_value)
        if advantages is not None:
            actor_loss, critic_loss = self.A2C_loss(
                buffer.internals.log_probs[: buffer.internals.len - n_steps],
                advantages[: buffer.internals.len],
                entropy,
            )
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=False)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=False)
            self.critic_optimizer.step()
            return actor_loss, critic_loss, advantages, target
        return None, None, None, None

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
        return torch.squeeze(self.critic(state))

    def save(self, folder: Path, name: str = "model") -> None:
        """
        Save the current model

        Args:
            name (str, optional): [Name of the model]. Defaults to "model".
        """
        print(f"Saving at {folder}/{name}_XXXX.pth")
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
