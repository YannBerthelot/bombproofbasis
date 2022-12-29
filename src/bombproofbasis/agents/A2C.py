"""
A2C agent
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# Base class for Agent
# from bombproofbasis.agents.agent import Agent
from bombproofbasis.network.network import BaseTorchNetwork
from bombproofbasis.network.utils import compute_KL_divergence, t
from bombproofbasis.types import A2CConfig, LoggingConfig

# from bombproofbasis.utils.buffer import RolloutBuffer

# # Network creator tool
# from bombproofbasis.utils.normalize import SimpleStandardizer

# class A2C(Agent):
#     def __init__(self, config: A2CConfig) -> None:
#         super(
#             A2C,
#             self,
#         ).__init__(config)

#         self.rollout = RolloutBuffer(
#             buffer_size=config.buffer.buffer_size,
#             gamma=config.buffer.gamma,
#             n_steps=config.buffer.n_steps,
#         )

#         # Initialize the policy network with the right shape
#         self.network = TorchA2C(self)
#         self.actor_hidden = self.network.actor.initialize_hidden_states()
#         self.critic_hidden = self.network.critic.initialize_hidden_states()
#         self.t, self.t_global = 1, 1

#     def select_action(
#         self,
#         observation: np.array,
#         hidden: Dict[int, torch.Tensor],
#     ) -> int:
#         """
#         Select the action based on the current policy and the observation

#         Args:
#             observation (np.array): State representation
#             testing (bool): Wether to be in test mode or not.

#         Returns:
#             int: The selected action
#         """
#         return self.network.select_action(observation, hidden)

#     def compute_value(self, observation: np.array, hidden: np.array) -> int:
#         """
#         Select the action based on the current policy and the observation

#         Args:
#             observation (np.array): State representation
#             testing (bool): Wether to be in test mode or not.

#         Returns:
#             int: The selected action
#         """
#         return self.network.get_value(observation, hidden)

#     def pre_train(self, env: gym.Env, nb_timestep: int, scaling=False) -> None:
#         # Init training
#         t_old, self.constant_reward_counter = 0, 0
#         actor_hidden, critic_hidden = self.actor_hidden, self.critic_hidden
#         # Pre-Training
#         if nb_timestep > 0:
#             print("--- Pre-Training ---")
#             t_pre_train = 1
#             pbar = tqdm(total=nb_timestep, initial=1)
#             while t_pre_train <= nb_timestep:
#                 pbar.update(t_pre_train - t_old)
#                 t_old = t_pre_train
#                 done, obs, rewards = False, env.reset(), []
#                 while not done:
#                     action, actor_hidden, loss_params = self.select_action(
#                         obs, actor_hidden
#                     )
#                     value, critic_hidden = self.network.get_value(obs, critic_hidden)
#                     action = self.env.action_space.sample()
#                     next_obs, reward, done, _ = env.step(action)
#                     if scaling:
#                         next_obs, reward = self.scaling(
#                             next_obs, reward, fit=True, transform=False
#                         )
#                     t_pre_train += 1
#             pbar.close()
#             print(
#                 f"Obs scaler - Mean : {self.obs_scaler.mean}, std : {self.obs_scaler.std}"
#             )
#             print(f"Reward scaler - std : {self.reward_scaler.std}")
#             self.obs_scaler.save(path="scalers", name="obs")
#             self.save_scalers("scalers", "scaler")
#         return actor_hidden, critic_hidden

#     def save_scalers(self, path, name) -> None:
#         self.obs_scaler.save(path=path, name="obs_" + name)
#         self.reward_scaler.save(path=path, name="reward_" + name)

#     def load_scalers(self, path: str, name: str) -> None:
#         self.obs_scaler, self.reward_scaler, self.target_scaler = self.get_scalers(True)
#         self.obs_scaler.load(path, "obs_" + name)
#         self.reward_scaler.load(path, "reward_" + name)

#     def train_MC(self, env: gym.Env, nb_timestep: int) -> None:
#         # actor_hidden, critic_hidden = self.pre_train(
#         #     env, self.config["GLOBAL"].getfloat("learning_start")
#         # )
#         actor_hidden, critic_hidden = self.actor_hidden, self.critic_hidden
#         self.rollout = RolloutBuffer(
#             buffer_size=self.env._max_episode_steps,
#             gamma=self.config["AGENT"].getfloat("gamma"),
#             n_steps=1,
#         )
#         self.t, self.constant_reward_counter, self.old_reward_sum = 1, 0, 0
#         print("--- Training ---")
#         t_old = 0
#         pbar = tqdm(total=nb_timestep, initial=1)
#         scaling = self.config["GLOBAL"].getboolean("scaling")
#         while self.t <= nb_timestep:
#             # tqdm stuff
#             pbar.update(self.t - t_old)
#             t_old, t_episode = self.t, 1

#             # actual episode
#             actions_taken = {action: 0 for action in range(self.action_shape)}
#             done, obs, rewards = False, env.reset(), []

#             reward_sum = 0
#             while not done:
#                 (action, next_actor_hidden, loss_params) = self.select_action(
#                     obs, actor_hidden
#                 )
#                 (
#                     log_prob,
#                     entropy,
#                     KL_divergence,
#                 ) = loss_params
#                 value, next_critic_hidden = self.network.get_value(obs, critic_hidden)
#                 next_obs, reward, done, _ = env.step(action)
#                 reward_sum += reward

#                 actions_taken[int(action)] += 1
#                 self.rollout.add(reward, done, value, log_prob, entropy, KL_divergence)

#                 self.t_global, self.t, t_episode = (
#                     self.t_global + 1,
#                     self.t + 1,
#                     t_episode + 1,
#                 )
#                 if scaling:
#                     next_obs, reward = self.scaling(
#                         next_obs, reward, fit=False, transform=True
#                     )
#                 obs = next_obs
#                 critic_hidden, actor_hidden = next_critic_hidden, next_actor_hidden

#             self.rollout.update_advantages(MC=True)
#             advantages = self.rollout.advantages
#             # for i in range(t_episode - 1):
#             #     loss_params_episode = (
#             #         self.rollout.log_probs[i],
#             #         self.rollout.entropies[i],
#             #         self.rollout.KL_divergences[i],
#             #     )
#             #     self.network.update_policy(
#             #         advantages[i], *loss_params_episode, finished=i == t_episode - 2
#             #     )
#             loss_params_episode = (
#                 self.rollout.log_probs,
#                 self.rollout.entropies,
#                 self.rollout.KL_divergences,
#             )
#             self.network.update_policy(advantages, *loss_params_episode, finished=True)
#             self.rollout.reset()
#             self.save_if_best(reward_sum)
#             if self.early_stopping(reward_sum):
#                 break

#             self.old_reward_sum, self.episode = reward_sum, self.episode + 1
#             self.episode_logging(reward_sum, actions_taken)

#         pbar.close()
#         self.train_logging(self.artifact)

#     def train_TD0(self, env: gym.Env, nb_timestep: int) -> None:
#         # actor_hidden, critic_hidden = self.pre_train(
#         #     env, self.config["GLOBAL"].getfloat("learning_start")
#         # )
#         actor_hidden, critic_hidden = self.actor_hidden, self.critic_hidden
#         self.constant_reward_counter, self.old_reward_sum = 0, 0
#         print("--- Training ---")
#         t_old = 0
#         pbar = tqdm(total=nb_timestep, initial=1)

#         while self.t <= nb_timestep:
#             # tqdm stuff
#             pbar.update(self.t - t_old)
#             t_old, t_episode = self.t, 1

#             # actual episode
#             actions_taken = {action: 0 for action in range(self.action_shape)}
#             done, obs, rewards = False, env.reset(), []

#             reward_sum = 0
#             while not done:
#                 action, next_actor_hidden, loss_params = self.select_action(
#                     obs, actor_hidden
#                 )
#                 value, critic_hidden = self.network.get_value(obs, critic_hidden)
#                 next_obs, reward, done, _ = env.step(action)
#                 reward_sum += reward
#                 if self.config["GLOBAL"].getboolean("scaling"):
#                     next_obs, reward = self.scaling(
#                         next_obs, reward, fit=False, transform=True
#                     )
#                 next_critic_hidden = critic_hidden.copy()
#                 next_value, next_next_critic_hidden = self.network.get_value(
#                     next_obs, critic_hidden
#                 )

#                 advantage = reward + next_value - value
#                 actions_taken[int(action)] += 1

#                 self.network.update_policy(advantage, *loss_params, finished=True)
#                 self.t_global, self.t, t_episode = (
#                     self.t_global + 1,
#                     self.t + 1,
#                     t_episode + 1,
#                 )
#                 obs = next_obs
#                 next_value, next_next_critic_hidden = self.network.get_value(
#                     next_obs, next_critic_hidden
#                 )
#                 critic_hidden, actor_hidden = next_next_critic_hidden, next_actor_hidden

#             self.save_if_best(reward_sum)
#             if self.early_stopping(reward_sum):
#                 break

#             self.old_reward_sum, self.episode = reward_sum, self.episode + 1
#             self.episode_logging(reward_sum, actions_taken)

#         pbar.close()
#         self.train_logging(self.artifact)

#     def test(
#         self, env: gym.Env, nb_episodes: int, render: bool = False, scaler_file=None
#     ) -> None:
#         """
#         Test the current policy to evalute its performance

#         Args:
#             env (gym.Env): The Gym environment to test it on
#             nb_episodes (int): Number of test episodes
#             render (bool, optional): Wether or not to render the visuals of \
#               the episodes while testing. Defaults to False.
#         """
#         print("--- Testing ---")
#         if scaler_file is not None and self.obs_scaler is not None:
#             with open(scaler_file, "rb") as input_file:
#                 scaler = pickle.load(input_file)
#             self.obs_scaler = scaler
#         episode_rewards = []
#         best_test_episode_reward = 0
#         # Iterate over the episodes
#         for episode in tqdm(range(nb_episodes)):
#             actor_hidden = self.network.actor.initialize_hidden_states()
#             # Init episode
#             done, obs, rewards_sum = False, env.reset(), 0

#             # Generate episode
#             while not done:
#                 # Select the action using the current policy
#                 if self.config["GLOBAL"].getboolean("scaling"):
#                     obs = self.obs_scaler.transform(obs)
#                 action, next_actor_hidden, _ = self.select_action(obs, actor_hidden)

#                 # Step the environment accordingly
#                 next_obs, reward, done, _ = env.step(action)

#                 # Log reward for performance tracking
#                 rewards_sum += reward

#                 # render the environment
#                 if render:
#                     env.render()

#                 # Next step
#                 obs, actor_hidden = next_obs, next_actor_hidden

#             if rewards_sum > best_test_episode_reward:
#                 best_test_episode_reward = rewards_sum
#                 if self.config["GLOBAL"]["logging"] == "wandb":
#                     wandb.run.summary["Test/best reward sum"] = rewards_sum
#             # Logging
#             if self.config["GLOBAL"]["logging"] == "wandb":
#                 wandb.log(
#                     {"Test/reward": rewards_sum, "Test/episode": episode}, commit=True
#                 )
#             elif self.config["GLOBAL"]["logging"] == "tensorboard":
#                 self.network.writer.add_scalar("Reward/test", rewards_sum, episode)
#             # print(f"test number {episode} : {rewards_sum}")
#             episode_rewards.append(rewards_sum)
#         env.close()
#         if self.config["GLOBAL"]["logging"] == "tensorboard":
#             self.network.writer.add_hparams(
#                 self.config,
#                 {
#                     "test mean reward": np.mean(episode_rewards),
#                     "test std reward": np.std(episode_rewards),
#                     "test max reward": max(episode_rewards),
#                     "min test reward": min(episode_rewards),
#                 },
#                 run_name="test",
#             )

#     def _learn(self) -> None:
#         for i, steps in enumerate(self.rollout.get_steps_list()):
#             advantage, log_prob, entropy, kl_divergence = steps
#             self.network.update_policy(
#                 advantage,
#                 log_prob,
#                 entropy,
#                 kl_divergence,
#                 finished=True,
#             )


class Logger:
    """
    Logger for A2C agent. Used to handle the logging into wandb for the \
        agent's training.
    """

    def __init__(self, config: LoggingConfig) -> None:
        """Create the logger based on the config

        Args:
            config (LoggingConfig): Logger config
        """
        self.config = config

    def log(self, timestep: int) -> None:
        """
        Logs the relevant values into wandb

        Args:
            timestep (int): The current timestep
        """

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

    def select_action(
        self,
        observation: np.ndarray,
        hiddens: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[
        int,
        Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, float],
    ]:
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
        probs, new_hiddens = self.actor.forward(
            x=t(observation).unsqueeze(0), hiddens=hiddens
        )
        dist = torch.distributions.Categorical(probs=probs)  # gradient needed
        action = dist.sample().detach()
        log_prob = dist.log_prob(action)  # gradient needed
        entropy = dist.entropy()  # gradient needed

        KL_divergence = (
            compute_KL_divergence(self.old_dist, dist)
            if self.old_dist is not None
            else 0
        )  # gradient needed

        return (
            int(action.item()),
            new_hiddens,
            (log_prob, entropy, KL_divergence),
        )

    def update_policy(
        self,
        advantages: torch.Tensor,
        log_prob: torch.Tensor,
        # entropy: torch.Tensor,
        logger: Logger,
    ) -> None:
        """
        Update the policy's parameters according to \
        the n-step A2C updates rules.

        Args:
            state (np.array): Observation of the state
            action (np.array): The selected action
            n_step_return (np.array): The n-step return
            next_state (np.array): The state n-step after state
            done (bool, optional): Wether the episode if finished or not at \
            next-state. Used to handle 1-step. Defaults to False.
        """

        # For logging purposes
        torch.autograd.set_detect_anomaly(True)

        self.index += 1
        # Losses (be careful that all its components are torch tensors with grad on)
        # entropy_loss = -entropy
        actor_loss = -(torch.mul(log_prob, advantages))
        critic_loss = advantages.pow(2)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)

        # Logging
        logger.log(self.index)

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
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
        return (
            self.actor.forward(x=t(state).unsqueeze(0), hiddens=self.actor.hiddens)[0]
            .detach()
            .cpu()
            .numpy()[0]
        )

    def get_value(
        self, state: np.ndarray, hiddens: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
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
        value, new_hidden = self.critic.forward(t(state).unsqueeze(0), hiddens)
        return value, new_hidden

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
