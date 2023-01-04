import os
import shutil

import gym
import numpy as np
import torch

from bombproofbasis.agents.A2C import A2C, A2CNetworks
from bombproofbasis.agents.utils import get_action_shape
from bombproofbasis.types import A2CConfig, BufferConfig, NetworkConfig, ScalerConfig
from bombproofbasis.utils.buffer import RolloutBuffer

from .test_buffer import fill_buffer

ENV = gym.make("CartPole-v1")
action_shape = get_action_shape(ENV)

actor_architecture = ["64", "tanh", "32", "tanh"]
critic_architecture = ["64", "relu", "32", "relu"]
actor_recurrent_architecture = ["64", "relu", "LSTM(32*1)", "relu"]
critic_recurrent_architecture = ["64", "tanh", "LSTM(32*1)", "tanh"]

policy_network_config = NetworkConfig(
    architecture=actor_architecture,
    input_shape=ENV.observation_space.shape[0],
    output_shape=action_shape,
    actor=True,
)
value_network_config = NetworkConfig(
    architecture=critic_architecture,
    input_shape=ENV.observation_space.shape[0],
    output_shape=1,
    actor=False,
)
buffer_config = BufferConfig(obs_shape=ENV.observation_space.shape, setting="MC")
scaler_config = ScalerConfig(scale=True)
A2C_CONFIG = A2CConfig(
    environment=ENV,
    agent_type="A2C",
    policy_network=policy_network_config,
    value_network=value_network_config,
    scaler=scaler_config,
    buffer=buffer_config,
)


# def test_A2CNetworks():
#     networks = A2CNetworks(A2C_CONFIG)
#     buffer = RolloutBuffer(networks.config.buffer)
#     buffer = fill_buffer(buffer, done=True)
#     obs, info = ENV.reset()
#     action, log_prob = networks.select_action(observation=buffer.obs2tensor(obs))
#     assert isinstance(action, int)
#     # log_prob = vals

#     action_probas = networks.get_action_probabilities(buffer.obs2tensor(obs))
#     for proba in action_probas:
#         assert 0 <= proba <= 1
#     value = networks.get_value(state=buffer.obs2tensor(obs))

#     # Test saving
#     model_folder = "./models"
#     os.makedirs(model_folder, exist_ok=True)
#     networks.save(folder=model_folder)
#     del networks
#     new_networks = A2CNetworks(A2C_CONFIG)
#     new_networks.load(folder=model_folder)
#     shutil.rmtree(model_folder)
#     assert torch.equal(new_networks.get_value(state=buffer.obs2tensor(obs)), value)
#     assert (
#         action_probas.all()
#         == new_networks.get_action_probabilities(buffer.obs2tensor(obs)).all()
#     )


def check_difference_in_policy(networks, obs, old_action_proba):
    return not np.array_equal(
        old_action_proba,
        networks.get_action_probabilities(obs),
    )


def check_difference_in_value(networks, obs, old_value):
    return not torch.equal(
        networks.get_value(state=obs),
        old_value,
    )


def test_A2C_basics():
    agent = A2C(A2C_CONFIG)
    obs, info = ENV.reset()
    agent.rollout.internals.states[0].copy_(agent.rollout.obs2tensor(obs))
    old_value = agent.networks.get_value(state=agent.rollout.get_state(0))
    old_action_probas = agent.networks.get_action_probabilities(
        agent.rollout.get_state(0)
    )
    final_value, entropy = agent.collect_episode(ENV)
    assert agent.rollout.full
    agent.update_policy(final_value, entropy)
    agent.rollout.reset()
    assert agent.rollout.internals.len == 0
    assert check_difference_in_policy(
        agent.networks, agent.rollout.get_state(0), old_action_probas
    )
    assert check_difference_in_value(
        agent.networks,
        agent.rollout.get_state(0),
        old_value,
    )

    # Test saving
    model_folder = "./models"
    os.makedirs(model_folder, exist_ok=True)
    agent.save(folder=model_folder)
    del agent
    agent = A2C(A2C_CONFIG)
    agent.load(folder=model_folder)
    shutil.rmtree(model_folder)
    agent.get_value(obs)


def test_A2C_train_test():
    # MC
    MC_agent = A2C(A2C_CONFIG)
    n_episodes = 3
    MC_agent.train(ENV, n_iter=n_episodes)
    MC_agent.test(ENV, n_episodes=n_episodes)
    # TD (1-step)
    buffer_config = BufferConfig(
        obs_shape=ENV.observation_space.shape, setting="n-step", n_steps=1
    )
    n_step_A2C_CONFIG = A2CConfig(
        environment=ENV,
        agent_type="A2C",
        policy_network=policy_network_config,
        value_network=value_network_config,
        scaler=scaler_config,
        buffer=buffer_config,
    )
    n_step_agent = A2C(n_step_A2C_CONFIG)
    n_episodes = 100
    n_step_agent.train(ENV, n_iter=n_episodes)
    n_step_agent.test(ENV, n_episodes=10, render=True)
    # 2-step
    buffer_config = BufferConfig(
        obs_shape=ENV.observation_space.shape, setting="n-step", n_steps=2
    )
    n_step_A2C_CONFIG = A2CConfig(
        environment=ENV,
        agent_type="A2C",
        policy_network=policy_network_config,
        value_network=value_network_config,
        scaler=scaler_config,
        buffer=buffer_config,
    )
    n_step_agent = A2C(n_step_A2C_CONFIG)
    n_episodes = 100
    n_step_agent.train(ENV, n_iter=n_episodes)
    n_step_agent.test(ENV, n_episodes=10, render=False)
