import os
import shutil
from typing import Dict, Tuple
from warnings import filterwarnings

import gym
import numpy as np
import pytest
import torch

from bombproofbasis.agents.A2C import A2CNetworks
from bombproofbasis.agents.utils import get_action_shape
from bombproofbasis.types import A2CConfig, BufferConfig, NetworkConfig, ScalerConfig


def test_A2CNetworks():
    env = gym.make("CartPole-v1")
    action_shape = get_action_shape(env)
    policy_network_config = NetworkConfig(
        learning_rate=1e-3,
        architecture=["4", "relu", "16", "relu", "LSTM(8*1)"],
        input_shape=env.observation_space.shape[0],
        output_shape=action_shape,
        actor=True,
    )
    value_network_config = NetworkConfig(
        learning_rate=1e-3,
        architecture=["4", "relu", "16", "relu", "LSTM(8*1)"],
        input_shape=env.observation_space.shape[0],
        output_shape=1,
        actor=False,
    )
    buffer_config = BufferConfig(setting="MC", gamma=0.99, buffer_size=5, n_steps=2)
    scaler_config = ScalerConfig(scale=True)
    A2C_config = A2CConfig(
        learning_rate=1e-3,
        environment=env,
        agent_type="A2C",
        continous=False,
        policy_network=policy_network_config,
        value_network=value_network_config,
        scaler=scaler_config,
        buffer=buffer_config,
    )

    networks = A2CNetworks(A2C_config)
    obs, info = env.reset()

    action, new_actor_hiddens, vals = networks.select_action(
        observation=obs, hiddens=networks.actor.hiddens
    )
    assert isinstance(action, int)
    assert isinstance(new_actor_hiddens, dict)
    assert not (torch.equal(new_actor_hiddens[4][0], networks.actor.hiddens[4][0]))
    log_prob, entropy, KL_div = vals

    action_probas = networks.get_action_probabilities(obs)
    for proba in action_probas:
        assert 0 <= proba <= 1
    hiddens = networks.critic.hiddens
    value, new_critic_hiddens = networks.get_value(state=obs, hiddens=hiddens)
    gamma = 0.99
    next_obs, reward, done, truncated, info = env.step(action=env.action_space.sample())
    next_val, _ = networks.get_value(state=next_obs, hiddens=new_critic_hiddens)
    advantage = torch.add(reward, torch.mul(gamma, torch.sub(next_val, value)))
    networks.update_policy(
        advantages=advantage, log_prob=log_prob
    )  # , entropy=entropy)
    value, new_critic_hiddens = networks.get_value(state=obs, hiddens=hiddens)
    # Assert update
    assert not (
        torch.equal(networks.get_value(state=obs, hiddens=new_critic_hiddens)[0], value)
    )
    networks.actor.hiddens = new_actor_hiddens
    print(action_probas, networks.get_action_probabilities(obs))
    assert not np.array_equal(action_probas, networks.get_action_probabilities(obs))
    # Test saving
    model_folder = "./models"
    os.makedirs(model_folder, exist_ok=True)
    networks.save(folder=model_folder)
    del networks
    new_networks = A2CNetworks(A2C_config)
    new_networks.load(folder=model_folder)
    shutil.rmtree(model_folder)
    assert torch.equal(new_networks.get_value(state=obs, hiddens=hiddens)[0], value)
    assert action_probas.all() == new_networks.get_action_probabilities(obs).all()
