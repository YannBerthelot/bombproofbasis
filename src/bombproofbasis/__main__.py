"""
Example of agent functions on CartPole-v1
"""
from pathlib import Path

import gym

from bombproofbasis.agents.A2C import A2C
from bombproofbasis.agents.utils import get_action_shape, get_obs_shape
from bombproofbasis.types import (
    A2CConfig,
    BufferConfig,
    LoggingConfig,
    NetworkConfig,
    ScalerConfig,
)

ENV = gym.make("CartPole-v1", render_mode="rgb_array")
action_shape = get_action_shape(ENV)
obs_shape = get_obs_shape(ENV)
critic_architecture = ["64", "relu", "32", "relu"]
actor_architecture = ["64", "tanh", "32", "tanh"]
# recurrent_architecture = ["4", "relu", "16", "relu", "LSTM(8*1)"]
policy_network_config = NetworkConfig(
    learning_rate=1e-3,
    architecture=actor_architecture,
    input_shape=obs_shape,
    output_shape=action_shape,
    actor=True,
)
value_network_config = NetworkConfig(
    learning_rate=1e-3,
    architecture=critic_architecture,
    input_shape=obs_shape,
    output_shape=1,
    actor=False,
)
n_steps = 1
buffer_size = 5
buffer_config = BufferConfig(
    obs_shape=obs_shape,
    buffer_size=buffer_size,
    setting="n-step",
    n_steps=n_steps,
)
buffer_MC_config = BufferConfig(obs_shape=obs_shape, setting="MC")
scaler_config = ScalerConfig(scale=True)
A2C_MC_CONFIG = A2CConfig(
    environment=ENV,
    agent_type="A2C",
    policy_network=policy_network_config,
    value_network=value_network_config,
    scaler=scaler_config,
    buffer=buffer_MC_config,
    entropy_coeff=0.0,
)
A2C_TD_CONFIG = A2CConfig(
    environment=ENV,
    agent_type="A2C",
    policy_network=policy_network_config,
    value_network=value_network_config,
    scaler=scaler_config,
    buffer=buffer_config,
    entropy_coeff=0.0,
)

if __name__ == "__main__":

    # agent = A2C(A2C_MC_CONFIG, LoggingConfig(run_name="MC"))
    # print("ACTOR", agent.networks.actor)
    # print("CRITIC", agent.networks.critic)
    # agent.train(ENV, n_iter=500)
    # agent.networks.load(folder="./models", name="best")
    # agent.test(ENV, n_episodes=10, render=True)

    agent = A2C(
        A2C_TD_CONFIG, LoggingConfig(run_name=f"n-step {n_steps=} {buffer_size=}")
    )
    print("ACTOR", agent.networks.actor)
    print("CRITIC", agent.networks.critic)
    agent.train(ENV, n_iter=25000 // buffer_size)
    agent.networks.load(folder=Path("./models"), name="best")
    agent.test(ENV, n_episodes=10, render=True)
