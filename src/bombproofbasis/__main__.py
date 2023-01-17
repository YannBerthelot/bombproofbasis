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
buffer_TD_config = BufferConfig(
    obs_shape=obs_shape,
    buffer_size=1,
    setting="n-step",
    n_steps=1,
)
buffer_MC_config = BufferConfig(obs_shape=obs_shape, setting="MC")
buffer_n_step_config = BufferConfig(
    obs_shape=obs_shape, setting="n-step", n_steps=2, buffer_size=3
)
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
    buffer=buffer_TD_config,
    entropy_coeff=0.00,
)
A2C_2_step_CONFIG = A2CConfig(
    environment=ENV,
    agent_type="A2C",
    policy_network=policy_network_config,
    value_network=value_network_config,
    scaler=scaler_config,
    buffer=buffer_n_step_config,
    entropy_coeff=0.00,
)

if __name__ == "__main__":
    for i in range(10):
        agent = A2C(A2C_MC_CONFIG, LoggingConfig(run_name=f"MC {i}"))
        print("ACTOR", agent.networks.actor)
        print("CRITIC", agent.networks.critic)
        agent.train(ENV, n_iter=300)
        agent.networks.load(folder=Path("./models"), name="best")
        agent.test(ENV, n_episodes=10, render=False)

        n_steps = 1
        buffer_size = 1
        agent = A2C(
            A2C_TD_CONFIG,
            LoggingConfig(run_name=f"n-step {i} {n_steps=} {buffer_size=}"),
        )
        print("ACTOR", agent.networks.actor)
        print("CRITIC", agent.networks.critic)
        agent.train(ENV, n_iter=25000)
        agent.networks.load(folder=Path("./models"), name="best")
        agent.test(ENV, n_episodes=10, render=False)

        n_steps = 2
        buffer_size = 3
        agent = A2C(
            A2C_TD_CONFIG,
            LoggingConfig(run_name=f"n-step {i} {n_steps=} {buffer_size=}"),
        )
        print("ACTOR", agent.networks.actor)
        print("CRITIC", agent.networks.critic)
        agent.train(ENV, n_iter=int(25000 / 2))
        agent.networks.load(folder=Path("./models"), name="best")
        agent.test(ENV, n_episodes=10, render=False)
