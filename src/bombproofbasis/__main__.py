"""
Example of agent functions on CartPole-v1
"""
import gym

from bombproofbasis.agents.A2C import A2C
from bombproofbasis.agents.utils import get_action_shape
from bombproofbasis.types import (
    A2CConfig,
    BufferConfig,
    LoggingConfig,
    NetworkConfig,
    ScalerConfig,
)

ENV = gym.make("CartPole-v1")
action_shape = get_action_shape(ENV)
critic_architecture = ["64", "relu", "32", "relu"]
actor_architecture = ["64", "tanh", "32", "tanh"]
# recurrent_architecture = ["4", "relu", "16", "relu", "LSTM(8*1)"]
policy_network_config = NetworkConfig(
    learning_rate=1e-3,
    architecture=actor_architecture,
    input_shape=ENV.observation_space.shape[0],
    output_shape=action_shape,
    actor=True,
)
value_network_config = NetworkConfig(
    learning_rate=1e-3,
    architecture=critic_architecture,
    input_shape=ENV.observation_space.shape[0],
    output_shape=1,
    actor=False,
)
n_steps = 2
buffer_size = 1 * 2 * n_steps + 1
buffer_config = BufferConfig(
    obs_shape=ENV.observation_space.shape,
    buffer_size=buffer_size,
    setting="n-step",
    n_steps=n_steps,
)
buffer_MC_config = BufferConfig(obs_shape=ENV.observation_space.shape, setting="MC")
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

logging_config = LoggingConfig()

if __name__ == "__main__":
    # agent = A2C(A2C_MC_CONFIG)
    # print("ACTOR", agent.networks.actor)
    # print("CRITIC", agent.networks.critic)
    # train_report = agent.train(ENV, setting="MC", n_episodes=1000)
    # test_report = agent.test(ENV, n_episodes=10)
    # with open("train_report.json", "w") as outfile:
    #     json.dump(train_report, outfile)
    # with open("test_report.json", "w") as outfile:
    #     json.dump(test_report, outfile)
    agent = A2C(A2C_TD_CONFIG, logging_config)
    print("ACTOR", agent.networks.actor)
    print("CRITIC", agent.networks.critic)
    agent.train(ENV, n_iter=int(20000 / buffer_size))
    agent.test(ENV, n_episodes=10, render=True)
