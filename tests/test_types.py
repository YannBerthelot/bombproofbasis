import gym
import pytest

from bombproofbasis.agents.utils import get_action_shape
from bombproofbasis.types import (
    AgentConfig,
    NetworkConfig,
    ScalerConfig,
    TrainingConfig,
)


def test_config():
    env = gym.make("CartPole-v1")
    input_shape = env.observation_space.shape[0]
    output_shape = get_action_shape(env)
    network_config = NetworkConfig(
        learning_rate=1e-3,
        architecture=["32", "relu", "64", "relu", "LSTM(128*3)"],
        input_shape=input_shape,
        output_shape=output_shape,
    )
    with pytest.raises(ValueError):
        faulty_network_config = NetworkConfig(
            learning_rate=1e-3,
            architecture=["32", "relu", "64", "relu", "LSTM(128*3)"],
            hardware="TPU",
            input_shape=input_shape,
            output_shape=output_shape,
        )
        faulty_network_config
    agent_config = AgentConfig(
        learning_rate=1e-3,
        environment=env,
        agent_type="A2C",
        continous=False,
        policy_network=network_config,
    )
    scaler_config = ScalerConfig(scale=True)
    agent_with_scaler_config = AgentConfig(
        learning_rate=1e-3,
        environment=env,
        agent_type="A2C",
        continous=False,
        policy_network=network_config,
        scaler=scaler_config,
    )
    agent_with_scaler_config
    training_config = TrainingConfig(
        agent=agent_config,
        nb_timesteps_train=int(1e3),
        nb_episodes_test=10,
        learning_start=0.1,
        logging="wandb",
        render=False,
    )

    training_config = TrainingConfig(
        agent=agent_config,
        nb_timesteps_train=int(1e3),
        nb_episodes_test=10,
        learning_start=int(1e3),
        logging="wandb",
        render=False,
    )

    training_config = TrainingConfig(
        agent=agent_config,
        nb_timesteps_train=int(1e3),
        nb_episodes_test=10,
        learning_start=1e3,
        logging="wandb",
        render=False,
    )
    training_config
    with pytest.raises(ValueError):
        faulty_training_config = TrainingConfig(
            agent=agent_config,
            nb_timesteps_train=int(1e3),
            nb_episodes_test=10,
            learning_start=1.5,
            logging="wandb",
            render=False,
        )

        faulty_training_config
