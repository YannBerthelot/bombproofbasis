import gym
import pytest

from bombproofbasis.types import AgentConfig, NetworkConfig, TrainingConfig


def test_config():
    network_config = NetworkConfig(
        actor_learning_rate=1e-3,
        actor_architecture=["32", "relu", "64", "relu", "LSTM(128*3)"],
        critic_learning_rate=1e-3,
        critic_architecture=["32", "relu", "64", "relu", "LSTM(128*3)"],
    )
    with pytest.raises(ValueError):
        faulty_network_config = NetworkConfig(
            actor_learning_rate=1e-3,
            actor_architecture=["32", "relu", "64", "relu", "LSTM(128*3)"],
            critic_learning_rate=1e-3,
            critic_architecture=["32", "relu", "64", "relu", "LSTM(128*3)"],
            hardware="TPU",
        )
        faulty_network_config
    agent_config = AgentConfig(
        learning_rate=1e-3,
        environment=gym.make("CartPole-v1"),
        agent_type="A2C",
        continous=False,
        network=network_config,
    )

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
