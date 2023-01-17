"""
Utils for the agents
"""

import gym


def get_action_shape(env: gym.Env) -> int:
    """
    Extract the action shape from the gym env

    Args:
        env (gym.Env): The env for which to get the action shape

    Returns:
        int: The action dimension
    """
    return (
        env.action_space.n
        if "n" in dir(env.action_space)
        else env.action_space.shape[0]
    )


def get_obs_shape(env: gym.Env) -> int:
    """
    Extract the action shape from the gym env

    Args:
        env (gym.Env): The env for which to get the action shape

    Returns:
        int: The action dimension
    """
    return (
        env.observation_space.n
        if "n" in dir(env.observation_space)
        else env.observation_space.shape[0]
    )
