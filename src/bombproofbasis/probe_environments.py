import random

import gym
import numpy as np
from gym import spaces


class ProbeEnv1(gym.Env):
    """
    Probe Env 1:
    One action, zero observation, one timestep long, +1 reward every timestep:\
    This isolates the value network. If my agent can't learn that the value of\
    the only observation it ever sees it 1, there's a problem with the value \
        loss calculation or the optimizer.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(1)
        # Example for using image as input:
        self.observation_space = spaces.Discrete(1)

    def step(self, action):
        return (np.array([0]), 1, True, False, None)

    def reset(self):
        # Reset the state of the environment to an initial state
        return np.array([0]), None

    def render(self):
        pass


def get_random_obs():
    return random.choice([-1, 1])


class ProbeEnv2(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(1)
        # Example for using image as input:
        self.observation_space = spaces.Discrete(1)
        self.random_obs = None

    def step(self, action):
        return (np.array([get_random_obs()]), self.random_obs, True, False, None)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.random_obs = get_random_obs()
        return np.array([self.random_obs]), None

    def render(self):
        pass


class ProbeEnv3(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(1)
        # Example for using image as input:
        self.observation_space = spaces.Discrete(1)
        self.t = 0

    def step(self, action):
        self.t += 1
        return (np.array([self.t]), int(self.t == 2), self.t == 2, False, None)

    def reset(self):
        self.t = 0
        # Reset the state of the environment to an initial state
        return np.array([self.t]), None

    def render(self):
        pass


class ProbeEnv4(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2)
        # Example for using image as input:
        self.observation_space = spaces.Discrete(1)

    def step(self, action):
        return (np.array([0]), 1 if action == 0 else -1, True, False, None)

    def reset(self):
        # Reset the state of the environment to an initial state
        return np.array([0]), None

    def render(self):
        pass


class ProbeEnv5(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2)
        # Example for using image as input:
        self.observation_space = spaces.Discrete(1)

    def step(self, action):
        reward = (
            1
            if (
                (self.random_obs == -1 and action == 0)
                or (self.random_obs == 1 and action == 1)
            )
            else -1
        )
        return (np.array([self.random_obs]), reward, True, False, None)

    def reset(self):
        self.random_obs = get_random_obs()
        # Reset the state of the environment to an initial state
        return np.array([self.random_obs]), None

    def render(self):
        pass
