from operator import sub

import gym
import numpy as np
import pytest
import torch

from bombproofbasis.agents.utils import get_obs_shape
from bombproofbasis.types import BufferConfig
from bombproofbasis.utils.buffer import BufferStep, RolloutBuffer

REWARD_VALUE = 1.0
VALUE = 0.1
ENV = gym.make("CartPole-v1")
STEP = BufferStep(
    reward=REWARD_VALUE,
    done=False,
    obs=np.ones((4,)),
    value=torch.tensor([0.1]),
    log_prob=torch.tensor([0.1]),
    action=1,
)
GAMMA = 0.5


def fill_buffer(buffer: RolloutBuffer, done=False) -> RolloutBuffer:
    if done:
        for i in range(buffer.config.buffer_size - 1):
            buffer.add(STEP)
        buffer.add(
            BufferStep(
                reward=REWARD_VALUE,
                done=True,
                obs=np.ones((4,)),
                value=torch.tensor([VALUE]),
                log_prob=torch.tensor([0.1]),
                action=1,
            )
        )
    else:
        while not buffer.full:
            buffer.add(STEP)
    return buffer


def MC_return_safe(buffer):
    # Computation checks
    rewards = np.array([REWARD_VALUE for i in range(buffer.config.buffer_size)])
    gamma = buffer.config.gamma
    expected_return_MC_1 = (
        rewards[0]
        + rewards[1] * gamma
        + rewards[2] * (gamma**2)
        + rewards[3] * (gamma**3)
    )
    expected_return_MC_2 = rewards[0] + rewards[1] * gamma + rewards[2] * (gamma**2)
    expected_return_MC_3 = rewards[0] + rewards[1] * gamma
    expected_return_MC_4 = rewards[0]
    returns_MC = [
        expected_return_MC_1,
        expected_return_MC_2,
        expected_return_MC_3,
        expected_return_MC_4,
    ]
    return returns_MC


def test_add_buffer():

    obs, info = ENV.reset()
    obs_shape = get_obs_shape(ENV)
    # faulty configs
    with pytest.raises(ValueError):
        faulty_buffer_config = BufferConfig(
            obs_shape=obs_shape, gamma=1.99, setting="n-step"
        )
        RolloutBuffer(faulty_buffer_config)
    with pytest.raises(ValueError):
        faulty_buffer_config = BufferConfig(
            obs_shape=obs_shape, buffer_size=-1, setting="n-step"
        )
        RolloutBuffer(faulty_buffer_config)
    with pytest.raises(ValueError):
        faulty_buffer_config = BufferConfig(
            obs_shape=obs_shape, n_steps=-1, setting="n-step"
        )
        RolloutBuffer(faulty_buffer_config)
    buffer_size = 10
    buffer_config = BufferConfig(
        obs_shape=obs_shape, buffer_size=buffer_size, setting="n-step"
    )
    buffer = RolloutBuffer(buffer_config)
    for i in range(buffer_size - 1):
        buffer.add(STEP)
    assert not (buffer.full)
    buffer.add(STEP)
    assert buffer.full
    with pytest.raises(ValueError):
        buffer.add(STEP)


def test_return_computation():
    obs, info = ENV.reset()
    obs_shape = get_obs_shape(ENV)

    # TD 4-update
    buffer_config = BufferConfig(
        obs_shape=obs_shape, buffer_size=4, n_steps=1, setting="n-step"
    )
    buffer = RolloutBuffer(buffer_config)
    buffer.internals.states[0].copy_(buffer.obs2tensor(obs))
    buffer = fill_buffer(buffer, done=False)
    returns = buffer.compute_return()
    assert returns.squeeze(-1).tolist() == pytest.approx([1, 1, 1, 1])

    # MC
    buffer_config = BufferConfig(obs_shape=obs_shape, setting="MC", buffer_size=4)
    buffer = RolloutBuffer(buffer_config)
    buffer.internals.states[0].copy_(buffer.obs2tensor(obs))
    buffer = fill_buffer(buffer, done=False)
    returns = buffer.compute_return()
    assert returns.squeeze(-1).tolist() == pytest.approx(MC_return_safe(buffer))

    # 2-step 1-update
    buffer_config = BufferConfig(
        obs_shape=obs_shape, setting="n-step", n_steps=2, buffer_size=3
    )
    buffer = RolloutBuffer(buffer_config)
    buffer.internals.states[0].copy_(buffer.obs2tensor(obs))
    buffer = fill_buffer(buffer, done=False)
    returns = buffer.compute_return()
    val = REWARD_VALUE + buffer.config.gamma * REWARD_VALUE
    assert returns.squeeze(-1).tolist() == pytest.approx([val, val])
    # TODO : n-steps i-updates


def test_advantage_computation():
    obs, info = ENV.reset()
    obs_shape = get_obs_shape(ENV)

    # MC
    buffer_size = 4
    buffer_config = BufferConfig(
        obs_shape=obs_shape, setting="MC", buffer_size=buffer_size
    )
    buffer = RolloutBuffer(buffer_config)
    buffer.internals.states[0].copy_(buffer.obs2tensor(obs))
    buffer = fill_buffer(buffer, done=False)
    advantages, target = buffer.compute_advantages_MC()
    expected_return = MC_return_safe(buffer)
    expected_advantages = list(
        map(sub, expected_return, [VALUE for _ in range(buffer_size)])
    )
    assert advantages.squeeze(-1).tolist() == pytest.approx(expected_advantages)
    assert target.squeeze(-1).tolist() == pytest.approx(expected_return)

    # TD
    buffer_size = 4
    buffer_config = BufferConfig(
        obs_shape=obs_shape,
        setting="n-step",
        buffer_size=buffer_size,
        n_steps=1,
        gamma=GAMMA,
    )
    buffer = RolloutBuffer(buffer_config)
    buffer.internals.states[0].copy_(buffer.obs2tensor(obs))
    buffer = fill_buffer(buffer, done=False)
    advantages, target = buffer.compute_advantages_TD(final_value=torch.tensor([VALUE]))
    expected_return = [1, 1, 1, 1]
    expected_value = [x + GAMMA * VALUE for x in expected_return]
    expected_advantages = list(
        map(sub, expected_value, [VALUE for _ in range(buffer_size)])
    )
    assert advantages.squeeze(-1).tolist() == pytest.approx(expected_advantages)
    assert target.squeeze(-1).tolist() == pytest.approx(expected_value)

    # 2-step 1-update
    buffer_size = 3
    n_steps = 2
    buffer_config = BufferConfig(
        obs_shape=obs_shape,
        setting="n-step",
        buffer_size=buffer_size,
        n_steps=n_steps,
        gamma=GAMMA,
    )
    buffer = RolloutBuffer(buffer_config)
    buffer.internals.states[0].copy_(buffer.obs2tensor(obs))
    buffer = fill_buffer(buffer, done=False)
    advantages, target = buffer.compute_advantages_n_step(
        final_value=torch.tensor([VALUE])
    )
    expected_return = [REWARD_VALUE + GAMMA * REWARD_VALUE for _ in range(n_steps)]
    expected_value = [x + (GAMMA**n_steps) * VALUE for x in expected_return]
    expected_advantages = list(
        map(sub, expected_value, [VALUE for _ in range(buffer_size)])
    )
    assert advantages.squeeze(-1).tolist() == pytest.approx(expected_advantages)
    assert target.squeeze(-1).tolist() == pytest.approx(expected_value)
