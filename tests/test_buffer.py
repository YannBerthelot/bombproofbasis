import gym
import numpy as np
import pytest
import torch

from bombproofbasis.types import BufferConfig
from bombproofbasis.utils.buffer import BufferStep, RolloutBuffer

REWARD_VALUE = -1
ENV = gym.make("CartPole-v1")
STEP = BufferStep(
    reward=REWARD_VALUE,
    done=False,
    obs=np.ones((4,)),
    value=torch.tensor([0.1]),
    log_prob=torch.tensor([0.1]),
    action=1,
)


def fill_buffer(buffer: RolloutBuffer, done=False) -> RolloutBuffer:
    if done:
        for i in range(buffer.config.buffer_size - 1):
            buffer.add(STEP)
        buffer.add(
            BufferStep(
                reward=REWARD_VALUE,
                done=True,
                obs=np.ones((4,)),
                value=torch.tensor([0.1]),
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
    expected_return_MC_1 = rewards[0] + rewards[1] * gamma + rewards[2] * (gamma**2)
    expected_return_MC_2 = rewards[0] + rewards[1] * gamma
    expected_return_MC_3 = rewards[0]
    expected_return_MC_4 = 0
    returns_MC = [
        expected_return_MC_1,
        expected_return_MC_2,
        expected_return_MC_3,
        expected_return_MC_4,
    ]
    return returns_MC


def test_add_buffer():

    obs, info = ENV.reset()
    # faulty configs
    with pytest.raises(ValueError):
        faulty_buffer_config = BufferConfig(
            obs_shape=obs.shape, gamma=1.99, setting="n-step"
        )
        RolloutBuffer(faulty_buffer_config)
    with pytest.raises(ValueError):
        faulty_buffer_config = BufferConfig(
            obs_shape=obs.shape, buffer_size=-1, setting="n-step"
        )
        RolloutBuffer(faulty_buffer_config)
    with pytest.raises(ValueError):
        faulty_buffer_config = BufferConfig(
            obs_shape=obs.shape, n_steps=-1, setting="n-step"
        )
        RolloutBuffer(faulty_buffer_config)
    buffer_size = 10
    buffer_config = BufferConfig(
        obs_shape=obs.shape, buffer_size=buffer_size, setting="n-step"
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
    buffer_config = BufferConfig(obs_shape=obs.shape, buffer_size=4, setting="n-step")
    buffer = RolloutBuffer(buffer_config)
    buffer.internals.states[0].copy_(buffer.obs2tensor(obs))
    buffer = fill_buffer(buffer, done=True)
    final_value = torch.tensor([REWARD_VALUE], dtype=torch.float32)
    returns = buffer.compute_return(final_value=final_value)
    returns_MC = MC_return_safe(buffer)
    assert returns.squeeze(-1).tolist() == pytest.approx(returns_MC)


# def test_return_computation():
#     buffer_config = BufferConfig(setting="MC", gamma=0.99, buffer_size=5, n_steps=2)
#     buffer = RolloutBuffer(buffer_config)
#     # Computation checks
#     rewards = np.array([1, 1, 1, 1])
#     gamma = 0.99
#     expected_return_MC_1 = 1 + 1 * 0.99 + 1 * (0.99**2) + 1 * (0.99**3)
#     assert buffer.compute_return(rewards=rewards, gamma=gamma) == pytest.approx(
#         expected_return_MC_1
#     )
#     expected_return_MC_2 = 1 + 1 * 0.99 + 1 * (0.99**2)
#     expected_return_MC_3 = 1 + 1 * 0.99
#     expected_return_MC_4 = 1
#     returns_MC = [
#         expected_return_MC_1,
#         expected_return_MC_2,
#         expected_return_MC_3,
#         expected_return_MC_4,
#     ]
#     assert buffer.compute_MC_returns(rewards=rewards, gamma=gamma) == pytest.approx(
#         returns_MC
#     )

#     expected_return_2_steps = 1 + 1 * 0.99
#     n_steps = 2
#     expected_returns_2_steps = [
#         expected_return_2_steps,
#         expected_return_2_steps,
#         expected_return_2_steps,
#         1,
#     ]
#     assert buffer.compute_n_step_returns(
#         rewards=rewards, gamma=gamma, buffer_size=4, n_steps=n_steps
#     ) == pytest.approx(expected_returns_2_steps)


# def test_advantage_computation():
#     gamma = 0.99
#     buffer_config = BufferConfig(setting="MC", gamma=gamma, buffer_size=5, n_steps=1)
#     buffer = RolloutBuffer(buffer_config)
#     rewards = np.array([1, 1, 1, 1])

#     # Simple case n-step = 1
#     n_steps = 1
#     returns = buffer.compute_n_step_returns(
#         rewards=rewards, gamma=gamma, buffer_size=4, n_steps=n_steps
#     )
#     values = torch.Tensor([1, 1, 1, 1])
#     dones = [False, False, False, True]
#     # last_val = 1  # To be made tensor?
#     assert buffer.compute_n_step_advantages(
#         returns=returns,
#         values=values,
#         dones=dones,
#         gamma=gamma,
#         n_steps=n_steps,
#     )[0] == torch.subtract(
#         torch.add(torch.tensor(rewards[0]), torch.multiply(gamma, values[1])), values[0]
#     )  # r + gamma * Vs' - Vs

#     # n-step>1
#     n_steps = 2
#     returns = buffer.compute_n_step_returns(
#         rewards=rewards, gamma=gamma, buffer_size=4, n_steps=n_steps
#     )
#     expected_advantage = torch.subtract(
#         torch.add(
#             torch.tensor(returns[0]),
#             torch.multiply(torch.pow(torch.Tensor([gamma]), n_steps), values[n_steps]),
#         ),
#         values[0],
#     )
#     assert (
#         buffer.compute_n_step_advantages(
#             returns=returns,
#             values=values,
#             dones=dones,
#             gamma=gamma,
#             n_steps=n_steps,
#         )[0]
#         == expected_advantage
#     )
#     with pytest.raises(ValueError):
#         # Invalid n-steps value
#         buffer.compute_n_step_advantages(
#             returns=returns,
#             values=values,
#             dones=dones,
#             gamma=gamma,
#             n_steps=0,
#         )
#     expected_advantages = torch.tensor(
#         [torch.subtract(returns[i], values[i]) for i in range(len(values))]
#     )
#     computed_advantages = buffer.compute_MC_advantages(returns, values)
#     # assert torch.allclose(computed_advantages.double(), expected_advantages.double())


# def test_update_advantages():
#     # Parameters
#     gamma = 0.99
#     buffer_size = 10
#     n_steps = 3
#     dones = np.zeros(buffer_size)
#     dones[-1] = 1
#     dones = [bool(x) for x in dones]
#     rewards = np.ones(buffer_size)
#     values = torch.ones((buffer_size, 1))

#     # MC
#     buffer_config = BufferConfig(
#         setting="MC", gamma=gamma, buffer_size=buffer_size, n_steps=n_steps
#     )
#     buffer = RolloutBuffer(buffer_config)

#     returns = buffer.compute_MC_returns(rewards=rewards, gamma=gamma)

#     for i in range(buffer_size):
#         step = BufferStep(
#             reward=rewards[i],
#             done=dones[i],
#             value=values[i],
#             KL_divergence=0.1,
#             log_prob=torch.tensor([[0.1]]),
#             entropy=torch.tensor([[0.1]]),
#         )
#         buffer.add(step)
#     buffer.update_advantages()
#     expected_advantages = torch.tensor(
#         [torch.subtract(returns[i], values[i]) for i in range(len(values))]
#     )
#     # assert torch.allclose(
#     #     buffer.internals.advantages.double(), expected_advantages.double()
#     # )

#     # n-step
#     buffer_config = BufferConfig(
#         setting="n-step", gamma=gamma, buffer_size=buffer_size, n_steps=n_steps
#     )
#     buffer = RolloutBuffer(buffer_config)

#     returns = buffer.compute_n_step_returns(
#         rewards=rewards, gamma=gamma, buffer_size=buffer_size, n_steps=n_steps
#     )

#     for i in range(buffer_size):
#         step = BufferStep(
#             reward=rewards[i],
#             done=dones[i],
#             value=values[i],
#             KL_divergence=0.1,
#             log_prob=torch.tensor([[0.1]]),
#             entropy=torch.tensor([[0.1]]),
#         )
#         buffer.add(step)
#     buffer.update_advantages()
#     expected_advantages = buffer.compute_n_step_advantages(
#         returns=returns,
#         values=values,
#         dones=dones,
#         gamma=gamma,
#         n_steps=n_steps,
#     )
#     # assert torch.allclose(
#     #     torch.Tensor(buffer.internals.advantages).double(),
#     #     torch.Tensor(expected_advantages).double(),
#     # )


# def test_other_buffer():
#     gamma = 0.99
#     buffer_size = 10
#     n_steps = 3
#     dones = np.zeros(buffer_size)
#     dones[-1] = 1
#     dones = [bool(x) for x in dones]
#     rewards = np.ones(buffer_size)
#     values = torch.ones((buffer_size, 1))

#     # MC
#     buffer_config = BufferConfig(
#         setting="MC", gamma=gamma, buffer_size=buffer_size, n_steps=n_steps
#     )
#     buffer = RolloutBuffer(buffer_config)

#     for i in range(buffer_size):
#         step = BufferStep(
#             reward=rewards[i],
#             done=dones[i],
#             value=values[i],
#             KL_divergence=0.1,
#             log_prob=torch.tensor([[0.1]]),
#             entropy=torch.tensor([[0.1]]),
#         )
#         buffer.add(step)
#     buffer.update_advantages()
#     buffer.show()
#     buffer.get_steps()
#     for x in buffer.get_steps_generator():
#         pass
#     rewards = buffer.internals.rewards
#     buffer.clean()
#     assert buffer.internals.rewards[n_steps:].all() == 0
#     assert (buffer.internals.rewards[:n_steps] == rewards[-n_steps:]).all()
#     assert buffer.internals.len == len(buffer.internals.rewards[:n_steps])
