from copy import copy

import numpy as np
import pytest

from bombproofbasis.types import ScalerConfig
from bombproofbasis.utils.normalize import SimpleStandardizer
from bombproofbasis.utils.scalers import Scaler


def test_SimpleStandardizer():
    input_data = [1, 2, 3, 4, 5, 6]

    standardizer_shift_mean = SimpleStandardizer(shift_mean=True, clip=False)
    standardizer_clip = SimpleStandardizer(
        shift_mean=False, clip=True, clipping_range=(1, 2)
    )
    standardizer_vanilla = SimpleStandardizer(shift_mean=False, clip=False)

    # Faulty standardizer configurations
    with pytest.raises(ValueError):
        # bound 1 > bound 2
        SimpleStandardizer(shift_mean=False, clip=True, clipping_range=(10, -10))
    with pytest.raises(ValueError):
        # bound 1 == bound 2
        SimpleStandardizer(shift_mean=False, clip=True, clipping_range=(10, 10))

    output_vanilla = []

    for j, standardizer in enumerate(
        [
            standardizer_vanilla,
            standardizer_clip,
            standardizer_shift_mean,
        ]
    ):
        data_upto_now = []
        output_data = []
        for i, x in enumerate(input_data):
            data_upto_now.append(x)
            x_np = np.array([x])
            standardizer.partial_fit(x_np)
            if i > 0:
                # Check for correct mean and std computation
                assert standardizer.mean == np.mean(data_upto_now)
                assert standardizer.std == np.std(data_upto_now)

        # We can only check for correcteness of scaling on static data \
        # so we need to wait till we have seen all examples to assert results \
        # because we are doing online algorithms.
        for i, x in enumerate(input_data):
            x_np = np.array([x])
            out = standardizer.transform(x_np)
            output_data.append(out)
        if j == 0:
            output_vanilla = copy(output_data)
        print(j, output_data, np.mean(output_data))
        # check if the transformation is a linear application
        assert sorted(output_data) == output_data
        # check that mean was correctly shifted to 0 with tolerance to numerical errors.
        if standardizer.shift_mean:
            assert np.mean(output_data) == pytest.approx(0)

        # check that clipping went correctly:
        # - everything is in bounds
        # - all data that should have been clipped has been clipped correctly
        if standardizer.clip:
            assert max(output_data) <= standardizer.clipping_range[1]
            assert min(output_data) >= standardizer.clipping_range[0]
            for k, y in enumerate(output_vanilla):
                if y < standardizer.clipping_range[0]:
                    assert output_data[k] == standardizer.clipping_range[0]
                elif y > standardizer.clipping_range[1]:
                    assert output_data[k] == standardizer.clipping_range[1]


def test_Scaler():
    scaler = Scaler(method="standardize")

    size = 100
    observations = np.random.uniform(low=5, high=10, size=(size, 5))
    low_std_rewards = np.random.uniform(low=5, high=6, size=size)
    high_std_rewards = np.random.uniform(low=0, high=6000, size=size)
    plus_minus_rewards = np.random.uniform(low=-6, high=6, size=size)
    for rewards in (low_std_rewards, high_std_rewards, plus_minus_rewards):
        new_observations, new_rewards = [], []
        # Warmup
        for i, (obs, reward) in enumerate(zip(observations, rewards)):
            new_obs, new_reward = scaler.scale(obs, reward, fit=True, transform=False)
        for i, (obs, reward) in enumerate(zip(observations, rewards)):
            # fit and transformed are tested in test_SimpleStandardizer
            new_obs, new_reward = scaler.scale(obs, reward, fit=False, transform=True)
            new_observations.append(new_obs)
            new_rewards.append(new_reward)
        assert np.mean(new_rewards) / max(new_rewards) == pytest.approx(
            np.mean(rewards) / max(rewards)
        )
        assert np.sign(new_rewards).all() == np.sign(rewards).all()
        assert abs(np.mean(new_rewards)) < abs(np.mean(rewards))
        assert np.mean(new_observations) == pytest.approx(0)

    with pytest.raises(ValueError):
        scaler = Scaler(method="undefined")
