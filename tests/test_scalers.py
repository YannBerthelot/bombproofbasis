import os
import pickle
from copy import copy

import numpy as np
import pytest
import torch

from bombproofbasis.network.utils import t
from bombproofbasis.types import ScalerConfig
from bombproofbasis.utils.normalize import SimpleStandardizer
from bombproofbasis.utils.scalers import Scaler


def test_SimpleStandardizer():

    # Faulty standardizer configurations
    with pytest.raises(ValueError):
        # bound 1 > bound 2
        SimpleStandardizer(shift_mean=False, clip=True, clipping_range=(10, -10))
    with pytest.raises(ValueError):
        # bound 1 == bound 2
        SimpleStandardizer(shift_mean=False, clip=True, clipping_range=(10, 10))
    with pytest.raises(ValueError):
        # Clip activated but no clipping range
        SimpleStandardizer(shift_mean=False, clip=True)

    # Scale without trained internals first
    with pytest.raises(ValueError):
        stdzer = SimpleStandardizer()
        stdzer.transform(np.array([3, 4, 5]))

    # Change in shape during fit
    with pytest.raises(ValueError):
        stdzer = SimpleStandardizer()
        stdzer.partial_fit(np.array([3, 4, 5]))
        stdzer.partial_fit(np.array([3, 4]))

    # Check for shape difference in internals and value
    with pytest.raises(ValueError):
        # Numpy
        stdzer = SimpleStandardizer()
        stdzer.partial_fit(np.array([1, 2, 3]))
        stdzer.partial_fit(np.array([3, 4, 5]))
        stdzer.transform(np.array([3, 4]))

    with pytest.raises(ValueError):
        # Pytorch
        stdzer = SimpleStandardizer()
        stdzer.partial_fit(t(np.array([1, 2, 3])))
        stdzer.partial_fit(t(np.array([3, 4, 5])))
        stdzer.transform(t(np.array([3, 4])))

    # Transform invalid type
    with pytest.raises(TypeError):
        stdzer = SimpleStandardizer()
        stdzer.partial_fit(np.array([1, 2, 3]))
        stdzer.partial_fit(np.array([3, 4, 5]))
        stdzer.transform([3, 4])

    input_data_raw_1 = np.arange(-10, 10, 0.1, dtype=np.float64)
    input_data_raw_2 = np.arange(5, 6, 0.01, dtype=np.float64)
    for input_data_raw in (input_data_raw_1, input_data_raw_2):
        for k, input_data in enumerate((input_data_raw, t(input_data_raw))):
            standardizer_shift_mean = SimpleStandardizer(shift_mean=True, clip=False)
            standardizer_clip = SimpleStandardizer(
                shift_mean=False, clip=True, clipping_range=(1, 2)
            )
            standardizer_vanilla = SimpleStandardizer(shift_mean=False, clip=False)
            output_vanilla = []
            for j, standardizer in enumerate(
                [
                    standardizer_vanilla,
                    standardizer_clip,
                    standardizer_shift_mean,
                ]
            ):
                data_upto_now = []
                output_data = np.array([])
                output_data_torch = torch.tensor([])

                for i, x in enumerate(input_data):
                    data_upto_now.append(x)
                    x = np.array([x])
                    standardizer.partial_fit(x)
                    if i > 0:
                        # Check for correct mean and std computation
                        assert standardizer.mean[0] == pytest.approx(
                            np.mean(data_upto_now), rel=1e-5
                        )
                        assert standardizer.std[0] == pytest.approx(
                            np.std(data_upto_now), rel=1e-5
                        )

                # We can only check for correcteness of scaling on static data \
                # so we need to wait till we have seen all examples to assert results \
                # because we are doing online algorithms.
                for i, x in enumerate(input_data):
                    if isinstance(input_data, np.ndarray):
                        x = np.array([x])
                        out = standardizer.transform(x)
                        output_data = np.append(output_data, out)
                        # output_data.append(out)
                    elif isinstance(input_data, torch.Tensor):
                        x = t(np.array([x]))
                        out = standardizer.transform(x)
                        output_data_torch = torch.cat([output_data_torch, out])
                        if i == len(input_data) - 1:
                            output_data = output_data_torch.numpy()

                    else:
                        raise ValueError("Unrecognized type of x")

                if j == 0:
                    output_vanilla = copy(output_data)

                # check if the transformation is a linear application
                assert sorted(list(output_data)) == list(output_data)
                # check that mean was correctly shifted to 0 with tolerance to numerical errors.
                if (standardizer.shift_mean) and (not standardizer.clip):
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

    # Test load and save
    stdzer = SimpleStandardizer()
    stdzer.partial_fit(np.array([1, 2, 3]))
    stdzer.partial_fit(np.array([3, 4, 5]))
    stdzer.save(".", "test_save")
    assert os.path.exists("./test_save.pkl")
    stdzer2 = SimpleStandardizer()
    stdzer2.load(".", "test_save")
    assert pickle.dumps(stdzer) == pickle.dumps(stdzer2)
    os.remove("./test_save.pkl")


def test_Scaler():
    scaler_config = ScalerConfig(scale=True)
    with pytest.raises(ValueError):
        scaler = Scaler(ScalerConfig(scale=True, method="undefined"))

    scaler = Scaler(config=scaler_config)

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

    # Test load and save
    scaler.save(".", "test_save")
    assert os.path.exists("./obs_test_save.pkl")
    assert os.path.exists("./reward_test_save.pkl")
    assert os.path.exists("./target_test_save.pkl")
    scaler2 = Scaler(config=scaler_config)
    scaler2.load(".", "test_save")
    assert scaler == scaler2
    os.remove("./obs_test_save.pkl")
    os.remove("./reward_test_save.pkl")
    os.remove("./target_test_save.pkl")
