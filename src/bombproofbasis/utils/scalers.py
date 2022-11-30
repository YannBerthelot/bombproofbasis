"""
Macro scaler objects
"""
from pathlib import Path
from typing import Tuple

import numpy as np

from bombproofbasis.types import ScalerConfig
from bombproofbasis.utils.normalize import SimpleStandardizer


class Scaler:
    """
    Scaler wrapper for various scaling methods
    """

    def __init__(self, config: ScalerConfig) -> None:
        """
        Init the object

        Args:
            config (ScalerConfig): Scaler configuration :
                - scale : bool
                - method (str, optional): Defaults to "standardize".
        """
        self.config = config
        self.obs_scaler, self.reward_scaler, self.target_scaler = self.get_scalers(
            self.config.method
        )

    def get_scalers(
        self, method: str = "standardize"
    ) -> Tuple[SimpleStandardizer, SimpleStandardizer, SimpleStandardizer]:
        """
        Get the actual scalers for the requested method.

        Args:
            method (str, optional): The scaling method \
                amongst : "standardize". \
                Defaults to "standardize".

        Raises:
            ValueError: If the scaling method is not valid or implemented.

        Returns:
            Tuple[SimpleStandardizer, SimpleStandardizer, SimpleStandardizer]:\
                The observation, reward and target scalers.
        """
        if method == "standardize":
            obs_scaler = SimpleStandardizer(shift_mean=True, clip=False)
            reward_scaler = SimpleStandardizer(shift_mean=False, clip=False)
            target_scaler = SimpleStandardizer(shift_mean=False, clip=False)
        else:
            raise ValueError(f"Method of scaling {method} not implemented")
        return obs_scaler, reward_scaler, target_scaler

    def save(self, path: Path, name: str) -> None:
        """
        Save the state of scalers to pickle format.

        Args:
            path (Path): Path to save folder
            name (str): Name of the savefile (without extension)
        """
        self.obs_scaler.save(path=path, name="obs_" + name)
        self.reward_scaler.save(path=path, name="reward_" + name)
        self.target_scaler.save(path=path, name="target_" + name)

    def load(self, path: Path, name: str) -> None:
        """
        Set the state of scalers from a savefile.

        Args:
            path (Path): Path to save folder
            name (str): Name of the savefile (without extension)
        """
        self.obs_scaler, self.reward_scaler, self.target_scaler = self.get_scalers()
        self.obs_scaler.load(path, "obs_" + name)
        self.reward_scaler.load(path, "reward_" + name)
        self.reward_scaler.load(path, "target_" + name)

    def scale(
        self, obs: np.ndarray, reward: float, fit: bool = True, transform: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Fit and/or Scale the values using the current internal state of scalers.

        Args:
            obs (np.ndarray): The observation to fit and/or scale
            reward (float): The reward to fit and/or scale
            fit (bool, optional): Wether or not to fit using the input.\
                Defaults to True.
            transform (bool, optional): Wether or not to scale the input.\
                Defaults to True.

        Returns:
            Tuple[np.ndarray, float]: The scaled observation and reward
        """
        if fit:
            self.obs_scaler.partial_fit(obs)
            self.reward_scaler.partial_fit(np.array([reward]))
        if transform:
            reward = self.reward_scaler.transform(np.array([reward]))[0]
            obs = self.obs_scaler.transform(obs)
        return obs, reward
