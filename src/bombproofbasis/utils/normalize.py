"""
Normalization/standardization/scaling classes and function
"""
import os
import pathlib
import pickle
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import torch

from bombproofbasis.network.utils import t


class SimpleStandardizer:
    """
    Simple standardizer adapted to online learning using Welfor's \
        online algorithm for variance computation
    """

    def __init__(
        self,
        clip: bool = False,
        shift_mean: bool = True,
        clipping_range: Tuple[int, int] = None,
    ) -> None:
        """_summary_

        Args:
            clip (bool, optional): Wether or not to clip values after scaling.\
                Defaults to False.
            shift_mean (bool, optional): Wether or not to shift \
                (hence, modify) the mean of the scaled values. \
                This should be avoided with rewards as\
                it can modify the sign (and thus the information provided by) \
                some rewards
                    Defaults to True.
            clipping_range (Tuple[int, int], optional): The range in which \
                to clip the scaled values. Defaults to None.

        Raises:
            ValueError: Clipping is set but no range is provided
            ValueError: Clipping is set and clipping range is not valid
        """

        # Init internals
        self._count = 0
        self.mean = None
        self.M2 = None
        self.std = None
        self._shape = None
        self.shift_mean = shift_mean
        self.clip = clip
        if self.clip:
            if clipping_range is None:
                raise ValueError(
                    "Clipping is activated but clipping range is not defined"
                )
            if clipping_range[0] > clipping_range[1]:
                raise ValueError(
                    f"Lower clipping bound ({clipping_range[0]}) is larger\
                        than Upper clipping bound ({clipping_range[1]})"
                )
            if clipping_range[0] == clipping_range[1]:
                raise ValueError(
                    f"Lower clipping bound ({clipping_range[0]}) is equal to\
                        Upper clipping bound ({clipping_range[1]})"
                )
            self.clipping_range = clipping_range

    def partial_fit(self, newValue: npt.NDArray[np.float64]) -> None:
        """
        Fit the internal values for scaling (std and mean) based on the new \
        input (the input can be multi-dimensionnal but only one input \
        should be provided at a time)

        Uses Welfor's online algorithm :
        https://en.m.wikipedia.org/wiki/Algorithms_for_calculating_variance

        Args:
            newValue (npt.NDArray[np.float64]): _description_

        Raises:
            ValueError: _description_
        """

        self._count += 1
        if self.mean is None:
            self.mean = newValue
            self.std = np.zeros(len(newValue))
            self.M2 = np.zeros(len(newValue))
            self._shape = newValue.shape
        else:
            if self._shape != newValue.shape:
                raise ValueError(
                    f"The shape of samples has changed ({self._shape} to \
                        {newValue.shape})"
                )
        delta = newValue - self.mean
        self.mean = self.mean + (delta / self._count)
        delta2 = newValue - self.mean
        self.M2 += np.multiply(delta, delta2)
        if self._count >= 2:
            self.std = np.sqrt(self.M2 / self._count)
            self.std = np.nan_to_num(self.std, nan=1)

    @staticmethod
    def numpy_transform(
        value: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        shift_mean: bool = True,
        clip: bool = False,
        clipping_range: tuple = None,
    ) -> np.ndarray:
        """
        Transform a numpy array into its scaled counterparts.

        Args:
            value (np.ndarray): The value to be scaled (can be multi-dim)
            mean (np.ndarray): The mean(s) to consider for scaling
            std (np.ndarray): The standard-deviation(s) to consider for scaling
            shift_mean (bool, optional): Wether or not to shift the mean of \
                the values (w.r.t the max value of the initial distribution) \
                when scaling. If shifted, the new mean will be 0.
                Defaults to True.
            clip (bool, optional): Wether or not to clip the scaled value. \
                Defaults to False.
            clipping_range (tuple, optional): the range in which to clip the \
                scaled value. Defaults to None.

        Raises:
            ValueError: If the means/stds and the value to be scaled are not \
                of the same shape. (each dim in the std/mean corresponds to a \
                dim in the values)

        Returns:
            np.ndarray: The scaled value
        """
        if (value.shape != mean.shape) or (value.shape != std.shape):
            raise ValueError(
                f"Different shape between either value to be scaled and std or\
                    value to be scaled and mean. \
                        Value to be scaled : {value.shape}, \
                        std :{std.shape},\
                        mean :{mean.shape}"
            )
        if shift_mean:
            new_value = (value - mean) / std
        else:
            for i, sigma in enumerate(std):
                if abs(sigma) < 1:
                    std[i] = 1 / sigma
            new_value = value / std
        if clip:
            return np.clip(new_value, clipping_range[0], clipping_range[1])
        else:
            return new_value

    @staticmethod
    def pytorch_transform(
        value: torch.Tensor,
        mean: np.ndarray,
        std: np.ndarray,
        shift_mean: bool = True,
        clip: bool = False,
        clipping_range: tuple = None,
    ) -> torch.Tensor:
        """
        Transform a Pytorch array into its scaled counterparts.

        Args:
            value (np.ndarray): The value to be scaled (can be multi-dim)
            mean (np.ndarray): The mean(s) to consider for scaling
            std (np.ndarray): The standard-deviation(s) to consider for scaling
            shift_mean (bool, optional): Wether or not to shift the mean of \
                the values (w.r.t the max value of the initial distribution) \
                when scaling. If shifted, the new mean will be 0.
                Defaults to True.
            clip (bool, optional): Wether or not to clip the scaled value. \
                Defaults to False.
            clipping_range (tuple, optional): the range in which to clip the \
                scaled value. Defaults to None.

        Raises:
            ValueError: If the means/stds and the value to be scaled are not \
                of the same shape. (each dim in the std/mean corresponds to a \
                dim in the values)

        Returns:
            np.ndarray: The scaled value
        """
        if (value.shape != mean.shape) or (value.shape != std.shape):
            raise ValueError(
                f"Different shape between either value to be scaled and std or\
                    value to be scaled and mean. \
                        Value to be scaled : {value.shape}, \
                        std :{std.shape},\
                        mean :{mean.shape}"
            )
        if shift_mean:
            new_value = torch.div((torch.sub(value, t(mean))), t(std))
        else:
            for i, sigma in enumerate(std):
                if abs(sigma) < 1:
                    std[i] = 1 / sigma
            new_value = torch.div(value, t(std))
        if clip:
            return torch.clip(new_value, clipping_range[0], clipping_range[1])
        else:
            return new_value

    def transform(
        self, value: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform the given value (can be multi-dim) into its scaled \
            counterpart based on the current scaler internals (std and mean)

        Args:
            value (Union[np.ndarray, torch.Tensor]): The value to be scaled

        Raises:
            TypeError: Type of input is not supported.

        Returns:
            Union[np.ndarray, torch.Tensor]: The scaled value
        """
        if (self.std is None) or (self.mean is None):
            raise ValueError(
                "Tried to scale without trained internals, mean and/or std is None"
            )
        std_temp = self.std
        std_temp[std_temp == 0.0] = 1
        if isinstance(value, np.ndarray):
            return self.numpy_transform(
                value,
                self.mean,
                self.std,
                self.shift_mean,
                self.clip,
                self.clipping_range,
            )
        elif isinstance(value, torch.Tensor):
            return self.pytorch_transform(
                value,
                self.mean,
                self.std,
                self.shift_mean,
                self.clip,
                self.clipping_range,
            )
        else:
            raise TypeError(f"type of transform input {type(value)} not handled atm")

    def save(
        self, path: Union[str, pathlib.Path] = ".", name: str = "standardizer"
    ) -> None:
        """
        Save the current state of the scaler in pickle format.

        Args:
            path (Union[str, pathlib.Path], optional): Where to save. \
                Defaults to ".".
            name (str, optional): The name of the saved file (without the extension). \
                Defaults to "standardizer".
        """
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, name + ".pkl"), "wb") as file:
            pickle.dump(self, file)

    def load(self, path: Union[str, pathlib.Path], name: str = "standardizer") -> None:
        """
        Load a pre-trained scaler from a savefile.

        Args:
            path (Union[str, pathlib.Path]): Save folder
            name (str, optional): The name of the saved file (without the \
                extension). Defaults to "standardizer".
        """
        with open(os.path.join(path, name + ".pkl"), "rb") as file:
            save = pickle.load(file)
            self.std = save.std
            self.mean = save.mean
            self._count = save._count
            self.M2 = save.M2
            self._shape = save._shape
            self.shift_mean = save.shift_mean
            self.clip = save.clip
            self.clipping_range = save.clipping_range
