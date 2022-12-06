"""
Type helpers for the project
"""
from pathlib import Path
from typing import Optional, Sequence

import gym
import numpy as np
import torch
from pydantic import BaseModel, validator


class NetworkConfig(BaseModel):
    learning_rate: float
    architecture: list
    input_shape: int
    output_shape: int
    actor: Optional[bool] = True
    model_path: Optional[Path]
    hardware: Optional[str] = "CPU"

    @validator("hardware", allow_reuse=True)
    def hardware_match(cls, v: str) -> str:
        possible_values = ["CPU", "GPU"]
        if v not in possible_values:
            raise ValueError(f"hardware must be in {possible_values}, you chose {v}")
        return v


NoneType = type(None)


class BufferStep(BaseModel):
    reward: float
    done: bool
    value: float
    log_prob: float
    entropy: float
    KL_divergence: float


class BufferInternals(BaseModel):
    rewards: np.ndarray
    dones: np.ndarray
    KL_divergences: np.ndarray
    values: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor
    len: int
    returns: Optional[Sequence[list]]
    advantages: Optional[Sequence[torch.Tensor]]

    class Config:
        arbitrary_types_allowed = True


class BufferConfig(BaseModel):
    setting: str
    gamma: float
    buffer_size: int
    n_steps: int


class ScalerConfig(BaseModel):
    scale: bool
    method: Optional[str] = "standardize"


class AgentConfig(BaseModel):
    learning_rate: float
    environment: gym.Env
    agent_type: str
    continous: bool = False
    law: Optional[str]
    policy_network: NetworkConfig
    value_network: Optional[NetworkConfig]
    scaler: Optional[ScalerConfig]

    class Config:
        arbitrary_types_allowed = True


class TrainingConfig(BaseModel):
    nb_timesteps_train: int
    nb_episodes_test: int
    learning_start: Optional[float] = 0
    logging: str
    render: Optional[bool] = False

    @validator("logging")
    def logging_match(cls, v: str) -> str:
        possible_values = ["tensorboard", "wandb"]
        if v.lower() not in possible_values:
            raise ValueError(f"logging must be in {possible_values}, you chose {v}")
        return v

    @validator("learning_start")
    def check_learning_start(cls, v: float) -> float:
        if isinstance(v, float):
            if not (v.is_integer()) and not (0 <= v <= 1):
                raise ValueError(
                    f"learning start is a fraction and not between 0 and 1 : \
                        {v}, make sure you pass either an int or a fraction\
                             between 0 and 1"
                )
        return v
