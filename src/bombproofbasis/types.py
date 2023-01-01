"""
Type helpers for the project
"""
from pathlib import Path
from typing import Optional, Union

import gym
import numpy as np
import torch
from pydantic import BaseModel, validator


class NetworkConfig(BaseModel):
    learning_rate: float = 1e-3
    architecture: list
    input_shape: int
    output_shape: int
    actor: bool = True
    model_path: Optional[Path] = Path("./models")
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
    value: torch.Tensor
    log_prob: torch.Tensor
    action: int
    obs: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class BufferInternals(BaseModel):
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    len: int

    class Config:
        arbitrary_types_allowed = True


class BufferConfig(BaseModel):
    obs_shape: tuple
    gamma: Optional[float] = 0.99
    buffer_size: Optional[int] = 1000
    n_steps: Optional[int] = 1
    n_envs: Optional[int] = 1


class ScalerConfig(BaseModel):
    scale: bool
    method: Optional[str] = "standardize"


class AgentConfig(BaseModel):
    learning_rate: float = 1e-3
    environment: gym.Env
    agent_type: str
    continous: bool = False
    law: Optional[str]
    policy_network: NetworkConfig
    log_path: Optional[Path] = Path("./logs")
    value_network: Optional[NetworkConfig]
    scaler: Optional[ScalerConfig]

    class Config:
        arbitrary_types_allowed = True


class A2CConfig(AgentConfig):
    buffer: BufferConfig


class LoggingConfig(BaseModel):
    logging_output: str = "wandb"
    logging_frequency: int = 100

    @validator("logging_output")
    def logging_match(cls, v: str) -> str:
        possible_values = ["tensorboard", "wandb"]
        if v.lower() not in possible_values:
            raise ValueError(f"logging must be in {possible_values}, you chose {v}")
        return v


class TrainingConfig(BaseModel):
    agent: AgentConfig
    nb_timesteps_train: int
    nb_episodes_test: int
    learning_start: Optional[Union[float, int]] = 0
    logging: LoggingConfig
    log_path: Optional[Path] = Path("./logs")
    model_path: Optional[Path] = Path("./models")
    render: Optional[bool] = False
    comment: Optional[str]

    @validator("learning_start")
    def learning_start_match(cls, v: float) -> float:
        if not ((v.is_integer()) or isinstance(v, int)) and not (0 <= v <= 1):
            raise ValueError(
                f"learning start is a fraction and not between 0 and 1 : \
                        {v}, make sure you pass either an int or a fraction\
                             between 0 and 1"
            )
        elif (v.is_integer()) and v < 0:
            raise ValueError(f"Learning start is negative : {v}")
        else:
            return v
