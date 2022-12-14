"""
Logging helpers
"""
import os
from typing import Dict, Union

from torch.utils.tensorboard import SummaryWriter

from bombproofbasis.types import LoggingConfig

# def wandb_train_logging(config: TrainingConfig, scaler: Optional[Scaler]) -> None:
#     os.makedirs("./data", exist_ok=True)
#     artifact = wandb.Artifact(f"{config.comment}_model_actor", type="model")
#     artifact.add_file(f"{config.model_path}/{config.comment}_best_actor.pth")
#     wandb.run.log_artifact(artifact)

#     artifact = wandb.Artifact(f"{config.comment}_model_critic", type="model")
#     artifact.add_file(f"{config.model_path}/{config.comment}_best_critic.pth")
#     wandb.run.log_artifact(artifact)

#     if scaler is not None:
#         artifact = wandb.Artifact(f"{config.comment}_obs_scaler", type="scaler")
#         pickle.dump(
#             scaler,
#             open(f"data/{config.comment}_obs_scaler.pkl", "wb"),
#         )
#         artifact.add_file(f"data/{config.comment}_obs_scaler.pkl")

#         # Save the artifact version to W&B and mark it as the output of this run
#         wandb.run.log_artifact(artifact)
#     else:
#         pass


class Logger:
    """
    Logger for A2C agent. Used to handle the logging into wandb for the \
        agent's training.
    """

    def __init__(self, config: LoggingConfig) -> None:
        """Create the logger based on the config

        Args:
            config (LoggingConfig): Logger config
        """
        self.config = config
        if self.config.logging_output == "tensorboard":
            os.makedirs(config.log_path, exist_ok=True)
            self.writer = SummaryWriter(comment="statistics", log_dir=config.log_path)

    def log_tensorboard(
        self, scalars: Dict[str, Union[int, float]], timestep: int
    ) -> None:
        """
        Log values into tensorboard

        Args:
            scalars (Dict[str, Union[int, float]]): dict of names of scalars \
                and their corresponding values.
            timestep (int): The timestep to report
        """
        for name, value in scalars.items():
            if value is not None:
                self.writer.add_scalar(name, value, timestep)

    def log(self, scalars: Dict[str, Union[int, float]], timestep: int) -> None:
        """
        Logs the relevant values according to the right method

        Args:
            timestep (int): The current timestep
        """

        if self.config.logging_output == "tensorboard":
            self.log_tensorboard(scalars, timestep)
