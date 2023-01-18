"""
Logging helpers
"""
import os
from typing import Any, Dict, Optional, Union

import wandb
from torch.utils.tensorboard import SummaryWriter

from bombproofbasis.types import A2CConfig, LoggingConfig


class Logger:
    """
    Logger for A2C agent. Used to handle the logging into wandb for the \
        agent's training.
    """

    def __init__(self, config: LoggingConfig, agent_config: A2CConfig) -> None:
        """Create the logger based on the config

        Args:
            config (LoggingConfig): Logger config
        """
        self.config = config
        if self.config.tensorboard:
            log_dir = os.path.join(
                self.config.log_path,
                f"{self.config.project_name} {self.config.run_name}",
            )
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        if self.config.wandb:
            folder = self.config.log_path
            os.makedirs(folder, exist_ok=True)
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=agent_config.dict(),
                group=self.config.group,
                dir=folder,
            )
            recurrent = "LSTM" in "".join(
                agent_config.value_network.architecture
            ) or "LSTM" in "".join(agent_config.policy_network.architecture)
            wandb.config.recurrent = recurrent

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

    def log_wandb(
        self, scalars: Dict[str, Union[int, float]], timestep: int, commit: bool = False
    ) -> None:
        """
            Log values into tensorboard

            Args:
                scalars (Dict[str, Union[int, float]]): dict of names of scalars \
                    and their corresponding values.
                timestep (int): The timestep to report
            """
        scalars["timestep"] = timestep
        wandb.log(scalars, commit=commit)

    def log(
        self, scalars: Dict[str, Union[int, float]], timestep: int, commit: bool = False
    ) -> None:
        """
        Logs the relevant values according to the right method

        Args:
            timestep (int): The current timestep
        """

        if self.config.tensorboard and timestep % self.config.logging_frequency == 0:
            self.log_tensorboard(scalars, timestep)
        if self.config.wandb and timestep % self.config.logging_frequency == 0:
            self.log_wandb(scalars, timestep, commit)

    def add_histogram(self, name: str, values: Any, timestep: int) -> None:
        """
        Wrapper function for adding single histograms

        Args:
            name (str): Name of the graph
            values (Any): Values to plot
            timestep (int): The current timestep
        """
        if self.config.tensorboard:
            self.writer.add_histogram(
                name,
                values,
                timestep,
            )
        if self.config.wandb:
            wandb.log({name: wandb.Histogram(values), "episode": timestep})

    def log_histograms(
        self,
        rollout,
        networks,
        timestep: int,
        weights: bool = False,
    ) -> None:
        """
        Log interesting KPIs distributions in histograms

        Args:
            rollout (RolloutBuffer): Rollout to extract values from
            networks (A2CNetworks): Networks for weights plotting
            timestep (int): The current timestep
            weights (bool, optional): Wether or not to plot networks \
                weights. Defaults to False.
        """
        if self.config.tensorboard:
            self.add_histogram(
                "Histograms/Values",
                rollout.logs.values,
                timestep,
            )
            self.add_histogram(
                "Histograms/Rewards",
                rollout.logs.rewards,
                timestep,
            )
            self.add_histogram(
                "Histograms/Advantages",
                rollout.logs.advantages,
                timestep,
            )
            self.add_histogram(
                "Histograms/Targets",
                rollout.logs.targets,
                timestep,
            )
            if weights and (timestep % 10 == 0):
                for name, weight in networks.actor.named_parameters():
                    self.add_histogram(f"Weights/{name} actor", weight, timestep)

                for name, weight in networks.critic.named_parameters():
                    self.add_histogram(f"Weights/{name} critic", weight, timestep)

        if self.config.wandb:
            wandb.log(
                {
                    "Values": wandb.Histogram(rollout.logs.values),
                    "Rewards": wandb.Histogram(rollout.logs.rewards),
                    "Advantages": wandb.Histogram(rollout.logs.advantages),
                    "Targets": wandb.Histogram(rollout.logs.targets),
                    "episode": timestep,
                }
            )

    def run_summary(self, values: Optional[dict]):
        """
        Add a run summary to a wandb run

        Args:
            values (Optional[dict]): The values (name and val) to add to the run
        """
        if self.config.wandb and (values is not None):
            for key, val in values.items():
                wandb.run.summary[key] = val

    def finish_logging(self):
        """
        Finish the wandb run and move on to the next
        """
        if self.config.wandb:
            wandb.finish()

    def log_model(self, paths: Optional[dict]):
        """
        Save models into wandb

        Args:
            paths (Optional[dict]): The paths (name and path) to the saved models.
        """
        if self.config.wandb and (paths is not None):
            for key, val in paths.items():
                artifact = wandb.Artifact(key, type="model")
                artifact.add_file(val)
                wandb.run.log_artifact(artifact)
