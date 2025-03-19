import argparse
import os
from math import log2
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import optunahub
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger

from src.training.experiments.utils import read_trial_results
from src.training.flops_logger import count_flops
from src.training.train import train

load_dotenv(Path(__file__).parents[4] / ".env")

base_experiment_name = f"tune_ladder_v2"


class LadderExperiment:
    def __init__(
        self, flops_target: float, previous_flops_target: Optional[float] = None
    ):
        self.flops_target = flops_target
        self.experiment_name = f"{base_experiment_name}-{flops_target:.0e}"

        if previous_flops_target is not None:
            previous_experiment_name = (
                f"{base_experiment_name}-{previous_flops_target:.0e}"
            )
            study = optuna.load_study(
                study_name=previous_experiment_name,
                storage=f"sqlite:///{os.getenv('OPTUNA_DB_PATH')}",
            )
            self.previous_best_trial = study.best_trial
        else:
            self.previous_best_trial = None

    def create_sampler(self) -> optuna.samplers.BaseSampler:
        if self.previous_best_trial is None:
            module = optunahub.load_module(package="samplers/auto_sampler")
            return module.AutoSampler()
        else:
            search_space = {
                "learning_rate": np.geomspace(
                    self.previous_best_trial.params["learning_rate"] * 0.5,
                    self.previous_best_trial.params["learning_rate"] * 2,
                    5,
                ).tolist(),
                "hidden_layers": [
                    self.previous_best_trial.params["hidden_layers"],
                    self.previous_best_trial.params["hidden_layers"] + 1,
                ],
                "log2_hidden_dim": [
                    self.previous_best_trial.params["log2_hidden_dim"],
                    self.previous_best_trial.params["log2_hidden_dim"] + 1,
                ],
                "log2_batch_size": [
                    self.previous_best_trial.params["log2_batch_size"],
                    self.previous_best_trial.params["log2_batch_size"] + 1,
                ],
            }
            return optuna.samplers.GridSampler(search_space)

    def objective(self, trial: optuna.Trial) -> tuple[float, float]:
        """Objective function for Optuna optimization."""
        # Define the hyperparameters to optimize
        if self.previous_best_trial is not None:
            min_batch_size = 2 ** self.previous_best_trial.params["log2_batch_size"]

            config = {
                "hidden_layers": trial.suggest_int(
                    "hidden_layers",
                    self.previous_best_trial.params["hidden_layers"],
                    self.previous_best_trial.params["hidden_layers"] + 1,
                ),
                "hidden_dim": 2
                ** trial.suggest_int(
                    "log2_hidden_dim",
                    self.previous_best_trial.params["log2_hidden_dim"],
                    self.previous_best_trial.params["log2_hidden_dim"] + 1,
                ),
                "batch_size": 2
                ** trial.suggest_int(
                    "log2_batch_size",
                    self.previous_best_trial.params["log2_batch_size"],
                    self.previous_best_trial.params["log2_batch_size"] + 1,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    self.previous_best_trial.params["learning_rate"] * 0.5,
                    self.previous_best_trial.params["learning_rate"] * 2,
                    log=True,
                ),
                "lr_cooldown_fraction": 0.4,
                "accelerator": "gpu",
            }
        else:
            min_batch_size = 2**5

            config = {
                "hidden_layers": trial.suggest_int("hidden_layers", 1, 10),
                "hidden_dim": 2 ** trial.suggest_int("log2_hidden_dim", 3, 8),
                "batch_size": 2 ** trial.suggest_int("log2_batch_size", 5, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
                "lr_cooldown_fraction": 0.4,
                "accelerator": "cpu",
            }

        # Add a 10% flops bonus per doubling of batch size
        real_flops = self.flops_target * 1.1 ** log2(
            config["batch_size"] / min_batch_size
        )
        trial.set_user_attr("real_flops", real_flops)

        # Calculate the number of steps to run based on the flops
        flops_per_batch = count_flops(config)
        steps = int(real_flops / flops_per_batch)
        config["steps"] = steps
        trial.set_user_attr("steps", steps)

        # Create a logger for this trial
        trial_num = trial.number
        csv_logger = CSVLogger(
            save_dir=os.getenv("EXPERIMENT_LOGS_DIR"),
            name=self.experiment_name,
            version=trial_num,
        )
        trial.set_user_attr("logs_path", csv_logger.log_dir)

        # Train the model with these hyperparameters
        train(
            config,
            csv_logger=csv_logger,
            **(
                {"max_time": {"minutes": 1}} if self.previous_best_trial is None else {}
            ),
        )

        # Get the results
        results = read_trial_results(self.experiment_name, trial_num)
        train_value_loss = results.iloc[-1]["train_value_loss_ema"]

        return train_value_loss


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization")
    parser.add_argument(
        "--flops-target",
        type=float,
        help=f"Nominal flops target of the experiment",
    )
    parser.add_argument(
        "--previous-flops-target",
        type=float,
        default=None,
        help=f"Previous flops target to use as a starting point",
    )
    args = parser.parse_args()

    # Create the study
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{os.getenv('OPTUNA_DB_PATH')}",
        heartbeat_interval=60,
        grace_period=300,
    )

    experiment = LadderExperiment(args.flops_target, args.previous_flops_target)
    study = optuna.create_study(
        study_name=experiment.experiment_name,
        direction="minimize",
        sampler=experiment.create_sampler(),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(experiment.objective, n_trials=100)


if __name__ == "__main__":
    main()
