import argparse
import os
from math import log2
from pathlib import Path

import optuna
import optunahub
import pandas as pd
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger

from src.training.experiments.utils import read_trial_results
from src.training.flops_logger import count_flops
from src.training.train import train

load_dotenv(Path(__file__).parents[4] / ".env")

experiment_name = "tune_ladder_v1"
flops_target = 1e11
previous_flops_target_experiment = ""


def objective(trial: optuna.Trial) -> tuple[float, float]:
    """Objective function for Optuna optimization."""
    # Define the hyperparameters to optimize
    config = {
        "hidden_layers": trial.suggest_int("hidden_layers", 1, 10),
        "hidden_dim": 2 ** trial.suggest_int("log2_hidden_dim", 3, 8),
        "batch_size": 2 ** trial.suggest_int("log2_batch_size", 5, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "lr_cooldown_fraction": 0.4,
        "accelerator": "gpu",
    }

    flops_per_batch = count_flops(config)
    config["steps"] = int(flops_target / flops_per_batch)

    # Create a logger for this trial
    trial_num = trial.number
    csv_logger = CSVLogger(
        save_dir=os.getenv("EXPERIMENT_LOGS_DIR"),
        name=experiment_name,
        version=trial_num,
    )
    trial.set_user_attr("logs_path", csv_logger.log_dir)

    # Train the model with these hyperparameters
    train(config, csv_logger=csv_logger, max_time={"minutes": 1})

    # Get the results
    results = read_trial_results(experiment_name, trial_num)
    train_value_loss = results.iloc[-1]["train_value_loss_ema"]

    return train_value_loss


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization")
    parser.add_argument(
        "--num-trials",
        type=int,
        help=f"Number of trials to run",
    )
    args = parser.parse_args()

    # Create the study
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{os.getenv('OPTUNA_DB_PATH')}",
        heartbeat_interval=60,
        grace_period=300,
    )

    module = optunahub.load_module(package="samplers/auto_sampler")

    study = optuna.create_study(
        study_name=f"{experiment_name}-{flops_target:.0e}",
        direction="minimize",
        sampler=module.AutoSampler(),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.num_trials)


if __name__ == "__main__":
    main()
