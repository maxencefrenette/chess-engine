import argparse
import os
from math import log2
from pathlib import Path

import numpy as np
import optuna
import optunahub
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger

from src.training.experiments.utils import read_trial_results
from src.training.flops_logger import count_flops
from src.training.train import train

load_dotenv(Path(__file__).parents[4] / ".env")

flops_target = 3e12
experiment_name = f"tune_ladder_v1-{flops_target:.0e}"
previous_flops_target_experiment = "tune_ladder_v1-1e+12"

if previous_flops_target_experiment is not None:
    # TODO: refactor this to not be in the global scope
    study = optuna.load_study(
        study_name=previous_flops_target_experiment,
        storage=f"sqlite:///{os.getenv('OPTUNA_DB_PATH')}",
    )
    best_trials = study.trials_dataframe()
    best_trials = best_trials.nsmallest(4, "value")
    best_trials = best_trials[
        best_trials["params_log2_batch_size"]
        == best_trials["params_log2_batch_size"].max()
    ]
    best_trial = best_trials.nsmallest(1, "value").iloc[0]

    hidden_layers = int(best_trial["params_hidden_layers"])
    log2_hidden_dim = int(best_trial["params_log2_hidden_dim"])
    log2_batch_size = int(best_trial["params_log2_batch_size"])
    learning_rate = float(best_trial["params_learning_rate"])


def objective(trial: optuna.Trial) -> tuple[float, float]:
    """Objective function for Optuna optimization."""
    # Define the hyperparameters to optimize
    if previous_flops_target_experiment is not None:
        config = {
            "hidden_layers": trial.suggest_int(
                "hidden_layers", hidden_layers, hidden_layers + 1
            ),
            "hidden_dim": 2
            ** trial.suggest_int(
                "log2_hidden_dim", log2_hidden_dim, log2_hidden_dim + 1
            ),
            "batch_size": 2
            ** trial.suggest_int(
                "log2_batch_size", log2_batch_size, log2_batch_size + 1
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", learning_rate * 0.5, learning_rate * 2, log=True
            ),
            "lr_cooldown_fraction": 0.4,
            "accelerator": "gpu",
        }
    else:
        config = {
            "hidden_layers": trial.suggest_int("hidden_layers", 1, 10),
            "hidden_dim": 2 ** trial.suggest_int("log2_hidden_dim", 3, 8),
            "batch_size": 2 ** trial.suggest_int("log2_batch_size", 5, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "lr_cooldown_fraction": 0.4,
            "accelerator": "gpu",
        }

    flops_per_batch = count_flops(config)
    steps = int(flops_target / flops_per_batch)
    config["steps"] = steps
    trial.set_user_attr("steps", steps)

    # Create a logger for this trial
    trial_num = trial.number
    csv_logger = CSVLogger(
        save_dir=os.getenv("EXPERIMENT_LOGS_DIR"),
        name=experiment_name,
        version=trial_num,
    )
    trial.set_user_attr("logs_path", csv_logger.log_dir)

    # Train the model with these hyperparameters
    train(config, csv_logger=csv_logger)

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

    if previous_flops_target_experiment is None:
        module = optunahub.load_module(package="samplers/auto_sampler")
        sampler = module.AutoSampler()
    else:
        search_space = {
            "learning_rate": np.geomspace(
                learning_rate * 0.5, learning_rate * 2, 5
            ).tolist(),
            "hidden_layers": [hidden_layers, hidden_layers + 1],
            "log2_hidden_dim": [log2_hidden_dim, log2_hidden_dim + 1],
            "log2_batch_size": [log2_batch_size, log2_batch_size + 1],
        }
        sampler = optuna.samplers.GridSampler(search_space)

    study = optuna.create_study(
        study_name=experiment_name,
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.num_trials)


if __name__ == "__main__":
    main()
