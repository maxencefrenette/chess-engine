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
from src.training.train import train

load_dotenv(Path(__file__).parents[4] / ".env")

num_trials = 5
experiment_name = "tune_v5"


def objective(trial: optuna.Trial) -> tuple[float, float]:
    """Objective function for Optuna optimization."""
    # Define the hyperparameters to optimize
    config = {
        "hidden_layers": trial.suggest_int("hidden_layers", 1, 10),
        "hidden_dim": 2 ** trial.suggest_int("log2_hidden_dim", 3, 8),
        "batch_size": 2 ** trial.suggest_int("log2_batch_size", 1, 7),
        "steps": trial.suggest_int("steps", 5000, 100000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "lr_cooldown_fraction": trial.suggest_float("lr_cooldown_fraction", 0.0, 0.6),
        "accelerator": "cpu",
    }

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
    flops = results.iloc[-1]["flops"]

    return flops, train_value_loss


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization")
    parser.add_argument(
        "--num-trials",
        type=int,
        default=num_trials,
        help=f"Number of trials to run (default: {num_trials})",
    )
    args = parser.parse_args()

    # Create a new study
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{os.getenv('OPTUNA_DB_PATH')}",
        heartbeat_interval=60,
        grace_period=300,
    )

    module = optunahub.load_module(package="samplers/auto_sampler")

    study = optuna.create_study(
        study_name=experiment_name,
        directions=["minimize", "minimize"],  # For flops and train_value_loss
        sampler=module.AutoSampler(),
        storage=storage,
        load_if_exists=True,
    )

    # Run the optimization
    study.optimize(objective, n_trials=args.num_trials)

    print("\nPareto Frontier:")
    print("Trial    FLOPS           Train Loss    Parameters")
    print("-" * 70)

    best_trials = study.best_trials
    best_trials = sorted(best_trials, key=lambda x: x.values[0])

    for trial in best_trials:
        flops, loss = trial.values
        params = trial.params
        print(
            f"{trial.number:<8} {flops:>.2e}    {loss:.4f}        "
            f"lr={params['learning_rate']:.1e}, "
            f"layers={params['hidden_layers']}, "
            f"dim={2**params['log2_hidden_dim']}"
        )

    print(f"\nCompleted {args.num_trials} trials for experiment '{experiment_name}'.")


if __name__ == "__main__":
    main()
