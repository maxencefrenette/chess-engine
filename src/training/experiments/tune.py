import os
from math import log2
from pathlib import Path

import optuna
import pandas as pd
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger

from src.training.train import train

load_dotenv(Path(__file__).parents[3] / ".env")

num_trials = 5
experiment_name = "tune"


def read_trial_results(experiment_name: str, version: int) -> pd.DataFrame:
    """Read a single trial result from the given path."""
    metrics_path = (
        Path(os.getenv("EXPERIMENT_LOGS_DIR"))
        / experiment_name
        / f"version_{version}"
        / "metrics.csv"
    )

    return pd.read_csv(metrics_path)


def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna optimization."""
    # Define the hyperparameters to optimize
    config = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "hidden_layers": trial.suggest_int("hidden_layers", 1, 10),
        "hidden_dim": 2 ** trial.suggest_int("log2_hidden_dim", 2, 6),
        "batch_size": 32,
        "steps": trial.suggest_int("steps", 100, 1000, log=True),
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
    train(config, csv_logger=csv_logger)

    # Get the results
    results = read_trial_results(experiment_name, trial_num)
    train_value_loss = results.iloc[-1]["train_value_loss"]
    flops = results.iloc[-1]["flops"]

    return flops, train_value_loss


def main():
    # Create a new study
    study = optuna.create_study(
        study_name=experiment_name,
        directions=["minimize", "minimize"],  # For flops and train_value_loss
        storage=f"sqlite:///{os.getenv('OPTUNA_DB_PATH')}",
        load_if_exists=True,
    )

    # Run the optimization
    study.optimize(objective, n_trials=num_trials)

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


if __name__ == "__main__":
    main()
