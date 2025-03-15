import concurrent.futures
import os
from pathlib import Path

import marimo as mo
import pandas as pd
import yaml
from lightning.pytorch.loggers import CSVLogger

from src.training.train import train


def read_experiment_results(experiment_name: str) -> pd.DataFrame:
    """
    Read experiment results from the experiment logs directory.

    Args:
        experiment_name: Name of the experiment (e.g., "learning_rate", "batch_size", "steps")

    Returns:
        A DataFrame with all experiment results
    """
    # Default to the experiment_logs directory relative to this file
    base_path = Path(__file__).parents[1] / "experiment_logs"

    path = base_path / experiment_name

    if not path.exists():
        print(f"Warning: experiment directory '{experiment_name}' not found at {path}")
        return pd.DataFrame()

    results = []

    # Directly iterate through all directories in the experiment path
    for d in path.glob("*"):
        if not d.is_dir():
            continue

        metrics_path = d / "metrics.csv"
        hparams_path = d / "hparams.yaml"

        if not metrics_path.exists():
            print(f"Warning: metrics file not found at {metrics_path}")
            continue

        # Read hyperparameters if available
        hparams = {}
        if hparams_path.exists():
            with open(hparams_path) as f:
                hparams = yaml.safe_load(f)

        # Read metrics
        metrics_df = pd.read_csv(metrics_path)

        # Extract config name from directory name
        # Assuming format like "config_name" or "config_name_version"
        dir_parts = d.name.split("_")
        metrics_df["config"] = dir_parts[0]

        # Add hyperparameters to metrics dataframe
        for param_name, param_value in hparams.items():
            metrics_df[param_name] = param_value

        # Add to results
        results.append(metrics_df)

    # Create results dataframe if we have results
    results_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    return results_df


def smooth_column(
    df: pd.DataFrame, column_name: str, window_size: int = 300, group_by: str = "config"
) -> pd.DataFrame:
    """
    Apply rolling mean smoothing to a column in a dataframe.

    Args:
        df: DataFrame containing the data
        column_name: Name of the column to smooth
        window_size: Size of the rolling window
        group_by: Column name to group by before smoothing

    Returns:
        DataFrame with the smoothed column
    """
    df_copy = df.copy()
    df_copy[column_name] = df_copy.groupby(group_by)[column_name].transform(
        lambda x: x.rolling(window_size, center=True, closed="both").mean()
    )
    return df_copy


def run_single_experiment(experiment_name: str, trial_name: str, config: dict):
    """
    Run a single experiment with the given configuration.

    Args:
        experiment_name: Name of the experiment (e.g., "learning_rate", "batch_size")
        trial_name: Unique identifier for this specific trial
        config: Configuration dictionary to pass to the train function
    """
    csv_logger = CSVLogger(
        save_dir=Path(__file__).parents[1] / "experiment_logs",
        name=experiment_name,
        version=trial_name,
    )

    train(config, csv_logger=csv_logger)
