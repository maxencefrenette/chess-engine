import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger
from vizier import service
from vizier.service import clients
from vizier.service import pyvizier as vz

from src.training.train import train

load_dotenv(Path(__file__).parents[3] / ".env")

num_trials = 10
experiment_name = "tune_1e10"


def read_trial_results(experiment_name: str, version: int) -> pd.DataFrame:
    """Read a single trial result from the given path."""
    metrics_path = (
        Path(os.getenv("EXPERIMENT_LOGS_DIR"))
        / experiment_name
        / f"version_{version}"
        / "metrics.csv"
    )

    return pd.read_csv(metrics_path)


def main():
    problem = vz.ProblemStatement()
    problem_root = problem.search_space.root
    problem_root.add_float_param(name="learning_rate", min_value=1e-5, max_value=1e-1)
    problem_root.add_int_param(name="hidden_layers", min_value=1, max_value=10)
    problem_root.add_discrete_param(name="hidden_dim", feasible_values=[4, 8, 16, 32])
    problem.metric_information.append(
        vz.MetricInformation(
            name="train_value_loss", goal=vz.ObjectiveMetricGoal.MINIMIZE
        )
    )

    study_config = vz.StudyConfig.from_problem(problem)
    study_config.algorithm = "DEFAULT"

    study_client = clients.Study.from_study_config(
        study_config, owner="maxence", study_id=experiment_name
    )
    print("Local SQL database file located at: ", service.VIZIER_DB_PATH)

    for i in range(num_trials):
        suggestion = study_client.suggest(count=1)[0]

        config = {
            **suggestion.parameters,
            "hidden_layers": int(suggestion.parameters["hidden_layers"]),
            "batch_size": 32,
            "steps": 500,
            "accelerator": "cpu",
        }

        csv_logger = CSVLogger(
            save_dir=os.getenv("EXPERIMENT_LOGS_DIR"),
            name=experiment_name,
            version=i,
        )
        train(config, csv_logger=csv_logger)

        results = read_trial_results(experiment_name, i)
        train_value_loss = results.iloc[-1]["train_value_loss"]
        print(f"Trial {i} completed with train_value_loss: {train_value_loss:.3f}")

        final_measurement = vz.Measurement({"train_value_loss": train_value_loss})
        suggestion.complete(final_measurement)

    for optimal_trial in study_client.optimal_trials():
        optimal_trial = optimal_trial.materialize()
        print(
            "Optimal Trial Suggestion and Objective:",
            optimal_trial.parameters,
            optimal_trial.final_measurement,
        )


if __name__ == "__main__":
    main()
