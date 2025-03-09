#!/usr/bin/env python3
import os
import sys
from math import log2
from pathlib import Path

import optuna
from dotenv import load_dotenv
from optuna.distributions import IntDistribution

load_dotenv(Path(__file__).parents[3] / ".env")


def migrate_study(source_db_path, target_db_path, study_name="tune"):
    """
    Migrate trials from one Optuna study to another, with custom parameter mapping.

    Args:
        source_db_path: Path to the source SQLite database
        target_db_path: Path to the target SQLite database
        study_name: Name of the source study (default: "tune")
    """
    # Load source study
    source_study = optuna.load_study(
        study_name=study_name, storage=f"sqlite:///{source_db_path}"
    )

    # Create target study
    target_study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{target_db_path}",
        directions=source_study.directions,
        load_if_exists=False,
    )

    # Migrate each trial
    for trial in source_study.get_trials(deepcopy=False):
        # ============================================================
        # Custom parameter mapping code goes here
        # This section should be modified for each migration

        # Example:
        # if "old_param" in trial.params:
        #     trial.params["new_param"] = trial.params.pop("old_param")

        # Add your custom mapping code below:

        if "hidden_dim" in trial.distributions:
            trial.params["log2_hidden_dim"] = int(log2(trial.params["hidden_dim"]))
            trial.distributions["log2_hidden_dim"] = IntDistribution(2, 6)

            del trial.params["hidden_dim"]
            del trial.distributions["hidden_dim"]

        # ============================================================

        target_study.add_trial(trial)

    return target_study


if __name__ == "__main__":
    # Default paths
    source_db = os.getenv("OPTUNA_DB_PATH")
    target_db = os.getenv("OPTUNA_DB_PATH").replace(".db", "_migrated.db")

    # Ensure source DB exists
    if not os.path.exists(source_db):
        print(f"Error: Source database '{source_db}' not found")
        sys.exit(1)

    # Run migration
    print(f"Migrating from '{source_db}' to '{target_db}'...")
    study = migrate_study(source_db, target_db)
    print("Migration complete!")
