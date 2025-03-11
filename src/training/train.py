import os
from pathlib import Path
from typing import Optional

import lightning as L
import yaml
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger

from src.training.data_module import Lc0Data
from src.training.flops_logger import FlopsLogger
from src.training.model import Model


def load_config(config_name: str):
    with open(Path(__file__).parent / f"configs/{config_name}.yaml") as f:
        config = yaml.safe_load(f)
    return config


def train(
    config: dict,
    *,
    verbose: bool = False,
    csv_logger: Optional[CSVLogger] = None,
    accumulate_grad_batches: int = 1,
    extra_callbacks: list[L.Callback] = [],
    max_time: Optional[int] = None,
) -> dict:
    # Initialize logger
    if csv_logger is None:
        csv_logger = CSVLogger(save_dir=Path(__file__).parent)

    flops_logger = FlopsLogger(config)

    model = Model(config)
    dataset = Lc0Data(
        config=config,
        file_path=os.getenv("TRAINING_DATA_PATH"),
    )
    trainer = L.Trainer(
        max_steps=config["steps"],
        log_every_n_steps=max(1, config["steps"] // 2000),
        logger=csv_logger,
        enable_model_summary=not verbose,
        callbacks=[flops_logger] + extra_callbacks,
        accelerator=config["accelerator"],
        accumulate_grad_batches=accumulate_grad_batches,
        max_time=max_time,
    )

    trainer.fit(model, dataset)

    return trainer.logged_metrics


def train_with_config(config: str):
    load_dotenv()
    config = load_config(config)
    metrics = train(config, verbose=True)

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key:20}: {value:.4f}")


def main_debug():
    train_with_config("debug")


def main_pico():
    train_with_config("pico")
