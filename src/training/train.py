from src.training.model import Model
from src.training.data_module import Lc0Data
import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import os
from dotenv import load_dotenv
from pathlib import Path
import yaml
from src.training.flops_logger import FlopsLogger
from typing import Optional


def train(config: dict, *, verbose: bool = False, csv_logger: Optional[CSVLogger] = None) -> dict:
    # Initialize wandb loggerloggers
    if csv_logger is None:
        csv_logger = CSVLogger(save_dir=Path(__file__).parent)
    wandb_logger = WandbLogger(
        project="chess-engine",
        name=None,
    )
    loggers = [csv_logger, wandb_logger]

    flops_logger = FlopsLogger(config)

    model = Model(config["model"])
    dataset = Lc0Data(
        config=config["training"],
        file_path=os.getenv("LEELA_DATA_PATH"),
    )
    trainer = L.Trainer(
        max_steps=config["training"]["steps"],
        log_every_n_steps=5,
        logger=loggers,
        enable_model_summary=not verbose,
        callbacks=[flops_logger],
    )
    
    trainer.fit(model, dataset)

    path = Path(os.getenv("MODELS_PATH")) / f"{wandb_logger.experiment.name}.pth"
    trainer.save_checkpoint(path)

    return trainer.logged_metrics, 

def train_with_config(config: str):
    load_dotenv()

    # Load config
    with open(Path(__file__).parent / f"configs/{config}.yaml") as f:
        config = yaml.safe_load(f)

    metrics = train(config, verbose=True)

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key:20}: {value:.4f}")

def main_debug():
    train_with_config("debug")

def main_pico():
    train_with_config("pico")
