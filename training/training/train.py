from training.model import Model
from training.data_module import Lc0Data
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os
from dotenv import load_dotenv
from pathlib import Path
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train the chess model')
    parser.add_argument(
        '--config',
        type=str,
        choices=['pico'],
        default='pico',
        help='Name of the config file to use for model hyperparameters'
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project="chess-engine",
        name=None,
    )

    model = Model(
        config_path=f"configs/{args.config}.yaml",
    )
    dataset = Lc0Data(
        file_path=os.getenv("LEELA_DATA_PATH"),
        batch_size=1024,
        shuffle_size=8192,
        sample=16,
    )
    
    trainer = L.Trainer(
        max_steps=200,
        log_every_n_steps=5,
        logger=wandb_logger,
    )
    
    trainer.fit(model, dataset)

    path = Path(os.getenv("MODELS_PATH")) / f"{wandb_logger.experiment.name}.pth"
    trainer.save_checkpoint(path)
