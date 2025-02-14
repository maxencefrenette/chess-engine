from training.model import Model
from training.data_module import Lc0Data
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os
from dotenv import load_dotenv
from pathlib import Path
import argparse
import yaml

def train(config: dict):
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project="chess-engine",
        name=None,
    )

    model = Model(config)
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

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description='Train the chess model')
    parser.add_argument(
        '--config',
        type=str,
        default='pico',
        help='Name of the config file to use for model hyperparameters'
    )
    args = parser.parse_args()

    # Load config
    with open(Path(__file__).parent / f"configs/{args.config}.yaml") as f:
        config = yaml.safe_load(f)["model"]

    train(config)
