from src.training.model import Model
from src.training.data_module import Lc0Data
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os
from dotenv import load_dotenv
from pathlib import Path
import argparse
import yaml

def train(config: dict, verbose: bool = False) -> dict:
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
        enable_model_summary=not verbose,
    )
    
    trainer.fit(model, dataset)

    path = Path(os.getenv("MODELS_PATH")) / f"{wandb_logger.experiment.name}.pth"
    trainer.save_checkpoint(path)

    return trainer.callback_metrics

def main():
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

    metrics = train(config, verbose=True)

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key:20}: {value:.4f}")
