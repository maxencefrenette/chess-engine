import time
from pathlib import Path
from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from src.training.model import Model


class RandomDataset(Dataset):
    """Dataset that generates random chess positions"""
    def __init__(self, num_samples: int, batch_size: int):
        self.num_samples = num_samples
        self.batch_size = batch_size
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Generate random data matching the model's input dimensions
        # board: 12 planes of 8x8
        # castling rights: 4 binary values
        # best_q: 3 possible outcomes (win/draw/loss)
        board = torch.randn(12, 8, 8)
        castling_rights = torch.randint(0, 2, (4,)).float()
        best_q = torch.zeros(3)
        best_q[torch.randint(0, 3, (1,))] = 1.0
        
        # Dummy values for unused tensors (_probs, winner, plies_left)
        _probs = torch.zeros(3)
        winner = torch.zeros(3)
        plies_left = torch.tensor(0)
        
        return board, castling_rights, _probs, winner, best_q, plies_left

def load_config(config_name: str = "debug") -> dict:
    """Load configuration from yaml file"""
    config_path = Path(__file__).parents[2] / "training" / "configs" / f"{config_name}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def benchmark_model(config: dict, num_batches: int = 100, force_cpu: bool = False) -> dict:
    """
    Benchmark a model configuration using random data.
    Returns statistics about the training speed.
    
    Args:
        config: Model configuration dictionary
        num_batches: Number of batches to process
        force_cpu: If True, forces CPU usage regardless of available hardware
    """
    # Time setup
    setup_start = time.time()
    model = Model(config=config)
    
    # Create random dataset
    batch_size = config["batch_size"]
    dataset = RandomDataset(num_samples=num_batches * batch_size, batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    
    # Create trainer
    trainer = L.Trainer(
        max_steps=num_batches,
        accelerator="cpu" if force_cpu else "auto",
        devices="auto",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    )
    setup_time = time.time() - setup_start
    
    # Time training
    train_start = time.time()
    trainer.fit(model, dataloader)
    train_time = time.time() - train_start
    
    total_positions = num_batches * batch_size
    avg_batch_time = train_time / num_batches
    throughput = total_positions / train_time
    
    return {
        "setup_time": setup_time,
        "total_time": train_time,
        "avg_batch_time": avg_batch_time,
        "throughput": throughput,
        "total_positions": total_positions,
        "accelerator": "cpu" if force_cpu else "auto"
    }

def benchmark_training(num_batches: int = 100):
    """
    Benchmark the training speed of different model configurations using random data.
    Tests each configuration with both default accelerator and forced CPU.
    """
    # Benchmark both configurations
    configs = ["debug", "pico"]
    
    for config_name in configs:
        config = load_config(config_name)
        print(f"\nBenchmarking {config_name} configuration:")
        print(f"Configuration: {config}")
        
        for force_cpu in [False, True]:
            mode = "CPU (forced)" if force_cpu else "Default accelerator"
            print(f"\nTesting with {mode}:")
            try:
                stats = benchmark_model(config, num_batches, force_cpu)
                
                print("\nResults:")
                print(f"Accelerator: {stats['accelerator']}")
                print(f"Model setup time: {stats['setup_time']:.2f} seconds")
                print(f"Total time: {stats['total_time']:.2f} seconds")
                print(f"Average time per batch: {stats['avg_batch_time']:.4f} seconds")
                print(f"Throughput: {stats['throughput']:.0f} positions/second")
                print(f"Total positions processed: {stats['total_positions']}")
                print()
                print()
                
            except Exception as e:
                print(f"Error during benchmarking with {mode}: {e}")

if __name__ == "__main__":
    benchmark_training() 
