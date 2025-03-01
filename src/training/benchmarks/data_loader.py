import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.training.data_module import Lc0Data


def benchmark_data_loading(num_batches: int = 100):
    """
    Benchmark the data loading speed by measuring:
    1. Time to setup the data loader
    2. Time to load and process num_batches batches
    3. Average time per batch
    4. Throughput (positions/second)
    """
    load_dotenv()
    
    # Configuration matching training settings
    config = {
        "batch_size": 1024,
        "shuffle_size": 1024,
        "sample": 1,
    }
    
    file_path = os.getenv("LEELA_DATA_PATH")
    if not file_path:
        raise ValueError("LEELA_DATA_PATH environment variable not set")
    
    print(f"Benchmarking data loading from {file_path}")
    print(f"Configuration: {config}")
    
    # Time setup
    setup_start = time.time()
    dm = Lc0Data(config=config, file_path=file_path)
    dm.setup("fit")
    setup_time = time.time() - setup_start
    print(f"\nSetup time: {setup_time:.2f} seconds")
    
    try:
        # Time batch loading
        dl = dm.train_dataloader()
        total_positions = 0
        batch_times = []
        
        print(f"\nLoading {num_batches} batches...")
        for i in range(num_batches):
            batch_start = time.time()
            batch = next(iter(dl))
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Count actual positions in this batch
            total_positions += batch[0].shape[0]  # board tensor's batch dimension
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} batches...")
        
        # Calculate statistics
        total_time = sum(batch_times)
        avg_batch_time = total_time / num_batches
        throughput = total_positions / total_time
        
        print("\nResults:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per batch: {avg_batch_time:.4f} seconds")
        print(f"Throughput: {throughput:.0f} positions/second")
        print(f"Total positions processed: {total_positions}")
    finally:
        # Ensure we properly clean up the worker processes
        if hasattr(dm.parser, 'shutdown'):
            dm.parser.shutdown()

if __name__ == "__main__":
    benchmark_data_loading() 
