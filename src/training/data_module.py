from glob import glob

import lightning as L
import numpy as np
import torch


class Lc0Data(L.LightningDataModule):
    def __init__(self, config: dict, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.save_hyperparameters(config)

    def train_dataloader(self):
        self.chunks = sorted(glob(self.file_path))
        batch_size = self.hparams.batch_size

        for chunk in self.chunks:
            chunk_data = np.load(chunk)
            features = chunk_data["features"]
            best_q = chunk_data["best_q"]
            assert len(features) == len(best_q)

            # Output batches of config.batch_size and skip the last little bit if it's not a full batch
            for i in range(0, len(features), batch_size):
                yield (
                    torch.from_numpy(features[i : i + batch_size]),
                    torch.from_numpy(best_q[i : i + batch_size]),
                )
