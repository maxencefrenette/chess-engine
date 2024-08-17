import os
from torch.utils.data import DataLoader, IterableDataset
import lightning as L
from src.lc0.chunkparser import ChunkParser


class Lc0DataModule(L.LightningDataModule):
    def __init__(self, file_path: str, batch_size: int):
        self.file_path = file_path
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.parser = ChunkParser(
            chunks=[self.file_path],
            expected_input_format=1,
            shuffle_size=1024,
            sample=1,
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return Lc0Dataset(self.parser)

    def teardown(self, stage: str):
        self.file.close()


class Lc0Dataset(IterableDataset):
    def __init__(self, parser: ChunkParser):
        self.parser = parser

    def __iter__(self):
        return self.parser.parse()


# Testing code
if __name__ == "__main__":
    FILE_PATH = "/Users/maxence/leela-data/training-run1-test80-20240817-1917/training.1387237598.gz"

    dm = Lc0DataModule(file_path=FILE_PATH, batch_size=2)
    dm.setup("fit")
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    print(batch)
