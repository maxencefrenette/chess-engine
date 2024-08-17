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
    import numpy as np
    from src.lc0.policy_index import policy_index

    FILE_PATH = "/Users/maxence/leela-data/training-run1-test80-20240817-1917/training.1387237598.gz"

    dm = Lc0DataModule(file_path=FILE_PATH, batch_size=1)
    dm.setup("fit")
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    (planes, probs, winner, best_q, plies_left) = batch

    planes = np.frombuffer(planes, dtype=np.float32)
    probs = np.frombuffer(probs, dtype=np.float32)
    winner = np.frombuffer(winner, dtype=np.float32)
    best_q = np.frombuffer(best_q, dtype=np.float32)
    plies_left = np.frombuffer(plies_left, dtype=np.float32)

    planes = planes.reshape(-1, 112, 8, 8)
    probs = probs.reshape(-1, 1858)
    winner = winner.reshape(-1, 3)
    best_q = best_q.reshape(-1, 3)
    plies_left = plies_left.reshape(-1, 1)

    policy = zip(policy_index, list(probs[0]))
    policy = [(move, prob) for move, prob in policy if prob != -1]
    policy = sorted(policy, key=lambda x: x[1], reverse=True)

    print(planes)
    print()

    print("Policy:")
    for move, prob in policy:
        print(f"{move}: {prob}")
    print()

    print(f"Result (WDL): {winner[0]}")
    print(f"Best Q (WDL): {best_q[0]}")
    print(f"Plies left: {plies_left[0]}")
