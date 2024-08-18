from glob import glob
import lightning as L
from src.lc0.chunkparser import ChunkParser


class Lc0DataModule(L.LightningDataModule):
    def __init__(self, file_path: str, batch_size: int):
        self.file_path = file_path
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        chunks = glob(self.file_path)
        if not chunks:
            raise ValueError(f"No chunks found at {self.file_path}")

        self.parser = ChunkParser(
            chunks=chunks,
            expected_input_format=1,
            shuffle_size=8192,
            sample=32,
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return self.parser.parse()

    def teardown(self, stage: str):
        self.file.close()


# Testing code
if __name__ == "__main__":
    import numpy as np
    from src.lc0.policy_index import policy_index

    FILE_PATH = "/Users/maxence/leela-data/*/training.*.gz"
    NUM_POSITIONS = 8

    dm = Lc0DataModule(file_path=FILE_PATH, batch_size=NUM_POSITIONS)
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

    for i in range(NUM_POSITIONS):
        # Decode board state
        PIECES = "PNBRQKpnbrqk"
        board = "        \n" * 8
        for piece_id, piece in enumerate(PIECES):
            for x in range(8):
                for y in range(8):
                    if planes[i][piece_id][y][x] == 1:
                        board = board[: x + 9 * y] + piece + board[x + 9 * y + 1 :]
        us_oo = planes[i][104][0][0]
        us_ooo = planes[i][105][0][0]
        them_oo = planes[i][106][0][0]
        them_ooo = planes[i][107][0][0]

        # Decode policy
        policy = zip(policy_index, list(probs[i]))
        policy = [(move, prob) for move, prob in policy if prob != -1]
        policy = sorted(policy, key=lambda x: x[1], reverse=True)

        print("Board:")
        print(board)
        print(f"Castling rights: {"K" if us_oo else "-"}{"Q" if us_ooo else "-"}{"k" if them_oo else "-"}{"q" if them_ooo else "-"}")

        print("Policy:")
        for move, prob in policy:
            print(f"{move}: {prob}")
        print()

        print(f"Result (WDL): {winner[i]}")
        print(f"Best Q (WDL): {best_q[i]}")
        print(f"Plies left: {plies_left[i]}")
        print("\n" * 4)
