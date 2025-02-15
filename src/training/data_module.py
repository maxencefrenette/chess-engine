from glob import glob
import warnings
import torch
import lightning as L
from src.training.lc0.chunkparser import ChunkParser


class Lc0Data(L.LightningDataModule):
    def __init__(self, file_path: str, batch_size: int, shuffle_size: int, sample: int):
        super().__init__()
        self.file_path = file_path
        self.save_hyperparameters(ignore=["file_path"])

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        chunks = glob(self.file_path)
        if not chunks:
            raise ValueError(f"No chunks found at {self.file_path}")

        self.parser = ChunkParser(
            chunks=chunks,
            expected_input_format=1,
            shuffle_size=self.hparams.shuffle_size,
            sample=self.hparams.sample,
            batch_size=self.hparams.batch_size,
        )

    def train_dataloader(self):
        for batch in self.parser.parse():
            planes, probs, winner, best_q, plies_left = batch

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                planes = torch.frombuffer(planes, dtype=torch.float32)
                probs = torch.frombuffer(probs, dtype=torch.float32)
                winner = torch.frombuffer(winner, dtype=torch.float32)
                best_q = torch.frombuffer(best_q, dtype=torch.float32)
                plies_left = torch.frombuffer(plies_left, dtype=torch.float32)

            planes = planes.reshape(-1, 112, 8, 8)
            probs = probs.reshape(-1, 1858)
            winner = winner.reshape(-1, 3)
            best_q = best_q.reshape(-1, 3)
            plies_left = plies_left.reshape(-1, 1)

            board = planes[:, :12, :, :]
            castling_rights = planes[:, 104:108, 0, 0]

            yield (board, castling_rights, probs, winner, best_q, plies_left)


# Testing code
if __name__ == "__main__":
    from training.lc0.policy_index import policy_index

    FILE_PATH = "/Users/maxence/leela-data/*/training.*.gz"
    NUM_POSITIONS = 8

    dm = Lc0Data(file_path=FILE_PATH, batch_size=NUM_POSITIONS)
    dm.setup("fit")
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    (board, castling_rights, probs, winner, best_q, plies_left) = batch

    for i in range(NUM_POSITIONS):
        print("Board:")
        PIECES = "PNBRQKpnbrqk"
        for x in range(8):
            for y in range(8):
                for piece_id, piece in enumerate(PIECES):
                    if board[i][piece_id][x][y] == 1:
                        print(piece, end="")
                        break
                else:
                    print(".", end="")
            print()

        [us_oo, us_ooo, them_oo, them_ooo] = castling_rights[i]
        print(
            "Castling rights: "
            f"{'K' if us_oo else '-'}"
            f"{'Q' if us_ooo else '-'}"
            f"{'k' if them_oo else '-'}"
            f"{'q' if them_ooo else '-'}"
        )
        print()

        print("Policy:")
        policy = zip(policy_index, list(probs[i]))
        policy = [(move, prob) for move, prob in policy if prob != -1]
        policy = sorted(policy, key=lambda x: x[1], reverse=True)
        for move, prob in policy:
            print(f"{move}: {prob:.2f}")
        print()

        print(f"Result (WDL): {winner[i]}")
        print(f"Best Q (WDL): {best_q[i]}")
        print(f"Plies left: {plies_left[i]}")
        print("\n" * 4)
