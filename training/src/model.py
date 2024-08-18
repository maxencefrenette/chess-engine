import torch
import torch.nn as nn
import lightning as L
from torch import optim

INPUT_DIM = 12 * 8 * 8 + 4
HIDDEN_DIM = 32
HIDDEN_LAYERS = 4
OUTPUT_DIM = 3


class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
                for _ in range(HIDDEN_LAYERS)
            ],
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        )

    def training_step(self, batch, batch_idx):
        (board, castling_rights, _probs, winner, best_q, plies_left) = batch

        board = board.reshape(-1, 12 * 8 * 8)
        x = torch.cat([board, castling_rights], dim=1)

        # For now it's safe to train on best_q since this is leela data and not self-play data
        y = best_q

        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.log(
            "train_accuracy_winner",
            (y_hat.argmax(dim=1) == winner.argmax(dim=1)).float().mean(),
        )
        self.log(
            "train_accuracy_best_q",
            (y_hat.argmax(dim=1) == best_q.argmax(dim=1)).float().mean(),
        )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
