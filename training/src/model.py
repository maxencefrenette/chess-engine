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
        board, castling_rights, _probs, _winner, best_q, plies_left = batch

        board = board.reshape(-1, 12 * 8 * 8)
        x = torch.cat([board, castling_rights], dim=1)

        y_hat = self.model(x)
        # For now it's safe to train on best_q since this is leela data and not self-play data
        loss = nn.functional.cross_entropy(y_hat, best_q)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
