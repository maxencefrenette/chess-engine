import torch
import torch.nn as nn
import lightning as L
from torch import optim


class Model(L.LightningModule):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Fixed architecture parameters
        self.input_dim = 12 * 8 * 8 + 4  # board state + castling rights
        self.output_dim = 3
        
        # Model
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hparams.hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
                    nn.ReLU()
                )
                for _ in range(self.hparams.hidden_layers)
            ],
            nn.Linear(self.hparams.hidden_dim, self.output_dim)
        )

    def training_step(self, batch, batch_idx):
        (board, castling_rights, _probs, winner, best_q, plies_left) = batch

        board = board.reshape(-1, 12 * 8 * 8)
        x = torch.cat([board, castling_rights], dim=1)

        # For now it's safe to train on best_q since we're doing supervised learning
        # on leela data and not reinforcement learning on self-play data
        y = best_q

        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_value_loss", loss)
        self.log(
            "train_value_accuracy",
            (y_hat.argmax(dim=1) == best_q.argmax(dim=1)).float().mean(),
        )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def predict(self, features: torch.Tensor):
        with torch.no_grad():
            return self.model(features)
