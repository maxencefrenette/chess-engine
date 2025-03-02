import lightning as L
import torch
import torch.nn as nn
from torch import optim


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ff = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.ff(x)
        out = self.activation(out)
        return x + out


class Model(L.LightningModule):
    def __init__(
        self,
        config: dict = None,
    ):
        super().__init__()

        if config is not None:
            self.save_hyperparameters(config)

        # Fixed architecture parameters
        self.input_dim = 780
        self.output_dim = 3

        # Model
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hparams.hidden_dim),  # input layer
            *[
                ResidualBlock(self.hparams.hidden_dim)
                for _ in range(self.hparams.hidden_layers)
            ],
            nn.Linear(self.hparams.hidden_dim, self.output_dim),  # output layer
        )

    def training_step(self, batch, batch_idx):
        (features, wdl) = batch

        y_hat = self.model(features)
        loss = nn.functional.cross_entropy(y_hat, wdl)
        self.log("train_value_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def predict(self, features: torch.Tensor):
        with torch.no_grad():
            return self.model(features)
