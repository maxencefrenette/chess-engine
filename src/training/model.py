import lightning as L
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

EMA_DECAY = 0.999


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ff = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.ff(x)
        out = self.activation(out)
        return x + out


def lr_schedule_fn(total_steps, cooldown_fraction, current_step):
    """
    Linear decay from 1.0 to 0.0 over the last cooldown_fraction of the total steps.
    """
    if current_step < total_steps * (1 - cooldown_fraction):
        return 1.0
    else:
        return 1.0 - (current_step - total_steps * (1 - cooldown_fraction)) / (
            total_steps * cooldown_fraction
        )


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

        self.train_value_loss_ema = None

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

        if self.train_value_loss_ema is None:
            self.train_value_loss_ema = loss
        else:
            self.train_value_loss_ema = (
                EMA_DECAY * self.train_value_loss_ema + (1 - EMA_DECAY) * loss
            )
        self.log("train_value_loss_ema", self.train_value_loss_ema)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_schedule_fn(
                self.hparams.steps, self.hparams.lr_cooldown_fraction, step
            ),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def predict(self, features: torch.Tensor):
        with torch.no_grad():
            return self.model(features)
