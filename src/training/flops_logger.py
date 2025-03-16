import lightning as L
import torch
from lightning.pytorch.utilities import measure_flops

from src.training.model import Model


def count_flops(config: dict) -> float:
    """Count the flops of a given config."""
    with torch.device("meta"):
        model = Model(config)
        x = torch.randn(config["batch_size"], 780)

    model_fwd = lambda: model.model(x)
    model_loss = lambda y: y.sum()
    return measure_flops(model, model_fwd, model_loss)


class FlopsLogger(L.Callback):
    def __init__(self, config):
        super().__init__()

        self.flops_per_batch = count_flops(config)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pl_module.log("flops", self.flops_per_batch * (1 + batch_idx))
