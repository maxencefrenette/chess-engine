from src.model import Model
from src.data_module import Lc0Data
import lightning as L


if __name__ == "__main__":
    model = Model()
    dataset = Lc0Data(
        file_path="/Users/maxence/leela-data/*/training.*.gz",
        batch_size=1024,
    )
    trainer = L.Trainer(
        max_steps=200,
        log_every_n_steps=5,
    )
    trainer.fit(model, dataset)
