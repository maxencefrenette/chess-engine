# Chess Engine

This is an unfinished toy chess engine that uses deep learning. I don't have a name for it yet. It is loosely based on Leela Chess Zero and trains on Leela Chess Zero's training data (supervised learning).

## Project Structure

- `benchmarks/`: Scripts to benchmark the elo of the engine.
 - `docs/`: Project documentation (see [docs/index.md](docs/index.md)).
- `src/engine/`: The engine code.
- `src/training/`: The model definition and training pipeline.

 .
 ├── benchmarks/            # Elo benchmarking scripts
 ├── docs/                  # Project documentation
 ├── src/                   # Source code
 │   ├── preprocessing/     # LC0 to .npz conversion scripts
 │   ├── engine/            # Chess engine implementation
 │   └── training/          # Model definition and training pipeline
 ├── tests/                 # Unit and integration tests
 └── Other directories (checkpoints, dist, optunahub) for ancillary data and packages.

 ## Running the Code

 - Engine: `uv run engine`

   For full usage details, see [docs/usage.md](docs/usage.md).
- Training
  - `uv run train_debug`
  - `uv run train_pico`

## Acknowledgements

- The search is a fork of [a0lite](https://github.com/dkappe/a0lite/tree/master)
- The model is trained on [Leela Chess Zero](https://lczero.org)'s training data
