# Chess Engine

This is an unfinished toy chess engine that uses deep learning. I don't have a name for it yet. It is loosely based on Leela Chess Zero and trains on Leela Chess Zero's training data (supervised learning).

## Project Structure

- `docs/`: The documentation.
- `src/engine/`: The engine code.
- `src/training/`: The model definition and training pipeline.

## Running the code

- Engine: `uv run engine`
- Training
  - `uv run train_debug`
  - `uv run train_pico`

## Acknowledgements

- The search is a fork of [a0lite](https://github.com/dkappe/a0lite/tree/master)
- The model is trained on [Leela Chess Zero](https://lczero.org)'s training data
