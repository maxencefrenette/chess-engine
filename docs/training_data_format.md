# Training Data Format

This document describes the format of the training data used by the chess engine.

## File Format

The training data is stored in `.npz` format (NumPy compressed archives), which offers several advantages over the LC0 format:

1. Faster loading times
2. Native integration with NumPy/PyTorch
3. Simpler structure for easier manipulation
4. Lower storage requirements through more efficient compression

## Data Structure

Each `.npz` file contains the following arrays:

| Array Name | Shape | Description |
|------------|-------|-------------|
| `features` | `(N, 780)` | Board position features, where N is the number of positions |
| `best_q` | `(N, 3)` | Win-Draw-Loss probabilities for each position |

### Features Format

The features array uses the same format as our model input, with dimensions:

- 12 piece planes (6 piece types × 2 colors) in 8×8 board positions (768 features)
- 4 castling rights (kingside/queenside for both sides) (4 features)
- 8 en passant possible files (one-hot encoded, indicating which file has en passant available) (8 features)

Total: 780 features per position

Note: En passant information is calculated by comparing consecutive positions in the game to detect double pawn moves.

For efficient storage, the features are stored in a packed bit format where:

- `1` indicates the presence of a piece or castling right
- `0` indicates the absence of a piece or castling right

The board is always viewed from the perspective of the side to move. When it's Black's turn, the board is mirrored vertically.

### Best Q Format

The `best_q` array contains three probabilities for each position:

- Index 0: Win probability
- Index 1: Draw probability
- Index 2: Loss probability

These probabilities sum to 1.0 and represent the expected outcome of the game from the perspective of the side to move, according to the engine's evaluation.

## Conversion

A conversion script (`lc0_to_npz.py`) is provided to convert LC0 training data to this format. The script handles the following tasks:

1. Reading LC0 chunk files
2. Extracting board positions and evaluations
3. Converting to the packed bit format
4. Saving as compressed `.npz` files
