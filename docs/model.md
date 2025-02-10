# Model

## Inputs

- Board: A 1-hot tensor of dimensions (batch_size, 12, 8, 8) representing the input board. Pieces are encoded in this order: "PNBRQKpnbrqk", where capital letters represent the side to move and lowercase letters represent the side to move next.
- Castling rights: A tensor of dimensions (batch_size, 4) representing the castling rights of the current position. The order of the castling rights is "KQkq".
- En passant (TODO): A tensor of dimensions (batch_size, 8) representing the en passant squares of the current position.

## Outputs

- Value: A tensor of dimensions (batch_size, 3) representing the expected outcome of the game as WDL probabilities.
- Policy (TODO): A tensor of dimensions (batch_size, 1858) representing the expoected probabilities of each move being the best move.
