from pathlib import Path

import chess
import torch

from src.training.model import Model

# constants for board_to_features
PIECE_MAPPING = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}
PIECE_ENCODING = "PNBRQKpnbrqk"


def board_to_features(board: chess.Board) -> torch.Tensor:  # [772]
    """
    Convert a chess board to a 1-hot tensor of features.
    Includes both piece positions (12x8x8=768) and castling rights (4) for a total of 772 features.
    The board is always viewed from the perspective of the side to move.
    When it's Black's turn, the board is rotated 180 degrees.
    """
    # Initialize tensor for both board state and castling rights
    tensor = torch.zeros((1, 12, 8, 8), dtype=torch.float32)
    castling_tensor = torch.zeros((1, 4), dtype=torch.float32)

    # Map pieces to board tensor
    side_to_move = board.turn
    for square, piece in board.piece_map().items():
        letter = (
            PIECE_MAPPING[piece.piece_type]
            if piece.color == side_to_move
            else PIECE_MAPPING[piece.piece_type].lower()
        )
        channel = PIECE_ENCODING.index(letter)
        rank = chess.square_rank(square)
        file = chess.square_file(square)

        # If Black to move, rotate the board 180 degrees
        if not side_to_move:
            rank = 7 - rank
            file = 7 - file

        tensor[0, channel, rank, file] = 1.0

    # Set castling rights
    castling_tensor[0, 0] = float(board.has_kingside_castling_rights(side_to_move))
    castling_tensor[0, 1] = float(board.has_queenside_castling_rights(side_to_move))
    castling_tensor[0, 2] = float(board.has_kingside_castling_rights(not side_to_move))
    castling_tensor[0, 3] = float(board.has_queenside_castling_rights(not side_to_move))

    # Flatten and concatenate
    return torch.cat([tensor.flatten(), castling_tensor.flatten()])


class TrivialNet:
    def __init__(self, model_path: Path):
        self.model = Model.load_from_checkpoint(model_path)
        self.model.eval()

    def evaluate(self, board: chess.Board):
        """
        Evaluate the board from the perspective of the side to move.
        """
        result = None
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)

        if result != None:
            if result == "1/2-1/2":
                return dict(), 0.0
            else:
                # Always return -1.0 when checkmated
                # and we are checkmated because it's our turn to move
                return dict(), -1.0

        policy = self.evaluate_policy(board)
        value = self.evaluate_value(board)

        return policy, value

    def evaluate_policy(self, board: chess.Board):
        """
        Evaluate the policy of a position from the perspective of the side to move.

        Gives a trivial policy of 1/n for each legal move, where n is the number of legal moves.
        """
        legal_moves = list(board.legal_moves)
        return {move.uci(): 1 / len(legal_moves) for move in legal_moves}

    def evaluate_value(self, board: chess.Board):
        """
        Evaluate the expected value of a position using the neural network.
        """
        features = board_to_features(board)
        wdl = self.model.predict(features)
        return torch.dot(torch.tensor([1.0, 0.0, -1.0]), wdl).item()
