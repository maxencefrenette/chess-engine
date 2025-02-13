from engine.search.trivial_net import board_to_features
import chess
import torch


def test_board_to_features():
    # Test starting position
    board = chess.Board()
    features = board_to_features(board)

    # Test shape
    assert isinstance(features, torch.Tensor)
    assert features.shape == (772,)  # 12x8x8 + 4 castling rights

    # Test piece positions for starting position (White to move)
    piece_features = features[:768].reshape(12, 8, 8)

    # Test white pawns (channel 0) from White's perspective
    assert piece_features[0, 1].sum() == 8  # All white pawns on rank 2

    # Test black pawns (channel 6) from White's perspective
    assert piece_features[6, 6].sum() == 8  # All black pawns on rank 7

    # Test castling rights
    castling_rights = features[768:]
    assert castling_rights.tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]  # All castling rights available

    # Test a position with some moves
    board.push_san("e4")  # 1.e4 - now Black to move
    features = board_to_features(board)
    piece_features = features[:768].reshape(12, 8, 8)

    # From Black's perspective (board rotated 180°):
    # - The white pawn on e4 is an opponent's piece, so it should be in channel 6 (first black piece channel)
    # - After 180° rotation, e4 becomes d5 (file e->d, rank 4->5)
    assert (
        piece_features[6, 4, 3] == 1
    )  # White pawn appears as opponent piece on d5 from Black's view
