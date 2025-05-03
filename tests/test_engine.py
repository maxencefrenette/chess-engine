import chess
import pytest

from src.engine.engine import process_position


def test_position_startpos():
    # 'position startpos' should yield the initial chess position
    tokens = ["position", "startpos"]
    board = process_position(tokens)
    # Compare full FEN (includes halfmove/fullmove counters)
    assert board.fen() == chess.Board().fen()


def test_position_startpos_moves():
    # 'position startpos moves e2e4 e7e5' applies two pawn moves
    tokens = ["position", "startpos", "moves", "e2e4", "e7e5"]
    board = process_position(tokens)
    # White pawn on e4
    piece_e4 = board.piece_at(chess.E4)
    assert piece_e4 is not None and piece_e4.piece_type == chess.PAWN
    assert piece_e4.color == chess.WHITE
    # Black pawn on e5
    piece_e5 = board.piece_at(chess.E5)
    assert piece_e5 is not None and piece_e5.piece_type == chess.PAWN
    assert piece_e5.color == chess.BLACK
    # After two half-moves, it is White's turn
    assert board.turn == chess.WHITE
    # Pawn moves reset the halfmove clock to zero
    assert board.halfmove_clock == 0


def test_position_fen_only():
    # 'position fen <FEN>' should load exactly that FEN
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq e3 42 10"
    tokens = ["position", "fen"] + fen.split()
    board = process_position(tokens)
    # Verify piece placement, side to move, castling rights, and move counters
    got_fields = board.fen().split()
    fen_fields = fen.split()
    # Piece placement
    assert got_fields[0] == fen_fields[0]
    # Side to move
    assert got_fields[1] == fen_fields[1]
    # Castling rights
    assert got_fields[2] == fen_fields[2]
    # Halfmove and fullmove counters
    assert got_fields[4] == fen_fields[4]
    assert got_fields[5] == fen_fields[5]


def test_position_fen_and_moves():
    # Starting from a custom FEN, apply simple king moves
    fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    tokens = ["position", "fen"] + fen.split() + ["moves", "a1b1", "h1g1"]
    board = process_position(tokens)
    # White king from a1 to b1
    king_b1 = board.piece_at(chess.B1)
    assert king_b1 is not None and king_b1.piece_type == chess.KING
    assert king_b1.color == chess.WHITE
    # Black king from h1 to g1
    king_g1 = board.piece_at(chess.G1)
    assert king_g1 is not None and king_g1.piece_type == chess.KING
    assert king_g1.color == chess.BLACK
    # Two halfmoves, back to White's turn
    assert board.turn == chess.WHITE


def test_draw_claimable_clears_stack():
    # Generate 100 non-capturing, non-pawn moves to trigger fifty-move rule
    moves = []
    # Repeat a 4-move cycle 25 times = 100 halfmoves
    for _ in range(25):
        moves.extend(["g1f3", "g8h6", "f3g1", "h6g8"])
    tokens = ["position", "startpos", "moves"] + moves
    board = process_position(tokens)
    # Should now be able to claim a draw by fifty-move rule
    assert board.can_claim_draw(), "Expected fifty-move draw available"
    # process_position should clear the move stack when a draw can be claimed
    assert board.move_stack == [], "Expected move stack to be cleared"
