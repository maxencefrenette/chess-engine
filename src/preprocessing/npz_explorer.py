#!/usr/bin/env python3
"""
NPZ Explorer

This script visualizes chess positions from NPZ training data files.
It helps verify the conversion from LC0 to NPZ format and inspect the
calculated en passant features.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import chess
import numpy as np


def create_board_from_features(
    feature: np.ndarray, side_to_move: bool = True
) -> chess.Board:
    """
    Create a chess.Board object from NPZ feature vector.

    Args:
        feature: Feature vector with 780 elements (piece planes + castling rights + en passant)
        side_to_move: True for white, False for black

    Returns:
        A chess.Board object representing the position
    """
    # Create empty board
    board = chess.Board.empty()

    # Extract the 12 piece planes (reshape to 12x8x8)
    piece_planes = feature[:768].reshape(12, 8, 8)

    # Extract castling rights
    castling_rights = feature[768:772].tolist()

    # Extract en passant
    en_passant = feature[772:780] if len(feature) >= 780 else None

    # Piece type mapping (0-5 for white pieces, 6-11 for black pieces)
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ] * 2

    # Color mapping (True for white pieces (0-5), False for black pieces (6-11))
    colors = [True] * 6 + [False] * 6

    # Fill the board with pieces
    for p in range(12):  # 12 piece types
        for r in range(8):  # 8 ranks
            for f in range(8):  # 8 files
                if piece_planes[p, r, f] == 1:
                    square = chess.square(f, r)
                    board.set_piece_at(square, chess.Piece(piece_types[p], colors[p]))

    # Set castling rights
    castling_fen = ""
    if castling_rights[0]:  # white kingside
        castling_fen += "K"
    if castling_rights[1]:  # white queenside
        castling_fen += "Q"
    if castling_rights[2]:  # black kingside
        castling_fen += "k"
    if castling_rights[3]:  # black queenside
        castling_fen += "q"

    if not castling_fen:
        castling_fen = "-"
    board.set_castling_fen(castling_fen)

    # Set the side to move
    board.turn = side_to_move

    # Set en passant square if available
    if en_passant is not None:
        en_passant_file = None
        for i in range(8):
            if en_passant[i] == 1:
                en_passant_file = i
                break

        if en_passant_file is not None:
            # In python-chess, the en passant square is where the capturing pawn would go
            rank = (
                5 if side_to_move else 2
            )  # Rank 6 for white, rank 3 for black (0-indexed)
            board.ep_square = chess.square(en_passant_file, rank)

    return board


def explore_npz_file(filename: str, max_positions: Optional[int] = None):
    """
    Explore and display chess positions from a NPZ training data file.

    Args:
        filename: Path to the NPZ file
        max_positions: Maximum number of positions to display (None for all)
    """
    print(f"Exploring NPZ file: {filename}")
    print("=" * 80)

    try:
        # Load the NPZ file
        with np.load(filename) as data:
            features = data["features"]
            best_q = data["best_q"]

            # Determine number of positions to display
            num_positions = len(features)
            if max_positions is not None:
                num_positions = min(num_positions, max_positions)

            # Display each position
            for position_count in range(num_positions):
                # Extract features for this position
                feature = features[position_count]
                q_values = best_q[position_count]

                # Determine side to move (alternate between positions)
                # This is a simplification since we don't have the original side to move information
                side_to_move = (
                    position_count % 2 == 0
                )  # Even positions are white, odd are black

                # Create chess board
                board = create_board_from_features(feature, side_to_move)

                # Get castling rights for display
                castling_rights = feature[768:772].tolist()

                # Extract en passant info if available
                en_passant_file = None
                if feature.shape[0] >= 780:  # If we have en passant features
                    en_passant = feature[772:780]
                    for i in range(8):
                        if en_passant[i] == 1:
                            en_passant_file = i
                            break

                # Display position information
                print(f"Position {position_count + 1}:")
                print(
                    f"Best Q: Win={q_values[0]:.4f}, Draw={q_values[1]:.4f}, Loss={q_values[2]:.4f}"
                )
                print(
                    f"Castling rights: {'K' if castling_rights[0] else ''}"
                    f"{'Q' if castling_rights[1] else ''}"
                    f"{'k' if castling_rights[2] else ''}"
                    f"{'q' if castling_rights[3] else ''}"
                )

                # Display side to move
                print(f"Side to move: {'Black' if not side_to_move else 'White'}")

                # Display en passant information if available
                if en_passant_file is not None:
                    file_letter = chr(97 + en_passant_file)  # 'a' through 'h'
                    rank = "6" if side_to_move else "3"  # Depends on whose turn it is
                    print(f"En passant possible on {file_letter}{rank}")
                    print(f"En passant feature: {list(feature[772:780])}")
                else:
                    print("No en passant possible")
                    if feature.shape[0] >= 780:
                        print(f"En passant feature: {list(feature[772:780])}")

                # Display the board
                print(board)
                print("-" * 80)

    except Exception as e:
        print(f"Error processing file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Explore NPZ training data files")
    parser.add_argument("input", help="Input NPZ file path")
    parser.add_argument(
        "--max", type=int, default=10, help="Maximum number of positions to display"
    )
    args = parser.parse_args()

    explore_npz_file(args.input, args.max)


if __name__ == "__main__":
    main()
