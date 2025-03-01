#!/usr/bin/env python3
"""
LC0 Data Explorer

This script loads a Leela Chess Zero (LC0) training data file, parses it,
and visualizes the chess boards using the python-chess library.
It helps in verifying that our data parsing is correct.
"""

import argparse
import gzip
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import chess
import numpy as np

# Import our chunk parser from the conversion script
from src.preprocessing.lc0_to_npz import LeelaChunkParser


def create_board_from_planes(planes_array: np.ndarray, castling_rights: List[int], 
                             side_to_move: bool = True) -> chess.Board:
    """
    Create a chess.Board object from LC0 planes, castling rights, and side to move.
    
    Args:
        planes_array: 12×8×8 array of piece positions
        castling_rights: List of [us_oo, us_ooo, them_oo, them_ooo]
        side_to_move: True for white, False for black
        
    Returns:
        A chess.Board object representing the position
    """
    # Create empty board
    board = chess.Board.empty()
    
    # Piece type mapping (0-5 for our pieces, 6-11 for their pieces)
    piece_types = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP, 
        chess.ROOK, chess.QUEEN, chess.KING
    ] * 2
    
    # Color mapping (True for our pieces (0-5), False for their pieces (6-11))
    colors = [True] * 6 + [False] * 6
    
    # Fill the board with pieces
    for p in range(12):  # 12 piece types
        for r in range(8):  # 8 ranks
            for f in range(8):  # 8 files
                if planes_array[p, r, f] == 1:
                    # In LC0 format when it's black's turn, the board is rotated 180 degrees
                    # Here we're reconstructing it as seen from white's perspective
                    square = chess.square(f, r)
                    board.set_piece_at(square, chess.Piece(piece_types[p], colors[p]))
    
    # Set castling rights - we need to build the castling FEN section
    castling_fen = ""
    if castling_rights[0]:  # us_oo - our kingside
        castling_fen += "K"
    if castling_rights[1]:  # us_ooo - our queenside
        castling_fen += "Q"
    if castling_rights[2]:  # them_oo - their kingside
        castling_fen += "k"
    if castling_rights[3]:  # them_ooo - their queenside
        castling_fen += "q"
    
    # Set castling rights through the set_castling_fen method
    if not castling_fen:
        castling_fen = "-"
    board.set_castling_fen(castling_fen)

    # Set the side to move
    board.turn = side_to_move
    
    return board


def explore_lc0_file(filename: str, max_positions: Optional[int] = None):
    """
    Explore and display chess positions from a LC0 training data file.
    
    Args:
        filename: Path to the LC0 chunk file
        max_positions: Maximum number of positions to display (None for all)
    """
    # Initialize parser
    parser = LeelaChunkParser(filename, batch_size=1)
    
    position_count = 0
    
    print(f"Exploring LC0 file: {filename}")
    print("=" * 80)
    
    try:
        # Process all batches from this file
        for features, best_q in parser.parse_chunk():
            # We only get one position at a time with batch_size=1
            position_count += 1
            
            if max_positions is not None and position_count > max_positions:
                break
                
            # Reshape the features to extract board planes and other info
            board_planes = features[0, :768].reshape(12, 8, 8)
            castling_rights = features[0, 768:772].tolist()
            
            # Set side to move based on the stm_or_enpassant value (1 = black, 0 = white)
            side_to_move = False if parser.current_stm_or_enpassant == 1 else True
            
            # Create chess board
            board = create_board_from_planes(board_planes, castling_rights, side_to_move)
            
            # Display position information
            print(f"Position {position_count}:")
            print(f"Best Q: Win={best_q[0,0]:.4f}, Draw={best_q[0,1]:.4f}, Loss={best_q[0,2]:.4f}")
            print(f"Castling rights: {'K' if castling_rights[0] else ''}"
                  f"{'Q' if castling_rights[1] else ''}"
                  f"{'k' if castling_rights[2] else ''}"
                  f"{'q' if castling_rights[3] else ''}")
            
            print(f"Side to move: {'Black' if not side_to_move else 'White'}")
                
            # Display the board
            print(board)
            print("-" * 80)
            
    except Exception as e:
        print(f"Error processing file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Explore LC0 training data files')
    parser.add_argument('input', help='Input LC0 chunk file (e.g., "/path/to/data/training.12345.gz")')
    parser.add_argument('--max', type=int, default=10, 
                        help='Maximum number of positions to display')
    args = parser.parse_args()
    
    explore_lc0_file(args.input, args.max)


if __name__ == "__main__":
    main()