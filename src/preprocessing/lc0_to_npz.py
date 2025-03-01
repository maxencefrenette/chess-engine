#!/usr/bin/env python3
"""
Convert LC0 training data to NPZ format for faster loading and simpler processing.

This script reads LC0 chunk files, extracts the relevant features and labels, and
saves them in the more efficient NPZ format as described in docs/training_data_format.md.
"""

import os
import argparse
import glob
import struct
import gzip
import numpy as np
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from typing import Tuple, Optional, Generator

# Define constants for LC0 chunk format
V6_VERSION = struct.pack("i", 6)
V5_VERSION = struct.pack("i", 5)
V4_VERSION = struct.pack("i", 4)
V3_VERSION = struct.pack("i", 3)
CLASSICAL_INPUT = struct.pack("i", 1)

# Define struct formats for different LC0 versions
V6_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHH4H"
V5_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffff"
V4_STRUCT_STRING = "4s7432s832sBBBBBBBbffff"
V3_STRUCT_STRING = "4s7432s832sBBBBBBBb"


class LeelaChunkParser:
    """Parser for Leela Chess Zero training data chunks."""

    def __init__(self, filename: str, batch_size: int = 256):
        """
        Initialize the chunk parser.
        
        Args:
            filename: Path to the chunk file.
            batch_size: Number of positions to process at once.
        """
        self.filename = filename
        self.batch_size = batch_size
        
        # Initialize struct parsers
        self.v6_struct = struct.Struct(V6_STRUCT_STRING)
        
        # Flat planes for common board representations
        self.flat_planes = []
        for i in range(2):
            self.flat_planes.append((np.zeros(64, dtype=np.float32) + i).tobytes())
        
        # Store the previous position's pawn locations to detect en passant
        self.prev_white_pawns = np.zeros((8, 8), dtype=np.float32)
        self.prev_black_pawns = np.zeros((8, 8), dtype=np.float32)

    def _reverse_expand_bits(self, plane: int) -> bytes:
        """Reverse and expand a byte into bits."""
        return (
            np.unpackbits(np.array([plane], dtype=np.uint8))[::-1]
            .astype(np.float32)
            .tobytes()
        )

    def _get_version_and_record_size(self, chunk_header: bytes) -> Tuple[bytes, int]:
        """
        Get the version and record size from a chunk header.
        
        Args:
            chunk_header: First 4 bytes of the chunk.
            
        Returns:
            Tuple of (version, record_size).
        """
        version = chunk_header[0:4]
        if version == V6_VERSION:
            record_size = self.v6_struct.size
        elif version == V5_VERSION:
            record_size = self.v5_struct.size
        elif version == V4_VERSION:
            record_size = self.v4_struct.size
        elif version == V3_VERSION:
            record_size = self.v3_struct.size
        else:
            raise ValueError(f"Unknown version in chunk file: {version}")
        
        return version, record_size

    def _parse_v6_record(self, record: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse a V6 format record into features and best_q.
        
        Args:
            record: Raw record bytes.
            
        Returns:
            Tuple of (features, best_q).
        """
        # Unpack the record
        (
            ver,
            input_format,
            probs,
            planes,
            us_ooo,
            us_oo,
            them_ooo,
            them_oo,
            stm,
            rule50_count,
            invariance_info,
            dep_result,
            root_q,
            best_q,
            root_d,
            best_d,
            root_m,
            best_m,
            plies_left,
            result_q,
            result_d,
            played_q,
            played_d,
            played_m,
            orig_q,
            orig_d,
            orig_m,
            visits,
            played_idx,
            best_idx,
            reserved1,
            reserved2,
            reserved3,
            reserved4,
        ) = self.v6_struct.unpack(record)
        
        # Ensure the input format is what we expect
        if input_format != 1:
            raise ValueError(f"Unsupported input format: {input_format}")
        
        # Parse planes (the actual board state)
        planes_array = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32)
        planes_array = planes_array.reshape(104, 8, 8)  # 104 planes of 8x8
        
        # We only need the first 12 planes (piece positions) and castling rights
        board_planes = planes_array[:12]  # 12 planes for pieces
        
        # Get castling rights directly from the record
        castling_rights = np.array([us_oo, us_ooo, them_oo, them_ooo], dtype=np.float32)
        
        # Extract current pawn positions
        # In LC0 format, pawns are at index 0 (white) and 6 (black)
        white_pawns = board_planes[0]  # White pawns
        black_pawns = board_planes[6]  # Black pawns
        
        # Detect en passant by comparing with previous position
        # The side to move now is the opposite of the previous position
        is_white_turn = stm == 0
        enpassant = np.zeros(8, dtype=np.float32)  # One-hot vector for en passant file
        
        if is_white_turn:
            # Black just moved, check if a black pawn moved two squares
            # Black pawn moved from rank 6 to rank 4
            for file in range(8):
                # Check if there's a black pawn at rank 4 (index 4)
                if black_pawns[4, file] == 1:
                    # Check if there was a black pawn at rank 6 (index 6) in the previous position
                    if self.prev_black_pawns[6, file] == 1:
                        # And no black pawn at rank 5 (index 5) in the previous position
                        if self.prev_black_pawns[5, file] == 0:
                            # And no black pawn at rank 4 (index 4) in the previous position
                            if self.prev_black_pawns[4, file] == 0:
                                # En passant is possible on this file
                                enpassant[file] = 1.0
        else:
            # White just moved, check if a white pawn moved two squares
            # White pawn moved from rank 1 to rank 3
            for file in range(8):
                # Check if there's a white pawn at rank 3 (index 3)
                if white_pawns[3, file] == 1:
                    # Check if there was a white pawn at rank 1 (index 1) in the previous position
                    if self.prev_white_pawns[1, file] == 1:
                        # And no white pawn at rank 2 (index 2) in the previous position
                        if self.prev_white_pawns[2, file] == 0:
                            # And no white pawn at rank 3 (index 3) in the previous position
                            if self.prev_white_pawns[3, file] == 0:
                                # En passant is possible on this file
                                enpassant[file] = 1.0
        
        # Store current pawn positions for next comparison
        self.prev_white_pawns = white_pawns.copy()
        self.prev_black_pawns = black_pawns.copy()
        
        # Parse best_q (Win-Draw-Loss probabilities)
        best_q_w = 0.5 * (1.0 - best_d + best_q)
        best_q_l = 0.5 * (1.0 - best_d - best_q)
        best_q_array = np.array([best_q_w, best_d, best_q_l], dtype=np.float32)
        
        # Create feature vector (board planes flattened + castling rights + en passant)
        features = np.concatenate([board_planes.flatten(), castling_rights, enpassant])
        
        return features, best_q_array

    def _adapt_v3_v4_v5_to_v6(self, record: bytes, version: bytes) -> bytes:
        """
        Adapt older record formats (V3, V4, V5) to V6 format.
        
        Args:
            record: Raw record bytes.
            version: Version of the record.
            
        Returns:
            Adapted record in V6 format.
        """
        # For earlier versions, append fake bytes to record to maintain size
        if version == V3_VERSION:
            # Add 16 bytes of fake root_q, best_q, root_d, best_d to match V4 format
            record += 16 * b"\x00"
        if version == V3_VERSION or version == V4_VERSION:
            # Add 12 bytes of fake root_m, best_m, plies_left to match V5 format
            record += 12 * b"\x00"
            # Insert 4 bytes of classical input format tag to match v5 format
            record = record[:4] + CLASSICAL_INPUT + record[4:]
        if version == V3_VERSION or version == V4_VERSION or version == V5_VERSION:
            # Add 48 bytes of fake result_q, result_d etc
            record += 48 * b"\x00"
        
        return record

    def parse_chunk(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Parse the chunk file and yield batches of (features, best_q).
        
        Yields:
            Batches of (features, best_q) arrays.
        """
        try:
            with gzip.open(self.filename, "rb") as chunk_file:
                # Get the version and record size from the first 4 bytes
                version = chunk_file.read(4)
                chunk_file.seek(0)
                
                version, record_size = self._get_version_and_record_size(version)
                
                # Initialize batch arrays
                features_batch = []
                best_q_batch = []
                
                # Read records from the chunk file
                while True:
                    # Read multiple records at once
                    chunk_data = chunk_file.read(self.batch_size * record_size)
                    if not chunk_data:
                        break
                    
                    # Process each record in the chunk
                    for i in range(0, len(chunk_data), record_size):
                        if i + record_size > len(chunk_data):
                            break
                            
                        # Extract one record
                        record = chunk_data[i:i + record_size]
                        
                        # Adapt older versions to V6 format
                        if version != V6_VERSION:
                            record = self._adapt_v3_v4_v5_to_v6(record, version)
                        
                        # Parse the record
                        features, best_q = self._parse_v6_record(record)
                        
                        # Add to batch
                        features_batch.append(features)
                        best_q_batch.append(best_q)
                        
                        # Yield batch when it reaches the desired size
                        if len(features_batch) >= self.batch_size:
                            yield np.array(features_batch), np.array(best_q_batch)
                            features_batch = []
                            best_q_batch = []
                
                # Yield any remaining records
                if features_batch:
                    yield np.array(features_batch), np.array(best_q_batch)
                    
        except Exception as e:
            print(f"Failed to parse {self.filename}: {e}")
            raise


def process_chunk_file(filename: str, output_dir: Path, batch_size: int = 1000) -> Tuple[int, int]:
    """
    Process a single LC0 chunk file and save as NPZ.
    
    Args:
        filename: Path to the LC0 chunk file
        output_dir: Directory to save the NPZ files
        batch_size: Number of positions to process at once
        
    Returns:
        Tuple of (positions_processed, bytes_saved)
    """
    # Track stats
    positions_processed = 0
    original_size = os.path.getsize(filename)
    npz_size = 0
    
    # Create output filename
    base_name = os.path.basename(filename)
    if base_name.endswith('.gz'):
        base_name = base_name[:-3]
    output_path = output_dir / f"{base_name}.npz"
    
    # Process batches
    all_features = []
    all_best_q = []
    
    try:
        # Initialize parser
        parser = LeelaChunkParser(filename, batch_size=batch_size)
        
        # Process all batches from this file
        for features, best_q in parser.parse_chunk():
            batch_size = features.shape[0]
            positions_processed += batch_size
            
            # Accumulate features and best_q
            all_features.append(features)
            all_best_q.append(best_q)
        
        # Combine all batches
        if all_features and all_best_q:
            all_features = np.vstack(all_features)
            all_best_q = np.vstack(all_best_q)
            
            # Save as NPZ
            np.savez_compressed(
                output_path,
                features=all_features,
                best_q=all_best_q
            )
            
            # Calculate compression stats
            npz_size = os.path.getsize(output_path)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return 0, 0
    
    return positions_processed, original_size - npz_size


def process_chunk_files(
    input_pattern: str, 
    output_dir: Path, 
    workers: Optional[int] = None,
    batch_size: int = 1000
) -> None:
    """
    Process multiple LC0 chunk files using multiple workers.
    
    Args:
        input_pattern: Glob pattern to match LC0 chunk files
        output_dir: Directory to save the NPZ files
        workers: Number of worker processes to use (defaults to CPU count - 1)
        batch_size: Number of positions to process at once
    """
    # Get all matching files
    chunk_files = glob.glob(input_pattern)
    if not chunk_files:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    print(f"Found {len(chunk_files)} chunk files to process")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)
    
    print(f"Using {workers} worker processes")
    
    # Process files in parallel
    with mp.Pool(workers) as pool:
        results = list(tqdm(
            pool.starmap(
                process_chunk_file, 
                [(f, output_dir, batch_size) for f in chunk_files]
            ),
            total=len(chunk_files),
            desc="Converting chunks"
        ))
    
    # Calculate and print statistics
    total_positions = sum(r[0] for r in results)
    total_bytes_saved = sum(r[1] for r in results)
    print(f"Processed {total_positions} positions")
    print(f"Saved {total_bytes_saved/1024/1024:.2f} MB of disk space")


def main():
    parser = argparse.ArgumentParser(description='Convert LC0 training data to NPZ format')
    parser.add_argument('input', help='Input pattern for LC0 chunk files (e.g., "/path/to/data/*.gz")')
    parser.add_argument('output', help='Output directory for NPZ files')
    parser.add_argument('--workers', type=int, help='Number of worker processes')
    parser.add_argument('--batch-size', type=int, default=1000, 
                        help='Number of positions to process at once')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    process_chunk_files(
        input_pattern=args.input,
        output_dir=output_dir,
        workers=args.workers,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
