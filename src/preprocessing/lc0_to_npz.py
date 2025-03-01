#!/usr/bin/env python3
"""
Convert LC0 training data to NPZ format for faster loading and simpler processing.

This script reads LC0 training data from a .tar archive containing multiple .gz files,
extracts the relevant features and labels, and saves them in the more efficient NPZ format
as described in docs/training_data_format.md.
"""

import argparse
import gzip
import io
import os
import struct
import tarfile
from pathlib import Path
from typing import BinaryIO, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

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

    def __init__(self, file_or_path: Union[str, BinaryIO]):
        """
        Initialize the chunk parser.

        Args:
            file_or_path: Path to the chunk file or a file-like object.
        """
        self.file_or_path = file_or_path

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
        planes_array = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(
            np.float32
        )
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

    def parse_game(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse a game file and return all features and best_q values at once.

        Returns:
            A tuple of (features, best_q) arrays containing all positions.
        """
        try:
            # Handle both file paths and file-like objects
            if isinstance(self.file_or_path, str):
                file_obj = gzip.open(self.file_or_path, "rb")
            else:
                # Assume it's already a file-like object
                file_obj = gzip.GzipFile(fileobj=self.file_or_path, mode="rb")

            with file_obj as chunk_file:
                # Get the version and record size from the first 4 bytes
                version = chunk_file.read(4)
                chunk_file.seek(0)

                version, record_size = self._get_version_and_record_size(version)

                # Initialize arrays for all features and best_q values
                features_all = []
                best_q_all = []

                # Read the entire file
                chunk_data = chunk_file.read()

                # Process each record in the chunk
                for i in range(0, len(chunk_data), record_size):
                    if i + record_size > len(chunk_data):
                        break

                    # Extract one record
                    record = chunk_data[i : i + record_size]

                    # Adapt older versions to V6 format
                    if version != V6_VERSION:
                        record = self._adapt_v3_v4_v5_to_v6(record, version)

                    # Parse the record
                    features, best_q = self._parse_v6_record(record)

                    # Add to arrays
                    features_all.append(features)
                    best_q_all.append(best_q)

                # Return all records at once
                return np.array(features_all), np.array(best_q_all)

        except Exception as e:
            source = (
                self.file_or_path
                if isinstance(self.file_or_path, str)
                else "in-memory file"
            )
            print(f"Failed to parse {source}: {e}")
            raise


def process_tar_archive(tar_path: str, output_dir: Path) -> None:
    """
    Process a tar archive containing LC0 chunk files (.gz files) and output a single NPZ file.

    Args:
        tar_path: Path to the tar archive containing .gz files
        output_dir: Directory to save the combined NPZ file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if the file exists and is a tar file
    if not os.path.exists(tar_path):
        print(f"File not found: {tar_path}")
        return

    if not tarfile.is_tarfile(tar_path):
        print(f"Not a valid tar file: {tar_path}")
        return

    # Create output filename from the tar filename
    tar_base_name = os.path.basename(tar_path)
    if tar_base_name.endswith(".tar"):
        tar_base_name = tar_base_name[:-4]
    output_path = output_dir / f"{tar_base_name}.npz"

    # Accumulators for all features and best_q values
    all_features = []
    all_best_q = []

    # Open the tar file and list all .gz files
    with tarfile.open(tar_path, "r") as tar:
        # Get all .gz files in the archive, excluding macOS hidden files
        gz_files = [
            m
            for m in tar.getmembers()
            if m.name.endswith(".gz") and not os.path.basename(m.name).startswith("._")
        ]

        if not gz_files:
            print(f"No .gz files found in the tar archive: {tar_path}")
            return

        print(f"Found {len(gz_files)} .gz files in the tar archive")

        # Process each file sequentially
        total_positions = 0
        for member in tqdm(gz_files, desc="Processing files"):
            file_obj = tar.extractfile(member)
            if file_obj is None:
                print(f"Could not extract {member.name} from tar")
                continue

            try:
                # Process the file directly without saving to disk
                parser = LeelaChunkParser(file_obj)

                # Process all positions from this file - we only get one yield with all data
                features, best_q = parser.parse_game()
                total_positions += features.shape[0]

                # Add to overall accumulators
                all_features.append(features)
                all_best_q.append(best_q)
            except Exception as e:
                print(f"Error processing {member.name}: {e}")

        # Combine all data from all files
        if all_features and all_best_q:
            combined_features = np.vstack(all_features)
            combined_best_q = np.vstack(all_best_q)

            # Save as a single NPZ file
            np.savez_compressed(
                output_path, features=combined_features, best_q=combined_best_q
            )

            print(f"Processed {total_positions} positions")
            print(
                f"Created combined NPZ file: {output_path} with {combined_features.shape[0]} positions"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Convert LC0 training data from a tar archive to NPZ format"
    )
    parser.add_argument(
        "input",
        help='Path to the tar archive containing .gz files (e.g., "/path/to/data.tar")',
    )
    parser.add_argument("output", help="Output directory for NPZ files")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of positions to process at once",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    process_tar_archive(
        tar_path=args.input, output_dir=output_dir, batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
