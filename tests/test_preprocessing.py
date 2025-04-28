import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.lc0_to_npz import process_tar_archive


def test_lc0_to_npz_integration():
    """
    Integration test for the lc0_to_npz preprocessor.
    Tests the full pipeline of converting a tar file containing LC0 .gz files to a single .npz file.
    """
    # Setup test directory
    temp_dir = tempfile.mkdtemp()
    output_dir = Path(temp_dir)
    test_tar_path = Path(__file__).parent / "test_data" / "lc0-data-sample.tar"

    try:
        # Ensure test data exists
        assert test_tar_path.exists(), "Test data file not found"

        # Process the tar file
        process_tar_archive(tar_path=str(test_tar_path), output_dir=output_dir)

        # Verify output file exists
        output_path = output_dir / "lc0-data-sample_chunk0.npz"
        assert output_path.exists(), "Output file was not created"

        # Load and verify the data
        data = np.load(output_path)

        # Check that the expected keys are present
        assert "features" in data, "Features not found in output data"
        assert "best_q" in data, "Best_q not found in output data"

        # Verify shapes and data types
        features = data["features"]
        best_q = data["best_q"]

        # Should be the same number of positions in both arrays
        assert (
            features.shape[0] == best_q.shape[0]
        ), "Features and best_q have different numbers of positions"

        # Features should have 780 elements (12 pieces * 8 * 8 + 4 castling rights + 8 en passant)
        assert features.shape[1] == 780, f"Features have wrong shape: {features.shape}"

        # Best_q should have 3 elements (win, draw, loss)
        assert best_q.shape[1] == 3, f"Best_q has wrong shape: {best_q.shape}"

        # Verify that all values in best_q are valid probabilities
        assert np.all(best_q >= 0) and np.all(
            best_q <= 1
        ), "Best_q contains invalid probability values"

        # Sum of probabilities should be close to 1
        assert np.allclose(
            np.sum(best_q, axis=1), 1.0, atol=1e-5
        ), "Win-Draw-Loss probabilities do not sum to 1"

        # Ensure we have at least some data
        assert features.shape[0] > 0, "No positions were processed"

        print(f"Integration test passed. Processed {features.shape[0]} positions.")

    finally:
        # Clean up
        # shutil.rmtree(temp_dir)
        pass


if __name__ == "__main__":
    test_lc0_to_npz_integration()
