import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def load_ui():
    # UI to select file and number of positions
    from pathlib import Path

    import marimo as mo

    # Default sample tar in tests/test_data
    default_tar = (
        Path(__file__).parents[2] / "tests" / "test_data" / "lc0-data-sample.tar"
    )
    file_path = mo.ui.text(
        label="LC0 tar archive",
        value=str(default_tar),
        placeholder="Enter path to .tar file containing LC0 chunks",
    )
    max_positions = mo.ui.number(label="Max positions", value=10)
    load_button = mo.ui.run_button(label="Load Data")
    mo.vstack([file_path, max_positions, load_button])
    # Return marimo handle for downstream cells
    return mo, file_path, max_positions, load_button


@app.cell
def parse_data(mo, file_path, max_positions, load_button):
    import sys
    import tarfile
    from pathlib import Path as _P

    import numpy as np

    from src.preprocessing.lc0_to_npz import LeelaChunkParser

    # Determine data source: UI or CLI argument
    path = file_path.value or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not path:
        raise ValueError("Provide a .tar LC0 data file via UI or CLI")

    features_list = []
    q_list = []

    # Only .tar archives supported: extract contained .gz chunks
    if not tarfile.is_tarfile(path):
        raise ValueError("Only .tar LC0 data files are supported")
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = _P(m.name).name
            # Skip non-chunk or hidden files
            if not name.endswith(".gz") or name.startswith("._"):
                continue
            fobj = tar.extractfile(m)
            parser = LeelaChunkParser(fobj)
            feats, qs = parser.parse_game()
            features_list.extend(feats.tolist())
            q_list.extend(qs.tolist())

    # Convert lists to arrays
    features_all = np.array(features_list, dtype=np.float32)
    best_q_all = np.array(q_list, dtype=np.float32)
    num_positions = min(int(max_positions.value), len(features_all))
    return mo, features_all, best_q_all, num_positions


@app.cell
def select_position(mo, features_all, best_q_all, num_positions):
    # Position selector UI
    position = mo.ui.number(label="Position", start=1, stop=num_positions, value=1)
    mo.vstack([position])
    return mo, features_all, best_q_all, num_positions, position


@app.cell
def display_position(mo, features_all, best_q_all, num_positions, position):
    import chess

    idx = int(position.value) - 1
    feat = features_all[idx]
    q = best_q_all[idx]

    # Extract features
    board_planes = feat[:768].reshape(12, 8, 8)
    castling_rights = feat[768:772].tolist()
    en_passant = feat[772:780].tolist() if feat.shape[0] >= 780 else [0] * 8

    # Build chess board
    board = chess.Board.empty()
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ] * 2
    colors = [True] * 6 + [False] * 6
    for p in range(12):
        for r in range(8):
            for f in range(8):
                if board_planes[p, r, f] == 1:
                    sq = chess.square(f, r)
                    board.set_piece_at(sq, chess.Piece(piece_types[p], colors[p]))

    # Format castling and en passant info
    castling_chars = [
        c for flag, c in zip(castling_rights, ["K", "Q", "k", "q"]) if flag
    ]
    cast_str = "".join(castling_chars) if castling_chars else "-"
    en_files = [chr(97 + i) for i, v in enumerate(en_passant) if v == 1]
    en_str = ", ".join(en_files) if en_files else "None"

    # Display details
    mo.md(f"**Position {idx+1} of {num_positions}**")
    mo.md(f"Best Q: Win={q[0]:.4f}, Draw={q[1]:.4f}, Loss={q[2]:.4f}")
    mo.md(f"Castling rights: {cast_str}")
    mo.md(f"En passant files: {en_str} (features: {en_passant})")
    mo.md(f"```{board}```")


if __name__ == "__main__":
    app.run()
