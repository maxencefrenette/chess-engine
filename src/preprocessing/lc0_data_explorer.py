import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def load_ui():
    """UI to select the LC0 .tar archive file"""
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
    file_path
    return file_path, mo


@app.cell
def list_chunks(file_path):
    """List LC0 chunk (.gz) files in the .tar archive"""
    import sys
    import tarfile
    from pathlib import Path as _P

    # Determine input path (UI or CLI)
    path = file_path.value or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not path or not tarfile.is_tarfile(path):
        raise ValueError("Provide a valid .tar LC0 data file via UI or CLI")

    def get_chunk_names(path):
        """Get the names of all gz chunk files in the tar archive"""
        chunk_names = []
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = _P(member.name).name
                # Only consider gz chunk files
                if not name.endswith(".gz") or name.startswith("._"):
                    continue
                chunk_names.append(name)
        return chunk_names

    chunk_names = get_chunk_names(path)
    return chunk_names, path, tarfile


@app.cell
def select_chunk(chunk_names, mo):
    """Select which chunk file to parse"""
    chunk = mo.ui.dropdown(
        label="Chunk file",
        options=chunk_names,
        value=chunk_names[0] if chunk_names else None,
    )
    mo.vstack([chunk])
    return (chunk,)


@app.cell
def parse_chunk(chunk, path, tarfile):
    """Parse the selected chunk file into features and Qs"""
    from pathlib import Path as _P

    import numpy as np

    from src.preprocessing.lc0_to_npz import LeelaChunkParser

    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = _P(member.name).name
            if name != chunk.value:
                continue
            fobj = tar.extractfile(member)
            parser = LeelaChunkParser(fobj)
            feats, qs = parser.parse_game()
            break
        else:
            raise ValueError(f"Selected chunk {chunk.value} not found in archive")

    return feats, qs


@app.cell
def select_position(feats, mo):
    """Select which position within the chosen chunk"""
    pos = mo.ui.number(label="Position", start=1, stop=len(feats), value=1)
    mo.vstack([pos])
    return (pos,)


@app.cell
def display_position(chunk, feats, mo, pos, qs):
    import chess

    idx = int(pos.value) - 1
    feat = feats[idx]
    q = qs[idx]

    # Parse features
    board_planes = feat[:768].reshape(12, 8, 8)
    castling = feat[768:772].tolist()
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
            for c in range(8):
                if board_planes[p, r, c] == 1:
                    sq = chess.square(c, r)
                    board.set_piece_at(sq, chess.Piece(piece_types[p], colors[p]))

    # Format info strings
    castling_chars = [c for flag, c in zip(castling, ["K", "Q", "k", "q"]) if flag]
    cast_str = "".join(castling_chars) if castling_chars else "-"
    en_files = [chr(97 + i) for i, v in enumerate(en_passant) if v == 1]
    en_str = ", ".join(en_files) if en_files else "None"

    # Display metadata and board
    title = mo.md(f"**Chunk: {chunk.value}, Position {idx+1} of {len(feats)}**")
    q_line = mo.md(f"Best Q: Win={q[0]:.4f}, Draw={q[1]:.4f}, Loss={q[2]:.4f}")
    cast_line = mo.md(f"Castling rights: {cast_str}")
    ep_line = mo.md(f"En passant files: {en_str} (features: {en_passant})")
    svg = board._repr_svg_()
    board_svg = mo.as_html(svg)

    mo.vstack([title, q_line, cast_line, ep_line, board_svg])
    return


if __name__ == "__main__":
    app.run()
