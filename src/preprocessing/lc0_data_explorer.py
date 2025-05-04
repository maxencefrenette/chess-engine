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
def parse_data(file_path):
    """Parse all games from the .tar archive into lists of features and Qs"""
    import sys
    import tarfile
    from pathlib import Path as _P

    import numpy as np

    from src.preprocessing.lc0_to_npz import LeelaChunkParser

    # Determine input path (UI or CLI)
    path = file_path.value or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not path or not tarfile.is_tarfile(path):
        raise ValueError("Provide a valid .tar LC0 data file via UI or CLI")

    game_names = []
    features_by_game = []
    best_q_by_game = []

    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = _P(member.name).name
            # Only consider gz chunk files
            if not name.endswith(".gz") or name.startswith("._"):
                continue
            fobj = tar.extractfile(member)
            parser = LeelaChunkParser(fobj)
            feats, qs = parser.parse_game()
            game_names.append(name)
            features_by_game.append(feats)
            best_q_by_game.append(qs)

    return best_q_by_game, feats, features_by_game, game_names, qs


@app.cell
def select_position(game_names, mo):
    """Select which game to explore"""
    game = mo.ui.dropdown(
        label="Game",
        options=game_names,
        value=game_names[0] if game_names else None,
    )
    mo.vstack([game])
    return (game,)


@app.cell
def _(best_q_by_game, features_by_game, game, game_names, mo):
    """Select which position within the chosen game"""
    idx_game = game_names.index(game.value)
    game_feats = features_by_game[idx_game]
    game_qs = best_q_by_game[idx_game]
    pos = mo.ui.number(label="Position", start=1, stop=len(game_feats), value=1)
    mo.vstack([pos])
    return (pos,)


@app.cell
def display_position(feats, game, mo, pos, qs):
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
    title = mo.md(f"**Game: {game.value}, Position {idx+1} of {len(feats)}**")
    q_line = mo.md(f"Best Q: Win={q[0]:.4f}, Draw={q[1]:.4f}, Loss={q[2]:.4f}")
    cast_line = mo.md(f"Castling rights: {cast_str}")
    ep_line = mo.md(f"En passant files: {en_str} (features: {en_passant})")
    svg = board._repr_svg_()
    board_svg = mo.as_html(svg)

    mo.vstack([title, q_line, cast_line, ep_line, board_svg])
    return


if __name__ == "__main__":
    app.run()
