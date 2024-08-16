const std = @import("std");

const Position = struct {
    board: [120]u8,
    score: i32,
    castling_rights: struct {
        white_king_side: bool,
        white_queen_side: bool,
        black_king_side: bool,
        black_queen_side: bool,
    },
    en_passant_square: u8,
    king_passant_square: u8,
    
    fn from_fen(fen: []const u8) Position {
        var pos = Position{
            .board = undefined,
            .score = 0,
            .castling_rights = undefined,
            .en_passant_square = undefined,
            .king_passant_square = undefined,
        };
        std.mem.copyForwards(u8, &pos.board, "         \n" ** 12);

        var tokens = std.mem.tokenizeScalar(u8, fen, ' ');

        const board = tokens.next() orelse @panic("FEN string is missing board");
        var i: u8 = 0;
        var j: u8 = 0;
        for (board) |c| {
            if (c == '/') {
                i += 1;
                j = 0;
                continue;
            }

            if (c >= '1' and c <= '8') {
                j += @intCast(c - '0');
                continue;
            }

            pos.board[10 * (i + 2) + j + 1] = c;
            j += 1;
        }

        // TODO: Parse the rest of the FEN string

        return pos;
    }
};

test "FEN parsing" {
    const pos = Position.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    const expected_board =
        \\         
        \\         
        \\ rnbqkbnr
        \\ pppppppp
        \\         
        \\         
        \\         
        \\         
        \\ PPPPPPPP
        \\ RNBQKBNR
        \\         
        \\         
        \\
        ;

    try std.testing.expectEqualStrings(expected_board, &pos.board);
}
