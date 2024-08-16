const std = @import("std");
const stdin = std.io.getStdIn().reader();
const stdout = std.io.getStdOut().writer();

pub fn uci_main_loop(allocator: std.mem.Allocator) !void {
    while (true) {
        const input = try stdin.readUntilDelimiterAlloc(allocator, '\n', 1024);
        defer allocator.free(input);

        if (std.mem.eql(u8, input, "uci")) {
            try stdout.print("id name zig-chess-engine\n", .{});
            try stdout.print("uciok\n", .{});
        } else if (std.mem.eql(u8, input, "isready")) {
            try stdout.print("readyok\n", .{});
        } else if (std.mem.eql(u8, input, "quit")) {
            break;
        } else {
            std.debug.print("Unknown command: {s}\n", .{input});
        }
    }
}
