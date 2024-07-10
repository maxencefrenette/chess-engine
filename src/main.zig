const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) @panic("TEST FAIL");
    }

    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();

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
