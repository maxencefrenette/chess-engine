import search
import chess
import chess.pgn
import sys
import traceback
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

CACHE_SIZE = 200000
MINTIME = 0.1
TIMEDIV = 25.0
NODES = 800
C = 3.0
MODEL_PATH = Path(os.getenv("MODELS_PATH")) / "swift-serenity-12.pth"


logfile = open("a0lite.log", "w")
LOG = True


def log(msg):
    # pretty print
    if isinstance(msg, list):
        msg = "\n".join(msg)

    if LOG:
        logfile.write(str(msg))
        logfile.write("\n")
        logfile.flush()


def send(str):
    log(">{}".format(str))
    sys.stdout.write(str)
    sys.stdout.write("\n")
    sys.stdout.flush()


def process_position(tokens):
    board = chess.Board()

    offset = 0

    if tokens[1] == "startpos":
        offset = 2
    elif tokens[1] == "fen":
        fen = " ".join(tokens[2:8])
        board = chess.Board(fen=fen)
        offset = 8

    if offset >= len(tokens):
        return board

    if tokens[offset] == "moves":
        for i in range(offset + 1, len(tokens)):
            board.push_uci(tokens[i])

    # deal with cutechess bug where a drawn positions is passed in
    if board.can_claim_draw():
        board.clear_stack()
    return board


def load_network():
    log(f"Loading network from {MODEL_PATH}")

    net = search.TrivialNet(MODEL_PATH)
    return net


def main():

    send("A0 Lite")
    board = chess.Board()
    nn = None

    while True:
        line = sys.stdin.readline()
        line = line.rstrip()
        log("<{}".format(line))
        tokens = line.split()
        if len(tokens) == 0:
            continue

        if tokens[0] == "uci":
            send("id name A0 Lite")
            send("id author Dietrich Kappe")
            send("uciok")
        elif tokens[0] == "quit":
            exit(0)
        elif tokens[0] == "isready":
            if nn == None:
                nn = load_network()
            send("readyok")
        elif tokens[0] == "ucinewgame":
            board = chess.Board()

        elif tokens[0] == "position":
            board = process_position(tokens)

        elif tokens[0] == "go":
            my_nodes = NODES
            my_time = None
            if (len(tokens) == 3) and (tokens[1] == "nodes"):
                my_nodes = int(tokens[2])
            if (len(tokens) == 3) and (tokens[1] == "movetime"):
                my_time = int(tokens[2]) / 1000.0
                if my_time < MINTIME:
                    my_time = MINTIME
            if (len(tokens) == 9) and (tokens[1] == "wtime"):
                wtime = int(tokens[2])
                btime = int(tokens[4])
                winc = int(tokens[6])
                binc = int(tokens[8])
                if wtime > 5 * winc:
                    wtime += 5 * winc
                else:
                    wtime += winc
                if btime > 5 * binc:
                    btime += 5 * binc
                else:
                    btime += binc
                if board.turn:
                    my_time = wtime / (TIMEDIV * 1000.0)
                else:
                    my_time = btime / (TIMEDIV * 1000.0)
                if my_time < MINTIME:
                    my_time = MINTIME
            if nn == None:
                nn = load_network()

            if my_time != None:
                best, score = search.UCT_search(
                    board, 1000000, net=nn, C=C, max_time=my_time, send=send
                )
            else:
                best, score = search.UCT_search(board, my_nodes, net=nn, C=C, send=send)
            send("bestmove {}".format(best))


try:
    main()
except:
    exc_type, exc_value, exc_tb = sys.exc_info()
    log(traceback.format_exception(exc_type, exc_value, exc_tb))

logfile.close()
