#!/bin/bash

source ../.env

$FASTCHESS_PATH \
    -engine cmd=../engine/engine.sh name=MyEngine nodes=800 \
    -engine cmd=$(which stockfish) name=stockfish nodes=100 \
    -openings file=$OPENING_BOOKS_PATH/8moves_v3.pgn format=pgn \
    -log file=vs_stockfish.log \
    -pgnout file=vs_stockfish.pgn nodes=true
