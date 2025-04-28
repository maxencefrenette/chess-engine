#!/bin/bash

BASEDIR=$(dirname "$0")

source $BASEDIR/../.env

$FASTCHESS_PATH \
    -engine cmd=$BASEDIR/engine.sh name=MyEngine nodes=800 \
    -engine cmd=$(which stockfish) name=stockfish nodes=100 \
    -openings file=$OPENING_BOOKS_PATH/8moves_v3.pgn format=pgn \
    -log file=$BASEDIR/vs_stockfish.log \
    -pgnout file=$BASEDIR/vs_stockfish.pgn nodes=true
