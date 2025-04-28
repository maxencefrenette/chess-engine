#! /bin/bash

BASEDIR=$(dirname "$0")

uv run $BASEDIR/../src/engine/engine.py "$@"
