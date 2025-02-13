#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
exec poetry run python src/engine.py
