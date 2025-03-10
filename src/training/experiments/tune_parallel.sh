#!/bin/bash

rm -rf tuning_logs
mkdir -p tuning_logs
seq 1 100 | parallel -j 4 --progress 'uv run src/training/experiments/tune.py --num-trials 10 > tuning_logs/{}.txt 2>&1'
