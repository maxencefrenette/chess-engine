#!/bin/bash

# Run the script with autoretry
for i in {1..10}; do
    uv run src/training/experiments/tune.py --num-trials 1000
    echo $? >> autoretry.log
done
