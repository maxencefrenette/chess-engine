#!/bin/bash

uv run src/training/experiments/tune_ladder/run.py --flops-target 1e11
uv run src/training/experiments/tune_ladder/run.py --flops-target 3e11 --previous-flops-target 1e11
uv run src/training/experiments/tune_ladder/run.py --flops-target 1e12 --previous-flops-target 3e11
uv run src/training/experiments/tune_ladder/run.py --flops-target 3e12 --previous-flops-target 1e12
uv run src/training/experiments/tune_ladder/run.py --flops-target 1e13 --previous-flops-target 3e12
uv run src/training/experiments/tune_ladder/run.py --flops-target 3e13 --previous-flops-target 1e13
uv run src/training/experiments/tune_ladder/run.py --flops-target 1e14 --previous-flops-target 3e13
uv run src/training/experiments/tune_ladder/run.py --flops-target 3e14 --previous-flops-target 1e14
uv run src/training/experiments/tune_ladder/run.py --flops-target 1e15 --previous-flops-target 3e14
uv run src/training/experiments/tune_ladder/run.py --flops-target 3e15 --previous-flops-target 1e15
