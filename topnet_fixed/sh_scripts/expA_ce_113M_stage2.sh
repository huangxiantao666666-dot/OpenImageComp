#!/bin/bash
# Experiment A (113M): Sparse CrossEntropy - Stage 2
cd "$(dirname "$0")/.."
python train_stage2.py --config configs/stage2_ce_113M.yaml --device cuda
