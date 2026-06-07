#!/bin/bash
# Experiment A (113M): Sparse CrossEntropy - Stage 1
cd "$(dirname "$0")/.."
python train_stage1.py --config configs/stage1_ce_113M.yaml --device cuda
