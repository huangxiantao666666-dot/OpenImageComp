#!/bin/bash
# Experiment A (Sparse CrossEntropy) - Stage 2
# Unfreeze all, full-model fine-tuning
cd "$(dirname "$0")/.."
python train_stage2.py --config configs/stage2_ce.yaml --device cuda
