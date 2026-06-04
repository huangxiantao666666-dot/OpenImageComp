#!/bin/bash
# Experiment A (Sparse CrossEntropy) - Stage 1
# Freeze encoders, train Transformer + Decoder with CE loss
cd "$(dirname "$0")/.."
python train_stage1.py --config configs/stage1_ce.yaml --device cuda
