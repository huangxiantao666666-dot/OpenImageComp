#!/bin/bash
# Experiment B (Partial Focal) - Stage 1
# Gaussian heatmap + masked Focal Loss (pos+neg regions only)
cd "$(dirname "$0")/.."
python train_stage1.py --config configs/stage1_focal.yaml --device cuda
