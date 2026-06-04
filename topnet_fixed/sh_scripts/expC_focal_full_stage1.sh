#!/bin/bash
# Experiment C (Full Focal) - Stage 1
# Gaussian heatmap + full-supervision Focal Loss (all pixels)
cd "$(dirname "$0")/.."
python train_stage1.py --config configs/stage1_focal_full.yaml --device cuda
