#!/bin/bash
# Experiment B (113M): Partial Focal Loss - Stage 2
cd "$(dirname "$0")/.."
python train_stage2.py --config configs/stage2_focal_113M.yaml --device cuda
