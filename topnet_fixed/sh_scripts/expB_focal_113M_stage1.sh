#!/bin/bash
# Experiment B (113M): Partial Focal Loss - Stage 1
cd "$(dirname "$0")/.."
python train_stage1.py --config configs/stage1_focal_113M.yaml --device cuda
