#!/bin/bash
# Experiment B (Partial Focal) - Stage 2
cd "$(dirname "$0")/.."
python train_stage2.py --config configs/stage2_focal.yaml --device cuda
