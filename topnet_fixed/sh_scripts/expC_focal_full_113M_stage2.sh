#!/bin/bash
# Experiment C (113M): Full-supervision Focal Loss - Stage 2
cd "$(dirname "$0")/.."
python train_stage2.py --config configs/stage2_focal_full_113M.yaml --device cuda
