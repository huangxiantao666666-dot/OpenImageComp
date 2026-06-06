#!/bin/bash
# Experiment C (113M): Full-supervision Focal Loss - Stage 1 + Stage 2
set -e
DIR="$(dirname "$0")"
python train_stage1.py --config configs/stage1_focal_full_113M.yaml --device cuda
python train_stage2.py --config configs/stage2_focal_full_113M.yaml --device cuda
echo "Experiment C (113M) done."
