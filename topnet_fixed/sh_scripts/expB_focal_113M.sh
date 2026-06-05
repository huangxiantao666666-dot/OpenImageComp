#!/bin/bash
# Experiment B (113M): Partial Focal Loss - Stage 1 + Stage 2
set -e
DIR="$(dirname "$0")"
bash "$DIR/expB_focal_113M_stage1.sh"
python train_stage2.py --config configs/stage2_focal_113M.yaml --device cuda
echo "Experiment B (113M) done."
