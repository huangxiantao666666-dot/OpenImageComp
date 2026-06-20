#!/bin/bash
# Experiment: Fixed attention + global MLP, partial Focal - Stage 1 + Stage 2
set -e
DIR="$(dirname "$0")"
bash "$DIR/expB_focal_globalMLP_stage1.sh"
python train_stage2.py --config configs/stage2_focal_globalMLP.yaml --device cuda
echo "Experiment (focal globalMLP) done."
