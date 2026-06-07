#!/bin/bash
# Experiment A2: Dilated CrossEntropy (r=3) - Stage 1 + Stage 2
set -e
DIR="$(dirname "$0")"
bash "$DIR/expA_ce_dilated_stage1.sh"
python train_stage2.py --config configs/stage2_ce_dilated.yaml --device cuda
echo "Experiment A2 (dilated) done."
