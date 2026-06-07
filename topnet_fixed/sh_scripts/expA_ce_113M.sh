#!/bin/bash
# Experiment A (113M): Sparse CrossEntropy - Stage 1 + Stage 2
set -e
DIR="$(dirname "$0")"
bash "$DIR/expA_ce_113M_stage1.sh"
python train_stage2.py --config configs/stage2_ce_113M.yaml --device cuda
echo "Experiment A (113M) done."
