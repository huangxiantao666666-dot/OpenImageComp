#!/bin/bash
# Experiment A2: Dilated CrossEntropy (r=3) - Stage 2
cd "$(dirname "$0")/.."
python train_stage2.py --config configs/stage2_ce_dilated.yaml --device cuda
