#!/bin/bash
# Experiment A2: Dilated CrossEntropy (r=3) - Stage 1
cd "$(dirname "$0")/.."
python train_stage1.py --config configs/stage1_ce_dilated.yaml --device cuda
