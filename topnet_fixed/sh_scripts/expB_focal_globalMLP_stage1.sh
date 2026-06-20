#!/bin/bash
# Experiment: Fixed attention + global MLP, partial Focal Loss - Stage 1
cd "$(dirname "$0")/.."
python train_stage1.py --config configs/stage1_focal_globalMLP.yaml --device cuda
