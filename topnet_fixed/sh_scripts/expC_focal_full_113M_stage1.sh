#!/bin/bash
# Experiment C (113M): Full-supervision Focal Loss - Stage 1
cd "$(dirname "$0")/.."
python train_stage1.py --config configs/stage1_focal_full_113M.yaml --device cuda
