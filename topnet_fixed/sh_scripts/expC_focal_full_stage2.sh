#!/bin/bash
# Experiment C (Full Focal) - Stage 2
cd "$(dirname "$0")/.."
python train_stage2.py --config configs/stage2_focal_full.yaml --device cuda
