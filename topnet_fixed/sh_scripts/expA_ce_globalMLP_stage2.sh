#!/bin/bash
# Experiment: Fixed attention + global MLP, sparse CE - Stage 2
cd "$(dirname "$0")/.."
python train_stage2.py --config configs/stage2_ce_globalMLP.yaml --device cuda
