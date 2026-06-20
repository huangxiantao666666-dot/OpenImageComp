#!/bin/bash
# Experiment: Fixed attention + buggy-style global MLP, sparse CE - Stage 1
cd "$(dirname "$0")/.."
python train_stage1.py --config configs/stage1_ce_globalMLP.yaml --device cuda
