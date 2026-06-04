#!/bin/bash
# Experiment A: Sparse CrossEntropy (baseline) — Stage 1 + Stage 2
set -e
DIR="$(dirname "$0")"
bash "$DIR/expA_ce_stage1.sh"
bash "$DIR/expA_ce_stage2.sh"
echo "Experiment A done."
