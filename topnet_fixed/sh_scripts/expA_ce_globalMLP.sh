#!/bin/bash
# Experiment: Fixed attention + global MLP, sparse CE - Stage 1 + Stage 2
set -e
DIR="$(dirname "$0")"
bash "$DIR/expA_ce_globalMLP_stage1.sh"
bash "$DIR/expA_ce_globalMLP_stage2.sh"
echo "Experiment (CE globalMLP) done."
