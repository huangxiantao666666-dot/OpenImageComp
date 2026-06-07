#!/bin/bash
# Experiment B (113M): Partial Focal Loss - Stage 1 + Stage 2
set -e
DIR="$(dirname "$0")"
bash "$DIR/expB_focal_113M_stage1.sh"
bash "$DIR/expB_focal_113M_stage2.sh"
echo "Experiment B (113M) done."
