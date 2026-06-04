#!/bin/bash
# Experiment B: Partial Focal Loss — Stage 1 + Stage 2
set -e
DIR="$(dirname "$0")"
bash "$DIR/expB_focal_stage1.sh"
bash "$DIR/expB_focal_stage2.sh"
echo "Experiment B done."
