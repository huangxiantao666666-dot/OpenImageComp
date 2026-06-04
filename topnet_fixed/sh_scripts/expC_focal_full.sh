#!/bin/bash
# Experiment C: Full-supervision Focal Loss — Stage 1 + Stage 2
set -e
DIR="$(dirname "$0")"
bash "$DIR/expC_focal_full_stage1.sh"
bash "$DIR/expC_focal_full_stage2.sh"
echo "Experiment C done."
