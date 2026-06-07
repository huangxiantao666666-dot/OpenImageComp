#!/bin/bash
# Experiment C (113M): Full-supervision Focal Loss - Stage 1 + Stage 2
set -e
DIR="$(dirname "$0")"
bash "$DIR/expC_focal_full_113M_stage1.sh"
bash "$DIR/expC_focal_full_113M_stage2.sh"
echo "Experiment C (113M) done."
