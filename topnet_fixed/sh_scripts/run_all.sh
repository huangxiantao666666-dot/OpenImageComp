#!/bin/bash
# Run all 3 experiments sequentially (6 training jobs total)
set -e
DIR="$(dirname "$0")"

echo "=============================================="
echo " Experiment A: Sparse CrossEntropy (baseline)"
echo "=============================================="
bash "$DIR/expA_ce.sh"

echo ""
echo "=============================================="
echo " Experiment B: Partial Focal Loss"
echo "=============================================="
bash "$DIR/expB_focal.sh"

echo ""
echo "=============================================="
echo " Experiment C: Full-supervision Focal Loss"
echo "=============================================="
bash "$DIR/expC_focal_full.sh"

echo ""
echo "All 3 experiments done."
