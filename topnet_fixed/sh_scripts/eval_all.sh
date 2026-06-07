#!/bin/bash
# Evaluate all 8 models sequentially — each in its own Python process
# so GPU memory is released between evaluations.
set -e
DIR="$(dirname "$0")/../evaluate"

echo "=============================================="
echo " Buggy TopNet (original)"
echo "=============================================="
python "$DIR/eval_buggy.py"

echo ""
echo "=============================================="
echo " Exp A: Sparse CrossEntropy (79M)"
echo "=============================================="
python "$DIR/eval_expA_ce.py"

echo ""
echo "=============================================="
echo " Exp A: Sparse CrossEntropy (113M)"
echo "=============================================="
python "$DIR/eval_expA_ce_113M.py"

echo ""
echo "=============================================="
echo " Exp A2: Dilated CrossEntropy (r=3)"
echo "=============================================="
python "$DIR/eval_expA_ce_dilated.py"

echo ""
echo "=============================================="
echo " Exp A2: Dilated CrossEntropy (113M)"
echo "=============================================="
python "$DIR/eval_expA_ce_dilated_113M.py"

echo ""
echo "=============================================="
echo " Exp B: Partial Focal (79M)"
echo "=============================================="
python "$DIR/eval_expB_focal.py"

echo ""
echo "=============================================="
echo " Exp B: Partial Focal (113M)"
echo "=============================================="
python "$DIR/eval_expB_focal_113M.py"

echo ""
echo "=============================================="
echo " Exp C: Full-supervision Focal (79M)"
echo "=============================================="
python "$DIR/eval_expC_focal_full.py"

echo ""
echo "=============================================="
echo " Exp C: Full-supervision Focal (113M)"
echo "=============================================="
python "$DIR/eval_expC_focal_full_113M.py"

echo ""
echo "All evaluations done. Results in evaluate/logs/"
