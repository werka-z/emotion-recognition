#!/bin/bash
# Runs the full 4-way ablation sequentially (variant 1 needs no training — it's
# the existing cnn_gap_aug_labelsmooth numbers, reported as-is). Each variant
# trains for 40 epochs fine-tuned from the GAP backbone, then evaluates on the
# test set. Intended to be wrapped in `caffeinate -i` for a long unattended run.
#
# Usage:
#   caffeinate -i bash CNN/fusion_gated/run_ablation.sh
set -e
cd "$(dirname "$0")/../.."   # repo root

PY=.venv/bin/python

echo "=== [1/3] Variant: gated (default, alpha init=0) ==="
$PY -m CNN.fusion_gated.train --finetune
$PY -m CNN.fusion_gated.test --ckpt CNN/fusion_gated/checkpoints/best.pt

echo "=== [2/3] Variant: concat ==="
$PY -m CNN.fusion_gated.train --finetune --ablation concat
$PY -m CNN.fusion_gated.test --ckpt CNN/fusion_gated/checkpoints/best_concat.pt

echo "=== [3/3] Variant: frozen (alpha pinned=1) ==="
$PY -m CNN.fusion_gated.train --finetune --ablation frozen
$PY -m CNN.fusion_gated.test --ckpt CNN/fusion_gated/checkpoints/best_frozen.pt

echo "=== Building ablation_summary.md ==="
$PY -m CNN.fusion_gated.build_ablation_summary

echo "All done. See CNN/fusion_gated/results/ablation_summary.md"
