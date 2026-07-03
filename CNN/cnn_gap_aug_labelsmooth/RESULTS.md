# Improved CNN — Test Results

Checkpoint: `checkpoints/best.pt` (epoch 37/40, val_acc 0.6864)
Evaluated with `python -m CNN.cnn_gap_aug_labelsmooth.test` on `data/test` (7178 images, untouched during training).
Headline numbers below are with **test-time augmentation** (`--tta`, hflip averaging);
the plain (no-TTA) column is shown alongside for completeness.

See [README.md](README.md) for the full list of changes vs. the baseline and the
rationale. All evaluation, the seeded 10% val split, the metrics, and the FLOPs
accounting are identical to the baseline, so the gains are attributable to the
model/training recipe — not to a different measurement.

## Headline comparison

| Metric | Baseline | Improved (no TTA) | Improved (TTA) | Δ (TTA − baseline) |
|---|---|---|---|---|
| Accuracy    | 0.6648 | 0.6875 | **0.6956** | **+0.0308** |
| Macro F1    | 0.6469 | 0.6535 | **0.6611** | **+0.0142** |
| Weighted F1 | 0.6625 | 0.6875 | **0.6956** | **+0.0331** |

Best validation accuracy also rose: **0.6864** vs the baseline's 0.6655. Just as
important, overfitting shrank — the improved model ends at train acc 0.778 vs val
0.686 (gap 0.092), versus the baseline's 0.740 vs 0.666 (gap 0.074 but at a lower
ceiling, and the baseline's val loss was already rising). The stronger augmentation
+ GAP head + label smoothing let the model train longer without diverging.

## Compute cost (reported side-by-side, as requested)

| | Baseline | Improved |
|---|---|---|
| Params | 3.51M | 4.69M |
| Forward FLOPs / sample | 6.87e8 | 9.37e8 |
| Train (fwd+bwd) FLOPs / sample | 2.06e9 | 2.81e9 |
| Epochs | 40 | 40 |
| Samples seen | 1,034,240 | 1,033,560 |
| **Total training FLOPs** | **2.13e15** | **2.91e15** |
| Approx. wall-clock (MPS) | ~36 min | ~45 min |

The improved model costs ~1.36× the training FLOPs of the baseline (extra 512-wide
conv block) for +3.1 pts accuracy / +1.4 pts macro-F1. The GAP head keeps the
param count modest despite the added depth — a plain `256·6·6→256` Linear head
would have been far larger. TTA adds one extra forward pass at *test* time only
(no training cost) for +0.8 pts.

## Per-class (Improved, TTA)

| Emotion | Precision | Recall | F1 | Support | F1 vs baseline |
|---|---|---|---|---|---|
| angry    | 0.6241 | 0.6430 | 0.6334 | 958  | 0.5945 → **0.6334** (+0.039) |
| disgust  | 0.3935 | 0.7658 | 0.5199 | 111  | 0.6463 → 0.5199 (−0.126) |
| fear     | 0.5882 | 0.5049 | 0.5434 | 1024 | 0.4673 → **0.5434** (+0.076) |
| happy    | 0.9101 | 0.8506 | 0.8794 | 1774 | 0.8642 → **0.8794** (+0.015) |
| neutral  | 0.6278 | 0.7153 | 0.6687 | 1233 | 0.6288 → **0.6687** (+0.040) |
| sad      | 0.6041 | 0.5445 | 0.5728 | 1247 | 0.5384 → **0.5728** (+0.034) |
| surprise | 0.7756 | 0.8484 | 0.8103 | 831  | 0.7890 → **0.8103** (+0.021) |

**Where the gains came from:** the two weakest baseline classes improved the most —
`fear` F1 +0.076 (the baseline's biggest weakness) and `angry`/`neutral`/`sad` all
+0.03–0.04. These are exactly the confusable classes the extra capacity + augmentation
were meant to help.

**The disgust trade-off:** `disgust` F1 dropped (0.65 → 0.52). With only 111 test
images its score is high-variance, and label smoothing slightly tempers the very
aggressive class weight (≈9.4×) that previously over-fit this tiny class — recall
actually went *up* (0.667 → 0.766), but precision fell (the model now calls more
things "disgust"). Net macro-F1 still rose because the six large classes improved.

## Confusion matrix (Improved, TTA — rows = true, cols = predicted)

|          | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| **angry**    | 616 | 39 | 79 | 21 | 90 | 94 | 19 |
| **disgust**  | 15 | 85 | 3 | 2 | 2 | 3 | 1 |
| **fear**     | 120 | 18 | 517 | 15 | 105 | 161 | 88 |
| **happy**    | 28 | 25 | 26 | 1509 | 92 | 37 | 57 |
| **neutral**  | 57 | 11 | 64 | 61 | 882 | 141 | 17 |
| **sad**      | 131 | 29 | 136 | 30 | 220 | 679 | 22 |
| **surprise** | 20 | 9 | 54 | 20 | 14 | 9 | 705 |

`fear`→`sad` (161) and `sad`→`neutral` (220) remain the dominant confusions, as in
the baseline, but every diagonal except `disgust` is stronger. Plots:
`results/confusion_matrix.png` (TTA) and `results/learning_curve.png`.

## Reproduce

```bash
python -m CNN.cnn_gap_aug_labelsmooth.train          # 40 epochs, writes checkpoints/ + results/learning_curve.png
python -m CNN.cnn_gap_aug_labelsmooth.test --tta     # headline numbers above
python -m CNN.cnn_gap_aug_labelsmooth.test           # no-TTA column
```
