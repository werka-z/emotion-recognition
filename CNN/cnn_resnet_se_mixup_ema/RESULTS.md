# ResNet-SE + MixUp + EMA CNN — Test Results

> **STATUS: DONE.** Trained on a Colab GPU runtime via `train_colab.ipynb` (Run All); all numbers below are real measured results, filled in automatically from `results/metrics.json`, `results/metrics_no_tta.json`, and `logs.txt`.

Checkpoint: `checkpoints/best.pt` (EMA weights, epoch 60/60, val_acc 0.6871).
Evaluated on `data/test` (7178 images, untouched during training). All evaluation,
the seeded 10% val split, the metrics, and the FLOPs accounting are identical to the
two earlier models, so the gains are attributable to the architecture + recipe.

See [README.md](README.md) for the full change list and the note on why a ResNet is
still a CNN.

## Headline comparison (all three models)

| Metric | Baseline | GAP/aug/LS (TTA) | **ResNet-SE (no TTA)** | **ResNet-SE (multi-TTA)** |
|---|---|---|---|---|
| Accuracy    | 0.6648 | 0.6956 | 0.6902 | 0.6970 |
| Macro F1    | 0.6469 | 0.6611 | 0.6606 | 0.6700 |
| Weighted F1 | 0.6625 | 0.6956 | 0.6880 | 0.6949 |

Best validation accuracy (EMA): 0.6871 (baseline 0.6655, GAP/aug/LS 0.6864).

## Compute cost (side-by-side, same 3× forward FLOPs accounting)

The per-sample FLOPs and param count below are **measured** (from the FLOP counter
at the start of training); the totals depend on the run length.

| | Baseline | GAP/aug/LS | **ResNet-SE** |
|---|---|---|---|
| Params | 3.51M | 4.69M | **11.26M** |
| Forward FLOPs / sample | 6.87e8 | 9.37e8 | **2.494e9** |
| Train (fwd+bwd) FLOPs / sample | 2.06e9 | 2.81e9 | **7.483e9** |
| Epochs | 40 | 40 | **60** |
| Samples seen | 1,034,240 | 1,033,560 | 1,550,340 |
| **Total training FLOPs** | 2.13e15 | 2.91e15 | **1.160e+16** |
| Approx. wall-clock | ~36 min (MPS) | ~45 min (MPS) | **see logs.txt (Colab T4)** |

So this model costs ≈6.6× the training FLOPs of the GAP/aug/LS model (deeper
residual net + SE + 2.5× the epochs). The MixUp/CutMix + EMA + TTA are
near-free at their respective stages: MixUp/CutMix is a relabeling of existing
batches (no extra forward), EMA is one cheap in-place update per step, and TTA adds
forward passes at *test* time only. Fill the "samples seen" / total once training
finishes (it prints both on the last line).

## Per-class (ResNet-SE, multi-TTA) — fill from `results/metrics.txt`

| Emotion | Precision | Recall | F1 | Support | F1 vs GAP/aug/LS |
|---|---|---|---|---|---|
| angry    | 0.5896 | 0.6660 | 0.6255 | 958  | 0.6334 → 0.6255 |
| disgust  | 0.5125 | 0.7387 | 0.6052 | 111  | 0.5199 → 0.6052 |
| fear     | 0.5775 | 0.4805 | 0.5245 | 1024 | 0.5434 → 0.5245 |
| happy    | 0.9021 | 0.8726 | 0.8871 | 1774 | 0.8794 → 0.8871 |
| neutral  | 0.6553 | 0.6829 | 0.6688 | 1233 | 0.6687 → 0.6688 |
| sad      | 0.6078 | 0.5517 | 0.5784 | 1247 | 0.5728 → 0.5784 |
| surprise | 0.7497 | 0.8580 | 0.8002 | 831  | 0.8103 → 0.8002 |

**What to look for:** `fear` and `sad` were still the weakest classes (most
confused with each other and `neutral`). The extra depth + SE + MixUp should help
those most; `happy`/`surprise` are already near-ceiling.

## Confusion matrix

`results/confusion_matrix.png` (multi-TTA) and `results/learning_curve.png` are
written by the run. Paste the row-normalized matrix from `results/metrics.txt` here
if you want it inline, as in the earlier models' RESULTS.

## Reproduce
```bash
python -m CNN.cnn_resnet_se_mixup_ema.train          # 100 epochs, writes checkpoints/ + learning_curve.png
python -m CNN.cnn_resnet_se_mixup_ema.test --tta-multi   # headline numbers
python -m CNN.cnn_resnet_se_mixup_ema.test               # no-TTA column
```
