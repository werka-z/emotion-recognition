# Improved CNN (raw pixels) — FER-2013

An improved version of the [baseline CNN](../baseline_CNN/) on 48×48 grayscale
faces, 7 emotion classes. Same dataset, same seeded 10% val split, same test
metrics and FLOPs accounting as the baseline — only the model and training
recipe change, so any accuracy gain is attributable to those changes.

## Files
- `model.py` — `EmotionCNNv2` (4 conv blocks 64→128→256→512 + Global Average Pooling head)
- `data_utils.py` — ImageFolder loaders with stronger augmentation, identical 10% val split (seed 42)
- `train.py` — training loop, cosine LR + warmup, label smoothing, saves `checkpoints/best.pt` and `results/learning_curve.png`
- `test.py` — accuracy, macro-F1, weighted-F1, per-class report, confusion matrix (+ optional `--tta`)
- `logs.txt` — full training log
- `RESULTS.md` — test-set results and baseline comparison

Plotting (`learning_curve.png`, `confusion_matrix.png`, `metrics.json/.txt`) reuses
the shared, model-agnostic helpers in [../../visualize.py](../../visualize.py),
so the baseline and improved model are charted identically.

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # from repo root

# From the repo root (paths auto-resolve to data/ and CNN/cnn_gap_aug_labelsmooth/):
python -m CNN.cnn_gap_aug_labelsmooth.train                   # ~40 epochs on MPS
python -m CNN.cnn_gap_aug_labelsmooth.test                    # evaluate best checkpoint on data/test
python -m CNN.cnn_gap_aug_labelsmooth.test --tta              # + test-time hflip augmentation
```

## What changed vs. the baseline, and why

The baseline overfits — it ends at train acc 0.74 vs val 0.66, and its weakest
classes are `fear` (F1 0.47) and `sad` (0.54). Every change below targets that.

| Change | Baseline | Improved | Why |
|---|---|---|---|
| Classifier head | `256·6·6 → 256` Linear (~2.4M params) | Global Average Pooling → `512 → 7` | Removes the biggest overfitting source; GAP generalizes better |
| Depth | 3 conv blocks (→256) | 4 conv blocks (→512) | A bit more capacity for hard classes |
| Augmentation | HFlip + Rotation(10°) | HFlip + Affine(rot/translate/scale) + RandomErasing | More regularization + occlusion robustness |
| Loss | class-weighted CE | class-weighted CE **+ label smoothing 0.1** | Robust to FER's noisy human labels |
| LR schedule | ReduceLROnPlateau | Cosine annealing + linear warmup | Smoother decay, reliable small gain |
| Optimizer | Adam | AdamW (decoupled weight decay) | Cleaner regularization |
| Eval | argmax | optional hflip **TTA** (`--tta`) | Near-free accuracy at test time |

Kept identical for a fair comparison: dataset, seeded 10% val split, normalization,
class weighting, best-checkpoint-by-val-acc selection, and the 3× forward FLOPs
training-compute estimate.
