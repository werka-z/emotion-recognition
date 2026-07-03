# ResNet-SE + MixUp + EMA CNN (raw pixels) — FER-2013

The third model in the series, after the [baseline CNN](../baseline_CNN/) and the
[GAP / augmentation / label-smoothing model](../cnn_gap_aug_labelsmooth/). Same
48×48 grayscale faces, 7 emotion classes, same dataset, same seeded 10% val split,
same test metrics and FLOPs accounting — only the model and training recipe change,
so any accuracy gain is attributable to those changes.

## Files
- `model.py` — `EmotionResNetSE`: a ResNet-18-style stack (pre-activation residual blocks 64→128→256→512, 2 blocks/stage) with a Squeeze-and-Excitation gate in every block and a Global Average Pooling head
- `data_utils.py` — ImageFolder loaders, identical 10% val split (seed 42), stronger train augmentation, and a MixUp/CutMix collate used by the train loader only
- `train.py` — training loop with MixUp/CutMix loss, EMA weights, cosine LR + warmup, label smoothing; saves `checkpoints/best.pt` (EMA weights) and `results/learning_curve.png`
- `test.py` — accuracy, macro-F1, weighted-F1, per-class report, confusion matrix (+ optional `--tta` / `--tta-multi`)
- `logs.txt` — full training log
- `RESULTS.md` — test-set results and comparison to the previous two models

Plotting reuses the shared, model-agnostic helpers in
[../../visualize.py](../../visualize.py), so all three models are charted identically.

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # from repo root

# From the repo root (paths auto-resolve to data/ and this folder):
python -m CNN.cnn_resnet_se_mixup_ema.train               # 100 epochs on MPS (~3.7h)
python -m CNN.cnn_resnet_se_mixup_ema.test                # evaluate best (EMA) checkpoint, no TTA
python -m CNN.cnn_resnet_se_mixup_ema.test --tta          # + test-time hflip
python -m CNN.cnn_resnet_se_mixup_ema.test --tta-multi    # + multi-crop TTA (headline numbers)
```

To save the training log alongside the previous models:
```bash
python -m CNN.cnn_resnet_se_mixup_ema.train 2>&1 | tee CNN/cnn_resnet_se_mixup_ema/logs.txt
```

## What changed vs. `cnn_gap_aug_labelsmooth`, and why

That model plateaued around val ≈ 0.69 with the train/val gap mostly closed by
augmentation + label smoothing. To gain more we need both more capacity (used
well) and a stronger regularizer. Each change below targets one of those.

| Change | `cnn_gap_aug_labelsmooth` | This model | Why |
|---|---|---|---|
| Architecture | Plain VGG-style stack (Conv-BN-ReLU ×2 + pool) | **Pre-activation Residual** blocks (`out = F(x)+x`) | Skip connections let the net go deeper (≈18 layers) and optimize cleanly — depth is the headroom for confusable classes |
| Channel attention | — | **Squeeze-and-Excitation** in every block | Per-image channel reweighting; cheap, reliable small gain on FER |
| Sample mixing | — | **MixUp + CutMix** (train only) | The strongest remaining regularizer; blends images/labels to fight overfitting + FER's noisy labels |
| Weight averaging | — | **EMA** of weights (evaluated & saved) | Smooths FER's noisy val curve; near-free accuracy |
| Augmentation | Affine + RandomErasing(0.25) | Wider affine + shear + ColorJitter + RandomErasing(0.35) | The deeper net + MixUp can absorb more aug before underfitting |
| Schedule | 40 epochs cosine | **100 epochs** cosine + warmup | "Max it out" — deeper net + heavy aug benefit from a longer schedule |
| Eval | argmax / hflip TTA | hflip TTA **+ optional multi-crop TTA** | A bit more test-time averaging for a few extra forward passes |

Kept identical for a fair comparison: dataset, seeded 10% val split (so the val set
matches the earlier models), normalization constants, class-weighted loss + label
smoothing, best-checkpoint-by-val-acc selection, and the 3× forward FLOPs
training-compute estimate. The cost goes up (≈11M params, deeper net, more epochs);
`RESULTS.md` reports compute side-by-side, as with the previous models.
