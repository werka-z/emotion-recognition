# Gated Landmark Fusion

Combines the `cnn_gap_aug_labelsmooth` CNN backbone with the 26 precomputed
geometric landmark features via a learned, gated residual connection.

## Architecture

```
image -> [EmotionCNNv2 conv stem + GAP] -> 512-d embedding ----+
                                                                 v
landmarks(26) -> [Linear-BN-ReLU-Linear] -> 512-d  -- * mask * tanh(alpha) -->  (+) -> Linear(512,7) -> logits
```

**Additive residual fusion with a scalar gate (`alpha`)**: at init `alpha=0`,
so `tanh(0)=0` and the model is a pure CNN — the optimizer only grows alpha if
geometry genuinely helps. Same principle as LayerScale / Flamingo gating,
applied to modality fusion. Concatenation has no equivalent clean zero-init
behavior, which is why additive gating was chosen over concat as the default.

**Per-sample mask**: images where dlib failed to detect a face get `mask=0`,
zeroing the landmark branch regardless of alpha — graceful degradation to pure
CNN, no imputation or average-vector fallback.

**Z-score normalization**: landmark features are normalized with mean/std
computed on the *present* training samples only (see "feature provenance"
below), applied identically at test time.

## Embedding dimension (deviation from the original spec)

The original instructions describe a 256-d embedding feeding `Linear(256,7)`.
The actual `cnn_gap_aug_labelsmooth` model (`CNN/cnn_gap_aug_labelsmooth/model.py`)
ends its conv stem at 512 channels, then GAP -> Dropout -> `Linear(512,7)`
directly — there is no intermediate 256-d projection. `FusionCNN` therefore uses
`cnn_dim=512` to exactly match the real backbone.

## Feature provenance (deviation from the original spec)

The original instructions assumed `feature_data/X_train.npy` etc. could be
joined to specific images via filename order. In practice those arrays were
saved as plain row-ordered floats with **no filenames or index**, so the join
is not recoverable after the fact — and the original extraction used dlib
19.24.2, which doesn't build in this environment (dlib 20.0.1 here), so even
replaying `extract_features.py`'s iteration order doesn't reproduce the same
per-image success/failure set or feature values.

Instead, `build_alignment.py` re-runs the identical 26-feature geometric
pipeline (same feature definitions, same upscale-then-detect-align-redetect
steps) directly over `data/train` and `data/test` in `ImageFolder` order,
producing a fresh per-image landmark vector + exact mask + a scaler re-fit on
the newly extracted training features. This keeps the fusion dataset fully
self-consistent; the landmark *values* differ slightly (sub-pixel-level, due to
the dlib version) from the SVM's saved `feature_data/`, but the feature
*definitions* and selected 26-feature set are identical.

## Files

- `build_alignment.py` — one-time step: extracts per-image landmarks + masks,
  caches to `cache/align_{train,test}.npz` and `cache/scaler.npz`. Run this
  first; takes a few minutes (dlib is CPU-only, no GPU/MPS path).
- `dataset.py` — `FusionDataset` / `get_loaders` / `get_test_loader`, joining
  `ImageFolder` images to the cached landmark/mask arrays. Reuses the exact
  image transforms from `cnn_gap_aug_labelsmooth/data_utils.py`.
- `model.py` — `FusionCNN` (backbone) + `GatedLandmarkFusion` / `ConcatFusion`
  (fusion heads, selected via `mode`).
- `train.py` — training loop; logs `alpha_gate` every epoch; supports
  `--finetune` (load pretrained backbone, freeze 5 epochs) and `--ablation
  {concat,frozen}` for the 4-way ablation.
- `test.py` — evaluation; reports accuracy/macro-F1/weighted-F1/per-class +
  confusion matrix + the final alpha gate value.

## Running

```bash
python -m CNN.fusion_gated.build_alignment        # one-time, ~8 min, CPU only

python -m CNN.fusion_gated.train --finetune        # main model (gated, fine-tuned)
python -m CNN.fusion_gated.test                    # evaluate checkpoints/best.pt

# ablation variants
python -m CNN.fusion_gated.train --finetune --ablation concat
python -m CNN.fusion_gated.train --finetune --ablation frozen
python -m CNN.fusion_gated.test --ckpt checkpoints/best_concat.pt
python -m CNN.fusion_gated.test --ckpt checkpoints/best_frozen.pt
```

See `results/ablation_summary.md` for the collected 4-way comparison.
