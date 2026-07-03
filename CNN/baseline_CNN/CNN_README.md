# CNN baseline (raw pixels) — FER-2013

Basic VGG-style CNN on 48×48 grayscale faces, 7 emotion classes. Trained on apple MPS.

## Files
- `model.py` — `EmotionCNN` (3 conv blocks 64→128→256 + dropout head)
- `data_utils.py` — ImageFolder loaders, augmentation, 10% val split from `data/train`
- `train.py` — training loop, class-weighted loss, saves `checkpoints/best.pt`
- `test.py` — accuracy, macro-F1, weighted-F1, per-class report, confusion matrix

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py            # ~40 epochs on MPS
python test.py             # evaluate best checkpoint on data/test
```

## Design choices (natural defaults for this dataset)
- **Conv–BN–ReLU ×2 + MaxPool** blocks, widths 64→128→256 — standard FER baseline.
- **BatchNorm** throughout — stabilizes training on noisy FER images.
- **Augmentation**: random horizontal flip + ±10° rotation (faces ~symmetric).
- **Class-weighted cross-entropy** — counters heavy imbalance (disgust ≈1.5%).
- **Adam (lr 1e-3, wd 1e-4)** + ReduceLROnPlateau on val accuracy.
- **Best checkpoint** selected by validation accuracy.

Expected test accuracy for this baseline: ~0.60–0.65 (FER-2013 human ≈ 0.65).
