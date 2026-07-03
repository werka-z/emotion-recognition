# 4-way fusion ablation

Isolates whether landmark geometry helps the CNN, and whether the
*learned gating dynamics* matter or just the extra parameters.
All fusion variants fine-tuned from `cnn_gap_aug_labelsmooth/checkpoints/best.pt`
(backbone frozen for 5 epochs, then unfrozen), 40 epochs, evaluated
without TTA. CNN-only row is the existing TTA-evaluated baseline (see
`CNN/cnn_gap_aug_labelsmooth/RESULTS.md`).

| Variant              | Accuracy | Macro F1 | alpha_final |
|----------------------|----------|----------|-------------|
| CNN only             | 0.6956   | 0.6611   | —           |
| CNN + concat         | 0.7019   | 0.6818   | —           |
| CNN + gated (α=0)    | 0.6804   | 0.6375   | 0.0554      |
| CNN + gated (frozen) | 0.6900   | 0.6489   | 1.0 (frozen)|

## Reading the table

- **alpha_final > 0.1**: geometry is genuinely helping; the gate learned to use it.
- **alpha_final ≈ 0**: the CNN already captured this information; geometry adds nothing.
- Compare gated vs frozen to see whether *learning* the gate (vs. always trusting
  geometry) matters.
- Compare gated vs concat to see whether the residual-gated fusion mechanism beats
  naive concatenation.
