"""Evaluate the ResNet-SE CNN on the FER-2013 test set.

Reports the same metrics as baseline / cnn_gap_aug_labelsmooth (accuracy,
macro-F1, weighted-F1, per-class report, confusion matrix) via the shared
visualize.py, so all three models are directly comparable. The loaded checkpoint
holds the EMA weights selected by best val accuracy.

TTA options (test-time only, no training cost):
    --tta        average logits over the image and its horizontal flip (as in mod)
    --tta-multi  multi-crop: hflip + a few small pixel shifts, averaged. A bit
                 stronger than plain hflip TTA for a few extra forward passes.

Usage:
    python -m CNN.cnn_resnet_se_mixup_ema.test --tta-multi   # headline numbers
    python -m CNN.cnn_resnet_se_mixup_ema.test               # no-TTA column
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F

try:
    from CNN.cnn_resnet_se_mixup_ema.model import EmotionResNetSE
    from CNN.cnn_resnet_se_mixup_ema.data_utils import get_test_loader
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import EmotionResNetSE
    from data_utils import get_test_loader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from CNN.visualize import plot_confusion_matrix

_DEFAULT_DATA = os.path.join(_REPO_ROOT, "data")
_DEFAULT_CKPT = os.path.join(_HERE, "checkpoints", "best.pt")
_DEFAULT_RESULTS = os.path.join(_HERE, "results")


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _shift(x, dy, dx):
    """Translate a batch by (dy, dx) pixels, zero-padding the exposed border."""
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


@torch.no_grad()
def collect_preds(model, loader, device, tta=False, tta_multi=False):
    model.eval()
    all_y, all_p = [], []
    for x, y in loader:
        x = x.to(device)
        logits = F.softmax(model(x), dim=1)
        if tta or tta_multi:
            logits = logits + F.softmax(model(torch.flip(x, dims=[3])), dim=1)
        if tta_multi:
            # a few small shifts (and their flips) — cheap multi-crop TTA
            for dy, dx in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
                xs = _shift(x, dy, dx)
                logits = logits + F.softmax(model(xs), dim=1)
                logits = logits + F.softmax(model(torch.flip(xs, dims=[3])), dim=1)
        all_p.append(logits.argmax(1).cpu())
        all_y.append(y)
    return torch.cat(all_y).numpy(), torch.cat(all_p).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=_DEFAULT_DATA)
    p.add_argument("--ckpt", default=_DEFAULT_CKPT)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--tta", action="store_true", help="hflip test-time augmentation")
    p.add_argument("--tta-multi", action="store_true",
                   help="multi-crop TTA (hflip + small shifts)")
    p.add_argument("--results-dir", default=_DEFAULT_RESULTS)
    args = p.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    test_loader, classes = get_test_loader(args.data_dir, args.batch_size, args.num_workers)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = EmotionResNetSE(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    mode = "multi-crop" if args.tta_multi else ("hflip" if args.tta else "none")
    print(f"Loaded {args.ckpt} (trained {ckpt.get('epoch','?')} epochs, "
          f"val_acc {ckpt.get('val_acc', float('nan')):.4f}) | TTA={mode}")

    y_true, y_pred = collect_preds(model, test_loader, device,
                                   tta=args.tta, tta_multi=args.tta_multi)

    from sklearn.metrics import accuracy_score, f1_score, classification_report

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("\n===== Test results =====")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro F1     : {macro_f1:.4f}")
    print(f"Weighted F1  : {weighted_f1:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    os.makedirs(args.results_dir, exist_ok=True)
    cm_path = os.path.join(args.results_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, classes, title="ResNet-SE + MixUp + EMA CNN",
                          save_path=cm_path)
    print(f"\nConfusion matrix image: {cm_path}")
    print(f"Metrics sidecar: {os.path.join(args.results_dir, 'metrics.json')}, "
          f"{os.path.join(args.results_dir, 'metrics.txt')}")


if __name__ == "__main__":
    main()
