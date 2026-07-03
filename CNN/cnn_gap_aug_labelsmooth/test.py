"""Evaluate the improved CNN on the FER-2013 test set.

Reports the same metrics as the baseline (accuracy, macro-F1, weighted-F1,
per-class report, confusion matrix) so the two are directly comparable, and
saves a confusion-matrix PNG to results/.

Adds optional test-time augmentation (--tta): average the logits of each image
and its horizontal flip. Faces are ~symmetric, so this is a near-free accuracy
bump and changes nothing about training.

Usage:
    python -m CNN.cnn_gap_aug_labelsmooth.test               # uses checkpoints/best.pt
    python -m CNN.cnn_gap_aug_labelsmooth.test --tta         # with test-time augmentation
    python test.py --ckpt path/to.pt     # from inside CNN/mod
"""

import argparse
import os
import sys
import torch

try:
    from CNN.cnn_gap_aug_labelsmooth.model import EmotionCNNv2
    from CNN.cnn_gap_aug_labelsmooth.data_utils import get_test_loader
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import EmotionCNNv2
    from data_utils import get_test_loader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from visualize import plot_confusion_matrix

_DEFAULT_DATA = os.path.join(_REPO_ROOT, "data")
_DEFAULT_CKPT = os.path.join(_HERE, "checkpoints", "best.pt")
_DEFAULT_RESULTS = os.path.join(_HERE, "results")


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def collect_preds(model, loader, device, tta=False):
    model.eval()
    all_y, all_p = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        if tta:
            out = out + model(torch.flip(x, dims=[3]))  # average with hflip
        all_p.append(out.argmax(1).cpu())
        all_y.append(y)
    return torch.cat(all_y).numpy(), torch.cat(all_p).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=_DEFAULT_DATA)
    p.add_argument("--ckpt", default=_DEFAULT_CKPT)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--tta", action="store_true", help="test-time hflip augmentation")
    p.add_argument("--results-dir", default=_DEFAULT_RESULTS)
    args = p.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    test_loader, classes = get_test_loader(args.data_dir, args.batch_size, args.num_workers)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = EmotionCNNv2(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded {args.ckpt} (trained {ckpt.get('epoch','?')} epochs, "
          f"val_acc {ckpt.get('val_acc', float('nan')):.4f}) | TTA={args.tta}")

    y_true, y_pred = collect_preds(model, test_loader, device, tta=args.tta)

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
    plot_confusion_matrix(y_true, y_pred, classes, title="Improved CNN",
                           save_path=cm_path)
    print(f"\nConfusion matrix image: {cm_path}")
    print(f"Metrics sidecar: {os.path.join(args.results_dir, 'metrics.json')}, "
          f"{os.path.join(args.results_dir, 'metrics.txt')}")


if __name__ == "__main__":
    main()
