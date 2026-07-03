"""Evaluate the trained CNN on the FER-2013 test set.

Reports overall accuracy, macro-F1, weighted-F1, a full per-class
precision/recall/F1 report, and the confusion matrix.

Usage:
    python test.py                      # uses checkpoints/best.pt
    python test.py --ckpt path/to.pt
"""

import argparse
import os
import torch

from CNN.baseline_CNN.model import EmotionCNN
from CNN.baseline_CNN.data_utils import get_test_loader
from visualize import plot_confusion_matrix


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        all_p.append(out.argmax(1).cpu())
        all_y.append(y)
    return torch.cat(all_y).numpy(), torch.cat(all_p).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    test_loader, classes = get_test_loader(args.data_dir, args.batch_size, args.num_workers)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = EmotionCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded {args.ckpt} (trained {ckpt.get('epoch','?')} epochs, "
          f"val_acc {ckpt.get('val_acc', float('nan')):.4f})")

    y_true, y_pred = collect_preds(model, test_loader, device)

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
    plot_confusion_matrix(y_true, y_pred, classes, title="Baseline CNN",
                           save_path=cm_path)
    print(f"\nConfusion matrix image: {cm_path}")
    print(f"Metrics sidecar: {os.path.join(args.results_dir, 'metrics.json')}, "
          f"{os.path.join(args.results_dir, 'metrics.txt')}")


if __name__ == "__main__":
    main()
