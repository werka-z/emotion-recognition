"""Evaluate the gated landmark-fusion CNN on the FER-2013 test set.

Reports the same metrics as cnn_gap_aug_labelsmooth/test.py (accuracy, macro-F1,
weighted-F1, per-class report, confusion matrix) plus the final fusion gate
alpha — the key result telling us how much the model relies on geometry.

Usage:
    python -m CNN.fusion_gated.test                         # checkpoints/best.pt (gated)
    python -m CNN.fusion_gated.test --ckpt checkpoints/best_concat.pt --mode concat
"""

import argparse
import os
import sys
import torch

try:
    from CNN.fusion_gated.model import FusionCNN
    from CNN.fusion_gated.dataset import get_test_loader
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import FusionCNN
    from dataset import get_test_loader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from CNN.visualize import plot_confusion_matrix

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
    for x, lm, mask, y in loader:
        x, lm, mask = x.to(device), lm.to(device), mask.to(device)
        out = model(x, lm, mask)
        if tta:
            out = out + model(torch.flip(x, dims=[3]), lm, mask)
        all_p.append(out.argmax(1).cpu())
        all_y.append(y)
    return torch.cat(all_y).numpy(), torch.cat(all_p).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=_DEFAULT_CKPT)
    p.add_argument("--mode", choices=["gated", "concat", "frozen"], default=None,
                    help="fusion mode; defaults to the mode stored in the checkpoint")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--tta", action="store_true", help="test-time hflip augmentation")
    p.add_argument("--results-dir", default=_DEFAULT_RESULTS)
    args = p.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    test_loader, classes = get_test_loader(args.batch_size, args.num_workers)

    ckpt = torch.load(args.ckpt, map_location=device)
    mode = args.mode or ckpt.get("mode", "gated")
    model = FusionCNN(lm_dim=26, num_classes=len(classes), mode=mode).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded {args.ckpt} (mode={mode}, trained {ckpt.get('epoch','?')} epochs, "
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

    if mode in ("gated", "frozen"):
        alpha_final = torch.tanh(model.fusion.alpha).item()
        print(f"\n*** alpha_final (gate value) = {alpha_final:.4f} ***")
        if mode == "frozen":
            print("    (frozen ablation — gate was pinned at 1.0, never trained)")

    os.makedirs(args.results_dir, exist_ok=True)
    cm_path = os.path.join(args.results_dir, f"confusion_matrix_{mode}.png")
    plot_confusion_matrix(y_true, y_pred, classes, title=f"Fusion CNN ({mode})",
                           save_path=cm_path)

    # plot_confusion_matrix always writes metrics.json/.txt (shared infra used by
    # every CNN model folder). Since fusion's 3 ablation variants share one
    # results/ dir, immediately rename to a mode-specific name so a later variant
    # doesn't clobber an earlier one.
    import shutil
    metrics_json = os.path.join(args.results_dir, "metrics.json")
    metrics_txt = os.path.join(args.results_dir, "metrics.txt")
    metrics_json_mode = os.path.join(args.results_dir, f"metrics_{mode}.json")
    metrics_txt_mode = os.path.join(args.results_dir, f"metrics_{mode}.txt")
    shutil.move(metrics_json, metrics_json_mode)
    shutil.move(metrics_txt, metrics_txt_mode)

    print(f"\nConfusion matrix image: {cm_path}")
    print(f"Metrics sidecar: {metrics_json_mode}, {metrics_txt_mode}")


if __name__ == "__main__":
    main()
