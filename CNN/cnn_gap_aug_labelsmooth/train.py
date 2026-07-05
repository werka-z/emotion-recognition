"""Train the improved CNN on FER-2013 using Apple MPS.

Improvements over CNN/baseline_CNN/train.py, all targeting the baseline's
overfitting (final train acc 0.74 vs val 0.66) and weak fear/sad classes:

- EmotionCNNv2 with a Global Average Pooling head (far fewer params than the
  baseline's 256*6*6 -> 256 Linear).  See model.py.
- Stronger augmentation (affine + random erasing).  See data_utils.py.
- Label smoothing 0.1 on top of the class-weighted loss (robustness to FER's
  noisy human labels; less overconfident logits).
- Cosine-annealing LR with a short linear warmup instead of ReduceLROnPlateau
  (smoother decay, a reliable point or two on FER).
- AdamW (decoupled weight decay) instead of Adam.
- Saves a learning-curve plot to results/learning_curve.png.

Everything else is kept identical to the baseline for a fair comparison: same
data, same seeded 10% val split, same FLOPs/compute accounting, best checkpoint
selected by validation accuracy.

Usage:
    python -m CNN.cnn_gap_aug_labelsmooth.train                  # from the project root (recommended)
    python train.py --epochs 40              # from inside CNN/mod (uses ../../data)
"""

import argparse
import json
import math
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

# Robust imports: work both as a package (python -m CNN.cnn_gap_aug_labelsmooth.train) and as a
# plain script run from inside this folder.
try:
    from CNN.cnn_gap_aug_labelsmooth.model import EmotionCNNv2
    from CNN.cnn_gap_aug_labelsmooth.data_utils import get_loaders
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import EmotionCNNv2
    from data_utils import get_loaders

# Shared, model-agnostic plotting lives at CNN/visualize.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from CNN.visualize import plot_learning_curve

# Anchor I/O to fixed locations so outputs land under CNN/cnn_gap_aug_labelsmooth/ and the dataset
# is found regardless of the current working directory.
_DEFAULT_DATA = os.path.join(_REPO_ROOT, "data")
_DEFAULT_CKPT = os.path.join(_HERE, "checkpoints")
_DEFAULT_RESULTS = os.path.join(_HERE, "results")


def count_forward_flops_per_sample(model, x):
    """FLOPs for one forward pass, normalized per-sample (analytical, not timed).
    BatchNorm needs batch>1 in train mode, so count over the batch and divide."""
    with FlopCounterMode(model, display=False) as fc:
        model(x)
    return fc.get_total_flops() / x.size(0)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr, min_lr_frac=0.02):
    """LR multiplier schedule: linear warmup then cosine decay to min_lr_frac*base."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * cosine)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=_DEFAULT_DATA)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ckpt-dir", default=_DEFAULT_CKPT)
    p.add_argument("--results-dir", default=_DEFAULT_RESULTS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    train_loader, val_loader, classes = get_loaders(
        args.data_dir, args.batch_size, args.val_split, args.num_workers, args.seed)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = EmotionCNNv2(num_classes=len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    # Forward FLOPs/sample (analytical). Train FLOPs = ~3x forward (fwd + ~2x bwd),
    # the standard scaling-law approximation — same accounting as the baseline.
    sample_x, _ = next(iter(train_loader))
    forward_flops_per_sample = count_forward_flops_per_sample(model, sample_x.to(device))
    train_flops_per_sample = 3 * forward_flops_per_sample
    print(f"Forward FLOPs/sample: {forward_flops_per_sample:.3e} | "
          f"Train (fwd+bwd) FLOPs/sample: {train_flops_per_sample:.3e}")

    # Class-weighted loss to counter FER-2013 imbalance (disgust is tiny),
    # combined with label smoothing for robustness to noisy labels.
    base = train_loader.dataset.dataset  # ImageFolder behind the Subset
    idxs = train_loader.dataset.indices
    targets = torch.tensor([base.targets[i] for i in idxs])
    counts = torch.bincount(targets, minlength=len(classes)).float()
    class_weights = (counts.sum() / (len(classes) * counts)).to(device)
    print("Class weights:", {c: round(w.item(), 3) for c, w in zip(classes, class_weights)})

    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    best_val_acc = 0.0
    total_samples_seen = 0
    global_step = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, seen, correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            global_step += 1

            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)

        train_loss, train_acc = run_loss / seen, correct / seen
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        total_samples_seen += seen

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "classes": classes,
                        "val_acc": val_acc,
                        "epoch": epoch}, best_path)
            flag = "  <- saved best"

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"lr {cur_lr:.1e} | {time.time()-t0:.0f}s{flag}")

    total_train_flops = train_flops_per_sample * total_samples_seen

    history_path = os.path.join(args.results_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    curve_path = os.path.join(args.results_dir, "learning_curve.png")
    plot_learning_curve(history, title="Improved CNN", save_path=curve_path)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Training history: {history_path}")
    print(f"Learning curve: {curve_path}")
    print(f"Total training compute: {total_train_flops:.3e} FLOPs "
          f"({total_samples_seen} samples seen)")


if __name__ == "__main__":
    main()
