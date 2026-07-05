"""Train the basic CNN on FER-2013 using Apple MPS.

Saves the best checkpoint (by validation accuracy) to checkpoints/best.pt.

Usage:
    python train.py                 # sensible defaults
    python train.py --epochs 40 --batch-size 64 --lr 1e-3
"""

import argparse
import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from CNN.baseline_CNN.model import EmotionCNN
from CNN.baseline_CNN.data_utils import get_loaders

# Anchor I/O to fixed locations so outputs land under CNN/baseline_CNN/ and
# the dataset is found regardless of the current working directory (e.g.
# when invoked as `python -m CNN.baseline_CNN.train` from the repo root
# instead of `python train.py` from this folder).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_DEFAULT_DATA = os.path.join(_REPO_ROOT, "data")
_DEFAULT_CKPT = os.path.join(_HERE, "checkpoints")
_DEFAULT_RESULTS = os.path.join(_HERE, "results")


def count_forward_flops_per_sample(model, x):
    """FLOPs for one forward pass at the given batch shape (analytical, not timed),
    normalized to a per-sample rate. BatchNorm requires batch size > 1 in train mode,
    so we count over the full batch and divide rather than using a single sample."""
    with FlopCounterMode(model, display=False) as fc:
        model(x)
    return fc.get_total_flops() / x.size(0)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    p.add_argument("--weight-decay", type=float, default=1e-4)
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

    model = EmotionCNN(num_classes=len(classes)).to(device)

    # Forward FLOPs/sample, measured analytically once on a single dummy batch.
    # Training FLOPs = forward + backward; backward costs ~2x forward, so we
    # use the standard 3x forward approximation (see Kaplan et al. scaling laws).
    sample_x, _ = next(iter(train_loader))
    forward_flops_per_sample = count_forward_flops_per_sample(model, sample_x.to(device))
    train_flops_per_sample = 3 * forward_flops_per_sample
    print(f"Forward FLOPs/sample: {forward_flops_per_sample:.3e} | "
          f"Train (fwd+bwd) FLOPs/sample: {train_flops_per_sample:.3e}")

    # Class-weighted loss to counter FER-2013 imbalance (disgust is tiny).
    counts = torch.zeros(len(classes))
    for _, ys in [(0, train_loader.dataset)]:
        pass
    # Compute class counts from the underlying ImageFolder targets of the train subset.
    base = train_loader.dataset.dataset  # ImageFolder behind the Subset
    idxs = train_loader.dataset.indices
    targets = torch.tensor([base.targets[i] for i in idxs])
    counts = torch.bincount(targets, minlength=len(classes)).float()
    class_weights = (counts.sum() / (len(classes) * counts)).to(device)
    print("Class weights:", {c: round(w.item(), 3) for c, w in zip(classes, class_weights)})

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    best_val_acc = 0.0
    total_samples_seen = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, seen, correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)

        train_loss, train_acc = run_loss / seen, correct / seen
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_acc)
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

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"lr {lr:.1e} | {time.time()-t0:.0f}s{flag}")

    total_train_flops = train_flops_per_sample * total_samples_seen

    history_path = os.path.join(args.results_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Training history: {history_path}")
    print(f"Total training compute: {total_train_flops:.3e} FLOPs "
          f"({total_samples_seen} samples seen)")


if __name__ == "__main__":
    main()
