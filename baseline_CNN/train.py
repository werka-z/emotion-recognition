"""Train the basic CNN on FER-2013 using Apple MPS.

Saves the best checkpoint (by validation accuracy) to checkpoints/best.pt.

Usage:
    python train.py                 # sensible defaults
    python train.py --epochs 40 --batch-size 64 --lr 1e-3
"""

import argparse
import os
import time
import torch
import torch.nn as nn

from model import EmotionCNN
from data_utils import get_loaders


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
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    train_loader, val_loader, classes = get_loaders(
        args.data_dir, args.batch_size, args.val_split, args.num_workers, args.seed)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = EmotionCNN(num_classes=len(classes)).to(device)

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
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    best_val_acc = 0.0

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

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
