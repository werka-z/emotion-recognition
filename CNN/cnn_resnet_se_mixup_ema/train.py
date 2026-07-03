"""Train the ResNet-SE FER-2013 CNN on Apple MPS.

Third model in the series. Architecture (EmotionResNetSE: pre-activation residual
blocks + Squeeze-Excitation + GAP head) lives in model.py. This file adds the
training recipe upgrades over cnn_gap_aug_labelsmooth, all aimed at squeezing the
last points out of FER-2013 raw pixels:

- MixUp / CutMix (in data_utils.py's train collate). Batches arrive as
  (x, y_a, y_b, lam); the loss is the lam-weighted mix of the two targets. This
  is the strongest remaining regularizer for FER's noisy labels + overfitting.
- Exponential Moving Average (EMA) of the weights. A shadow copy of the model is
  updated each step as ema = decay*ema + (1-decay)*model; the EMA weights are what
  we VALIDATE and SAVE. EMA smooths FER's noisy val curve and is a near-free win.
- Longer cosine schedule (default 60 epochs, "max it out") with linear warmup.
- Same class-weighted + label-smoothed CrossEntropy, AdamW, seeded 10% val split,
  best-by-val-acc checkpoint, and the SAME 3x-forward FLOPs accounting as the
  baseline / cnn_gap_aug_labelsmooth, so compute is reported on the same footing.

Usage:
    python -m CNN.cnn_resnet_se_mixup_ema.train               # from project root
    python -m CNN.cnn_resnet_se_mixup_ema.train --epochs 60
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

try:
    from CNN.cnn_resnet_se_mixup_ema.model import EmotionResNetSE
    from CNN.cnn_resnet_se_mixup_ema.data_utils import get_loaders
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import EmotionResNetSE
    from data_utils import get_loaders

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from visualize import plot_learning_curve

_DEFAULT_DATA = os.path.join(_REPO_ROOT, "data")
_DEFAULT_CKPT = os.path.join(_HERE, "checkpoints")
_DEFAULT_RESULTS = os.path.join(_HERE, "results")


class EMA:
    """Exponential moving average of model parameters (and buffers).

    store(): snapshot current model weights. copy_to(): load EMA weights into the
    model (used before eval). restore(): put the live training weights back.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(m, alpha=1 - d)
        # buffers (BN running stats) tracked directly — use the latest.
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)

    def copy_to(self, model):
        self._backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow.state_dict())

    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None


def count_forward_flops_per_sample(model, x):
    """FLOPs for one forward pass, per-sample (analytical). Same as baseline/mod."""
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
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * cosine)


def mix_criterion(criterion, out, y_a, y_b, lam):
    """lam-weighted loss over the two MixUp/CutMix targets (lam=1 -> plain loss)."""
    return lam * criterion(out, y_a) + (1.0 - lam) * criterion(out, y_b)


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
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-mixup", action="store_true", help="disable MixUp/CutMix")
    p.add_argument("--ckpt-dir", default=_DEFAULT_CKPT)
    p.add_argument("--results-dir", default=_DEFAULT_RESULTS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    train_loader, val_loader, classes = get_loaders(
        args.data_dir, args.batch_size, args.val_split, args.num_workers,
        args.seed, use_mixup=not args.no_mixup)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | "
          f"MixUp/CutMix: {not args.no_mixup}")

    model = EmotionResNetSE(num_classes=len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    # Same FLOPs accounting as baseline/mod: forward FLOPs/sample, train ~= 3x.
    sample_x, *_ = next(iter(train_loader))
    forward_flops_per_sample = count_forward_flops_per_sample(model, sample_x.to(device))
    train_flops_per_sample = 3 * forward_flops_per_sample
    print(f"Forward FLOPs/sample: {forward_flops_per_sample:.3e} | "
          f"Train (fwd+bwd) FLOPs/sample: {train_flops_per_sample:.3e}")

    # Class-weighted + label-smoothed loss (same as mod) to counter imbalance.
    base = train_loader.dataset.dataset
    idxs = train_loader.dataset.indices
    targets = torch.tensor([base.targets[i] for i in idxs])
    counts = torch.bincount(targets, minlength=len(classes)).float()
    class_weights = (counts.sum() / (len(classes) * counts)).to(device)
    print("Class weights:", {c: round(w.item(), 3) for c, w in zip(classes, class_weights)})

    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    ema = EMA(model, decay=args.ema_decay)

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
        for x, y_a, y_b, lam in train_loader:
            x, y_a, y_b = x.to(device), y_a.to(device), y_b.to(device)

            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            out = model(x)
            loss = mix_criterion(criterion, out, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
            ema.update(model)
            global_step += 1

            run_loss += loss.item() * x.size(0)
            # train acc measured against the dominant (lam-weighted) target — a
            # rough proxy under mixing, shown only for the learning curve.
            correct += (lam * (out.argmax(1) == y_a).sum().item()
                        + (1 - lam) * (out.argmax(1) == y_b).sum().item())
            seen += x.size(0)

        train_loss, train_acc = run_loss / seen, correct / seen

        # Validate (and checkpoint) the EMA weights, not the raw training weights.
        ema.copy_to(model)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        ema.restore(model)
        total_samples_seen += seen

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": ema.shadow.state_dict(),
                        "classes": classes,
                        "val_acc": val_acc,
                        "epoch": epoch}, best_path)
            flag = "  <- saved best (EMA)"

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"lr {cur_lr:.1e} | {time.time()-t0:.0f}s{flag}")

    total_train_flops = train_flops_per_sample * total_samples_seen

    history_path = os.path.join(args.results_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    curve_path = os.path.join(args.results_dir, "learning_curve.png")
    plot_learning_curve(history, title="ResNet-SE + MixUp + EMA CNN",
                        save_path=curve_path)

    print(f"\nBest val accuracy (EMA): {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Training history: {history_path}")
    print(f"Learning curve: {curve_path}")
    print(f"Total training compute: {total_train_flops:.3e} FLOPs "
          f"({total_samples_seen} samples seen)")


if __name__ == "__main__":
    main()
