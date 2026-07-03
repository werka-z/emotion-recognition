"""Train the gated landmark-fusion CNN on FER-2013.

Mirrors CNN/cnn_gap_aug_labelsmooth/train.py's recipe (AdamW, cosine warmup LR,
label smoothing, class weights, seeded 10% val split) but the model takes three
inputs (image, landmark vector, mask) and logs the fusion gate alpha every epoch
— the key diagnostic for how much the model relies on geometry.

Usage:
    python -m CNN.fusion_gated.train                       # gated, from scratch
    python -m CNN.fusion_gated.train --finetune             # gated, pretrained backbone
    python -m CNN.fusion_gated.train --ablation concat      # variant 2
    python -m CNN.fusion_gated.train --ablation frozen      # variant 4
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

try:
    from CNN.fusion_gated.model import FusionCNN
    from CNN.fusion_gated.dataset import get_loaders
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import FusionCNN
    from dataset import get_loaders

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from visualize import plot_learning_curve

_DEFAULT_CKPT = os.path.join(_HERE, "checkpoints")
_DEFAULT_RESULTS = os.path.join(_HERE, "results")
_GAP_BACKBONE_CKPT = os.path.join(_REPO_ROOT, "CNN", "cnn_gap_aug_labelsmooth",
                                  "checkpoints", "best.pt")

ABLATION_TO_MODE = {"gated": "gated", "concat": "concat", "frozen": "frozen"}


def count_forward_flops_per_sample(model, x, lm, mask):
    with FlopCounterMode(model, display=False) as fc:
        model(x, lm, mask)
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


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, lm, mask, y in loader:
        x, lm, mask, y = x.to(device), lm.to(device), mask.to(device), y.to(device)
        out = model(x, lm, mask)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
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
    p.add_argument("--ablation", choices=["concat", "frozen"], default=None,
                    help="run an ablation variant instead of the default gated model")
    p.add_argument("--finetune", action="store_true",
                    help="load pretrained backbone from cnn_gap_aug_labelsmooth, "
                         "freeze it for 5 epochs then unfreeze")
    args = p.parse_args()

    mode = ABLATION_TO_MODE[args.ablation] if args.ablation else "gated"

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device} | mode: {mode} | finetune: {args.finetune}")

    train_loader, val_loader, classes = get_loaders(
        args.batch_size, args.val_split, args.num_workers, args.seed)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = FusionCNN(lm_dim=26, num_classes=len(classes), mode=mode).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    freeze_epochs = 0
    if args.finetune:
        n_loaded = model.load_backbone_from(_GAP_BACKBONE_CKPT, map_location=device)
        print(f"Loaded {n_loaded} backbone tensors from {_GAP_BACKBONE_CKPT}")
        freeze_epochs = 5
        model.set_backbone_requires_grad(False)
        print(f"Backbone frozen for the first {freeze_epochs} epochs")

    sample_x, sample_lm, sample_mask, _ = next(iter(train_loader))
    forward_flops_per_sample = count_forward_flops_per_sample(
        model, sample_x.to(device), sample_lm.to(device), sample_mask.to(device))
    train_flops_per_sample = 3 * forward_flops_per_sample
    print(f"Forward FLOPs/sample: {forward_flops_per_sample:.3e} | "
          f"Train (fwd+bwd) FLOPs/sample: {train_flops_per_sample:.3e}")

    base = train_loader.dataset.dataset.folder  # FusionDataset.folder behind the Subset
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
    ckpt_name = "best.pt" if mode == "gated" else f"best_{mode}.pt"
    best_path = os.path.join(args.ckpt_dir, ckpt_name)
    best_val_acc = 0.0
    total_samples_seen = 0
    global_step = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
              "alpha_gate": []}
    log_lines = []

    for epoch in range(1, args.epochs + 1):
        if args.finetune and epoch == freeze_epochs + 1:
            model.set_backbone_requires_grad(True)
            print(f"Epoch {epoch}: backbone unfrozen")

        model.train()
        t0 = time.time()
        run_loss, seen, correct = 0.0, 0, 0
        for x, lm, mask, y in train_loader:
            x, lm, mask, y = x.to(device), lm.to(device), mask.to(device), y.to(device)

            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            out = model(x, lm, mask)
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

        if mode in ("gated", "frozen"):
            alpha_gate = torch.tanh(model.fusion.alpha).item()
        else:
            alpha_gate = float("nan")  # concat has no scalar gate

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["alpha_gate"].append(alpha_gate)

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "classes": classes,
                        "mode": mode,
                        "val_acc": val_acc,
                        "epoch": epoch}, best_path)
            flag = "  <- saved best"

        cur_lr = optimizer.param_groups[0]["lr"]
        line = (f"epoch {epoch:2d}  train_loss {train_loss:.4f}  "
                f"val_acc {val_acc:.4f}  alpha_gate {alpha_gate:.4f}")
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"alpha_gate {alpha_gate:.4f} | "
              f"lr {cur_lr:.1e} | {time.time()-t0:.0f}s{flag}")
        log_lines.append(line)

    total_train_flops = train_flops_per_sample * total_samples_seen

    history_path = os.path.join(args.results_dir, f"history_{mode}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    log_path = os.path.join(args.results_dir, f"alpha_log_{mode}.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    curve_path = os.path.join(args.results_dir, f"learning_curve_{mode}.png")
    plot_learning_curve(history, title=f"Fusion CNN ({mode})", save_path=curve_path)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Training history: {history_path}")
    print(f"Alpha log: {log_path}")
    print(f"Learning curve: {curve_path}")
    print(f"Total training compute: {total_train_flops:.3e} FLOPs "
          f"({total_samples_seen} samples seen)")


if __name__ == "__main__":
    main()
