"""Compute total training FLOPs for an already-trained model, after the fact.

FLOPs/sample only depends on model architecture + input shape, not on training
history, so this can be run standalone without touching train.py or retraining.

Usage:
    python compute_flops.py --train-samples 25856 --epochs 40
    python compute_flops.py --train-batches 404 --batch-size 64 --epochs 40
"""

import argparse
import json
import os
import torch

from torch.utils.flop_counter import FlopCounterMode
from CNN.baseline_CNN.model import EmotionCNN


def count_forward_flops_per_sample(model, x):
    with FlopCounterMode(model, display=False) as fc:
        model(x)
    return fc.get_total_flops() / x.size(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--img-size", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--train-samples", type=int, default=None,
                    help="Total training samples per epoch. If omitted, "
                         "derive it from --train-batches * --batch-size.")
    p.add_argument("--train-batches", type=int, default=None,
                    help="Number of training batches per epoch (e.g. from logs.txt).")
    p.add_argument("--results-dir", default=None,
                    help="If given, write compute.json into this directory.")
    args = p.parse_args()

    if args.train_samples is not None:
        samples_per_epoch = args.train_samples
    elif args.train_batches is not None:
        samples_per_epoch = args.train_batches * args.batch_size
    else:
        raise SystemExit("Provide --train-samples or --train-batches")

    model = EmotionCNN(num_classes=args.num_classes)
    dummy = torch.randn(args.batch_size, 1, args.img_size, args.img_size)
    forward_flops_per_sample = count_forward_flops_per_sample(model, dummy)
    train_flops_per_sample = 3 * forward_flops_per_sample  # fwd + ~2x bwd

    total_samples = samples_per_epoch * args.epochs
    total_flops = train_flops_per_sample * total_samples

    print(f"Forward FLOPs/sample: {forward_flops_per_sample:.3e}")
    print(f"Train (fwd+bwd) FLOPs/sample: {train_flops_per_sample:.3e}")
    print(f"Samples/epoch: {samples_per_epoch} | Epochs: {args.epochs} | "
          f"Total samples seen: {total_samples}")
    print(f"Total training compute: {total_flops:.3e} FLOPs")

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        out_path = os.path.join(args.results_dir, "compute.json")
        with open(out_path, "w") as f:
            json.dump({
                "forward_flops_per_sample": forward_flops_per_sample,
                "train_flops_per_sample": train_flops_per_sample,
                "samples_per_epoch": samples_per_epoch,
                "epochs": args.epochs,
                "total_samples_seen": total_samples,
                "total_train_flops": total_flops,
            }, f, indent=2)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
