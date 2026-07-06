"""Plot a confusion matrix and learning curve for any trained CNN, and save
a text/JSON sidecar of the underlying numbers for easy report extraction.

Model-agnostic across the CNN variants in this repo: works with anything
that produces y_true/y_pred arrays (logits.argmax(1)). Pass the
predictions/history in, get a figure out.

Usage:
    from CNN.visualize import plot_confusion_matrix, plot_learning_curve

    plot_confusion_matrix(y_true, y_pred, classes, title="Baseline CNN",
                           save_path="CNN/baseline_CNN/results/confusion_matrix.png")
    # Also writes results/metrics.json (and a .txt counterpart) alongside it

    plot_learning_curve(history, title="Baseline CNN",
                         save_path="CNN/baseline_CNN/results/learning_curve.png")
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)


def _save_metrics_sidecar(save_path, y_true, y_pred, classes, acc, macro_f1,
                           weighted_f1, cm):
    """Write metrics.json + metrics.txt next to save_path's confusion matrix PNG."""
    results_dir = os.path.dirname(save_path)
    report = classification_report(y_true, y_pred, target_names=classes,
                                    digits=4, output_dict=True)

    payload = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {c: report[c] for c in classes},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": (
            cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        ).tolist(),
        "classes": classes,
    }

    json_path = os.path.join(results_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    txt_path = os.path.join(results_dir, "metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Accuracy     : {acc:.4f}\n")
        f.write(f"Macro F1     : {macro_f1:.4f}\n")
        f.write(f"Weighted F1  : {weighted_f1:.4f}\n\n")
        f.write("Per-class report:\n")
        f.write(classification_report(y_true, y_pred, target_names=classes, digits=4))
        f.write("\nConfusion matrix, row-normalized 0-1 (rows = true, cols = pred):\n")
        header = "          " + " ".join(f"{c[:6]:>7}" for c in classes)
        f.write(header + "\n")
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        for i, c in enumerate(classes):
            row = " ".join(f"{v:>7.3f}" for v in cm_norm[i])
            f.write(f"{c[:9]:>9} {row}\n")

    return json_path, txt_path


def plot_confusion_matrix(y_true, y_pred, classes, title=None, save_path=None, ax=None):
    """Draw a confusion matrix heatmap, row-normalized to 0-1, with
    accuracy/macro-F1/weighted-F1 below it. If save_path is given, also writes
    metrics.json and metrics.txt (full per-class report + confusion matrix
    numbers) into the same directory for easy extraction into a report.

    Args:
        y_true, y_pred: 1D arrays/lists of integer class indices (same length).
        classes: list of class names, in the order used by the label indices.
        title: optional plot title.
        save_path: if given, save the figure to this path (e.g. "out.png")
            and write metrics.json/.txt alongside it.
        ax: optional existing matplotlib Axes to draw into. If omitted, a new
            figure is created (and shown/saved by this function).

    Returns:
        The matplotlib Figure.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    im = ax.imshow(cm_display, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)

    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            value = cm_display[i, j]
            ax.text(j, i, f"{value:.2f}",
                    ha="center", va="center",
                    color="white" if value > thresh else "black")

    metrics_text = (f"Accuracy: {acc:.4f}    "
                     f"Macro F1: {macro_f1:.4f}    "
                     f"Weighted F1: {weighted_f1:.4f}")
    fig.text(0.5, 0.02, metrics_text, ha="center", va="bottom", fontsize=11)

    if own_fig:
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            _save_metrics_sidecar(save_path, y_true, y_pred, classes,
                                   acc, macro_f1, weighted_f1, cm)
        else:
            plt.show()

    return fig


def plot_learning_curve(history, title=None, save_path=None):
    """Plot per-epoch train/val loss and accuracy side by side.

    Args:
        history: dict with any of the keys "train_loss", "val_loss",
            "train_acc", "val_acc", each mapping to a list of per-epoch
            values (epoch 1, 2, 3, ... in order). Missing keys are skipped,
            so this also works for models that only log a subset (e.g. a
            classical model with just train/val accuracy).
        title: optional figure title.
        save_path: if given, save the figure to this path (e.g. "out.png").

    Returns:
        The matplotlib Figure.
    """
    has_loss = "train_loss" in history or "val_loss" in history
    has_acc = "train_acc" in history or "val_acc" in history
    n_panels = sum([has_loss, has_acc]) or 1

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    axes = np.atleast_1d(axes)

    panel = 0
    if has_loss:
        ax = axes[panel]
        if "train_loss" in history:
            epochs = range(1, len(history["train_loss"]) + 1)
            ax.plot(epochs, history["train_loss"], label="train")
        if "val_loss" in history:
            epochs = range(1, len(history["val_loss"]) + 1)
            ax.plot(epochs, history["val_loss"], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        panel += 1

    if has_acc:
        ax = axes[panel]
        if "train_acc" in history:
            epochs = range(1, len(history["train_acc"]) + 1)
            ax.plot(epochs, history["train_acc"], label="train")
        if "val_acc" in history:
            epochs = range(1, len(history["val_acc"]) + 1)
            ax.plot(epochs, history["val_acc"], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy")
        ax.legend()
        panel += 1

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    cm_p = sub.add_parser("confusion-matrix", help="Plot a confusion matrix")
    cm_p.add_argument("--y-true", required=True, help="Path to .npy file of true labels")
    cm_p.add_argument("--y-pred", required=True, help="Path to .npy file of predicted labels")
    cm_p.add_argument("--classes", nargs="+", required=True, help="Class names in label order")
    cm_p.add_argument("--title", default=None)
    cm_p.add_argument("--save-path", default=None)

    lc_p = sub.add_parser("learning-curve", help="Plot a learning curve")
    lc_p.add_argument("--history", required=True,
                       help="Path to a JSON file with train_loss/val_loss/train_acc/val_acc lists")
    lc_p.add_argument("--title", default=None)
    lc_p.add_argument("--save-path", default=None)

    args = p.parse_args()

    if args.command == "confusion-matrix":
        y_true = np.load(args.y_true)
        y_pred = np.load(args.y_pred)
        plot_confusion_matrix(y_true, y_pred, args.classes, title=args.title,
                               save_path=args.save_path)
    elif args.command == "learning-curve":
        with open(args.history) as f:
            history = json.load(f)
        plot_learning_curve(history, title=args.title, save_path=args.save_path)
