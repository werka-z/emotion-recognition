"""Assemble results/ablation_summary.md from each variant's metrics.json + the
existing cnn_gap_aug_labelsmooth (CNN-only) numbers. Run after run_ablation.sh
(or after manually training+testing all three fusion variants).
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
RESULTS_DIR = os.path.join(_HERE, "results")
CKPT_DIR = os.path.join(_HERE, "checkpoints")

# Reported with TTA in cnn_gap_aug_labelsmooth/RESULTS.md.
CNN_ONLY = {"accuracy": 0.6956, "macro_f1": 0.6611}

VARIANTS = [
    ("CNN only", None, None),
    ("CNN + concat", "metrics_concat.json", "best_concat.pt"),
    ("CNN + gated (α=0)", "metrics_gated.json", "best.pt"),
    ("CNN + gated (frozen)", "metrics_frozen.json", "best_frozen.pt"),
]


def alpha_from_ckpt(ckpt_name, mode):
    import torch
    path = os.path.join(CKPT_DIR, ckpt_name)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state"]
    key = "fusion.alpha"
    if key not in state:
        return None
    return float(torch.tanh(state[key]).item())


def main():
    rows = []
    rows.append("| Variant              | Accuracy | Macro F1 | alpha_final |")
    rows.append("|----------------------|----------|----------|-------------|")
    rows.append(f"| CNN only             | {CNN_ONLY['accuracy']:.4f}   "
                f"| {CNN_ONLY['macro_f1']:.4f}   | —           |")

    for label, metrics_file, ckpt_name in VARIANTS[1:]:
        mpath = os.path.join(RESULTS_DIR, metrics_file)
        if not os.path.exists(mpath):
            rows.append(f"| {label:21s}| ?        | ?        | ?           |")
            continue
        with open(mpath) as f:
            m = json.load(f)
        acc = m["accuracy"]
        mf1 = m["macro_f1"]
        if "frozen" in metrics_file:
            alpha_str = "1.0 (frozen)"
        elif "concat" in metrics_file:
            alpha_str = "—"
        else:
            a = alpha_from_ckpt(ckpt_name, "gated")
            alpha_str = f"{a:.4f}" if a is not None else "?"
        rows.append(f"| {label:21s}| {acc:.4f}   | {mf1:.4f}   | {alpha_str:<12}|")

    table = "\n".join(rows)
    out = (
        "# 4-way fusion ablation\n\n"
        "Isolates whether landmark geometry helps the CNN, and whether the\n"
        "*learned gating dynamics* matter or just the extra parameters.\n"
        "All fusion variants fine-tuned from `cnn_gap_aug_labelsmooth/checkpoints/best.pt`\n"
        "(backbone frozen for 5 epochs, then unfrozen), 40 epochs, evaluated\n"
        "without TTA. CNN-only row is the existing TTA-evaluated baseline (see\n"
        "`CNN/cnn_gap_aug_labelsmooth/RESULTS.md`).\n\n"
        f"{table}\n\n"
        "## Reading the table\n\n"
        "- **alpha_final > 0.1**: geometry is genuinely helping; the gate learned to use it.\n"
        "- **alpha_final ≈ 0**: the CNN already captured this information; geometry adds nothing.\n"
        "- Compare gated vs frozen to see whether *learning* the gate (vs. always trusting\n"
        "  geometry) matters.\n"
        "- Compare gated vs concat to see whether the residual-gated fusion mechanism beats\n"
        "  naive concatenation.\n"
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "ablation_summary.md")
    with open(out_path, "w") as f:
        f.write(out)
    print(f"Wrote {out_path}\n")
    print(table)


if __name__ == "__main__":
    main()
