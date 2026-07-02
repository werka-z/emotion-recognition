"""
plot_clahe.py

Plots detection rate results saved by compare_all_preprocessing.py.

Environment: dlib (see environment.yml) — only needs numpy and matplotlib

Usage:
    python ablation_studies/plot_clahe.py

Reads:
    ablation_results/clahe/detection_counts.npy

Saves:
    ablation_results/clahe/compare_clahe.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import EMOTION_LABELS, ablation_results_subdir

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR = ablation_results_subdir("clahe")

CONDITIONS = [
    ("no_up_no_cl", "48px, no CLAHE"),
    ("no_up_cl",    "48px, CLAHE"),
    ("up_no_cl",    "144px, no CLAHE"),
    ("up_cl",       "144px, CLAHE"),
]

# ── LOAD ──────────────────────────────────────────────────────────────────────
results = np.load(os.path.join(OUT_DIR, "detection_counts.npy"),
                   allow_pickle=True).item()

# ── PLOT ──────────────────────────────────────────────────────────────────────
n_conditions = len(CONDITIONS)
n_emotions   = len(EMOTION_LABELS)
x            = np.arange(n_emotions)
bar_width    = 0.18

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig.suptitle("Detection rate by preprocessing condition\n(dlib frontal face detector)")

for ax, split in zip(axes, ("train", "test")):
    for i, (prefix, label) in enumerate(CONDITIONS):
        rates = []
        for emotion in EMOTION_LABELS:
            s   = results[prefix][split][emotion]
            pct = 100 * s["detected"] / s["total"] if s["total"] > 0 else 0
            rates.append(pct)
        offset = (i - (n_conditions - 1) / 2) * bar_width
        bars   = ax.bar(x + offset, rates, bar_width, label=label)
        for bar, v in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=6)

    ax.set_title(f"{split.capitalize()} split")
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS, rotation=30, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Detection rate (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "compare_clahe.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved compare_clahe.png → {OUT_DIR}/")