"""
plot_upscaling.py

Plots detection rate results saved by compare_upscaling.py.

Environment: dlib (see environment.yml) — only needs numpy and matplotlib

Usage:
    python plot_upscaling.py

Reads:
    processed_final/upscaling_results.npy

Saves:
    processed_final/compare_upscaling.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR        = os.path.join(BASE_DIR, "processed_final")
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
UPSCALE_FACTORS = [1, 2, 3, 4, 5, 6]

# ── LOAD ──────────────────────────────────────────────────────────────────────
results = np.load(os.path.join(OUT_DIR, "upscaling_results.npy"),
                  allow_pickle=True).item()

# ── PLOT ──────────────────────────────────────────────────────────────────────
labels      = [f"{f}x\n({48*f}px)" for f in UPSCALE_FACTORS]
n_factors   = len(UPSCALE_FACTORS)
n_emotions  = len(EMOTION_LABELS)
x           = np.arange(n_emotions)
bar_width   = 0.12

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig.suptitle("Detection rate by upscale factor\n(dlib frontal face detector)")

for ax, split in zip(axes, ("train", "test")):
    for i, factor in enumerate(UPSCALE_FACTORS):
        rates = []
        for emotion in EMOTION_LABELS:
            s   = results[factor][split][emotion]
            pct = 100 * s["detected"] / s["total"] if s["total"] > 0 else 0
            rates.append(pct)

        offset = (i - (n_factors - 1) / 2) * bar_width
        bars   = ax.bar(x + offset, rates, bar_width, label=labels[i])

        for bar, v in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=5)

    ax.set_title(f"{split.capitalize()} split")
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS, rotation=30, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Detection rate (%)")
    ax.legend(title="Upscale factor", fontsize=7, title_fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "compare_upscaling.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved compare_upscaling.png → {OUT_DIR}/")