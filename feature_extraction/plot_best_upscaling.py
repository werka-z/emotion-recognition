"""
plot_upscaling_best.py

Plots detection rate comparing no upscaling (1x, 48px) vs the best
upscaling factor found by compare_upscaling.py.

The best factor is determined automatically as the one with the highest
overall detection rate on the train split.

Environment: dlib (see environment.yml) — only needs numpy and matplotlib

Usage:
    python plot_upscaling_best.py

Reads:
    processed_final/upscaling_results.npy

Saves:
    processed_final/compare_upscaling_best.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR         = os.path.join(BASE_DIR, "processed_final")
EMOTION_LABELS  = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
UPSCALE_FACTORS = [1, 2, 3, 4, 5, 6]

# ── LOAD ──────────────────────────────────────────────────────────────────────
results = np.load(os.path.join(OUT_DIR, "upscaling_results.npy"),
                  allow_pickle=True).item()

# ── FIND BEST FACTOR ──────────────────────────────────────────────────────────
best_factor = max(
    UPSCALE_FACTORS,
    key=lambda f: sum(results[f]["train"][e]["detected"]
                      for e in EMOTION_LABELS)
)
best_size = 48 * best_factor
print(f"Best upscale factor: {best_factor}x ({best_size}x{best_size} px)")

# ── PLOT ──────────────────────────────────────────────────────────────────────
x         = np.arange(len(EMOTION_LABELS))
bar_width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle(f"Detection rate: no upscaling (48px) vs best upscaling ({best_size}px)\n"
             f"(dlib frontal face detector)")

for ax, split in zip(axes, ("train", "test")):
    rates_base = []
    rates_best = []

    for emotion in EMOTION_LABELS:
        s_base = results[1][split][emotion]
        s_best = results[best_factor][split][emotion]
        rates_base.append(100 * s_base["detected"] / s_base["total"] if s_base["total"] > 0 else 0)
        rates_best.append(100 * s_best["detected"] / s_best["total"] if s_best["total"] > 0 else 0)

    bars_base = ax.bar(x - bar_width / 2, rates_base, bar_width,
                       label=f"48px (no upscale)")
    bars_best = ax.bar(x + bar_width / 2, rates_best, bar_width,
                       label=f"{best_size}px ({best_factor}x upscale)")

    for bar, v in list(zip(bars_base, rates_base)) + list(zip(bars_best, rates_best)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_title(f"{split.capitalize()} split")
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS, rotation=30, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Detection rate (%)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "compare_upscaling_best.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved compare_upscaling_best.png → {OUT_DIR}/")