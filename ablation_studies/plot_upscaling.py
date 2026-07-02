"""
plot_upscaling.py

Plots detection rate results saved by compare_upscaling.py:
  1. Grouped bar chart comparing all upscale factors, per emotion.
  2. Bar chart comparing no upscaling (48px) vs the best upscale factor
     found automatically (highest overall detection rate on train split).

Environment: dlib (see environment.yml) — only needs numpy and matplotlib

Usage:
    python ablation_studies/plot_upscaling.py

Reads:
    ablation_results/upscaling/upscaling_results.npy

Saves:
    ablation_results/upscaling/compare_upscaling.png
    ablation_results/upscaling/compare_upscaling_best.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import EMOTION_LABELS, ablation_results_subdir

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR         = ablation_results_subdir("upscaling")
UPSCALE_FACTORS = [1, 2, 3, 4, 5, 6]

# ── LOAD ──────────────────────────────────────────────────────────────────────
results = np.load(os.path.join(OUT_DIR, "upscaling_results.npy"),
                   allow_pickle=True).item()


# ── PLOT 1: all upscale factors, grouped bar chart ────────────────────────────
def plot_all_factors():
    labels     = [f"{f}x\n({48*f}px)" for f in UPSCALE_FACTORS]
    n_factors  = len(UPSCALE_FACTORS)
    n_emotions = len(EMOTION_LABELS)
    x          = np.arange(n_emotions)
    bar_width  = 0.12

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


# ── PLOT 2: no upscaling vs best factor ───────────────────────────────────────
def plot_best_factor():
    best_factor = max(
        UPSCALE_FACTORS,
        key=lambda f: sum(results[f]["train"][e]["detected"]
                           for e in EMOTION_LABELS)
    )
    best_size = 48 * best_factor
    print(f"Best upscale factor: {best_factor}x ({best_size}x{best_size} px)")

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
                            label="48px (no upscale)")
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


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_all_factors()
    plot_best_factor()