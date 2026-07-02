"""
show_detection_failures.py

Shows example images that dlib fails to detect at 48px (no upscaling)
for each emotion class, alongside whether upscaling to 144px recovers them.

Three categories per emotion:
    - Failed at 48px, recovered at 144px  (upscaling helps)
    - Failed at both 48px and 144px       (hard failures)

Environment: dlib (see environment.yml)

Usage:
    python ablation_studies/show_detection_failures.py

Saves:
    ablation_results/upscaling/failures_recovered.png   -- upscaling helps
    ablation_results/upscaling/failures_hard.png        -- undetectable at any size
"""

import os
import sys
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import FER_ROOT, EMOTION_LABELS, ablation_results_subdir

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR     = ablation_results_subdir("upscaling")
TARGET_SIZE = 144
N_EXAMPLES  = 5   # examples per emotion class

detector = dlib.get_frontal_face_detector()


def try_detect(img: np.ndarray) -> bool:
    rects = detector(img, 0)
    if not rects:
        rects = detector(img, 1)
    return len(rects) > 0


def collect_failures(split: str = "train"):
    """
    For each emotion class collect up to N_EXAMPLES images that:
        - fail at 48px (no upscale)
        - succeed at TARGET_SIZE px (recovered)
    and up to N_EXAMPLES that:
        - fail at both 48px and TARGET_SIZE px (hard failures)
    """
    recovered  = {e: [] for e in EMOTION_LABELS}
    hard_fails = {e: [] for e in EMOTION_LABELS}

    for emotion in EMOTION_LABELS:
        class_dir = os.path.join(FER_ROOT, split, emotion)
        if not os.path.isdir(class_dir):
            continue
        for fn in sorted(os.listdir(class_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if (len(recovered[emotion]) >= N_EXAMPLES and
                    len(hard_fails[emotion]) >= N_EXAMPLES):
                break

            img = cv2.imread(os.path.join(class_dir, fn), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48))

            # test at 48px
            detected_48 = try_detect(img)
            if detected_48:
                continue   # dlib found it at 48px — not a failure case

            # test at TARGET_SIZE px
            img_target      = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE),
                                          interpolation=cv2.INTER_CUBIC)
            detected_target = try_detect(img_target)

            if detected_target and len(recovered[emotion]) < N_EXAMPLES:
                recovered[emotion].append(img)
            elif not detected_target and len(hard_fails[emotion]) < N_EXAMPLES:
                hard_fails[emotion].append(img)

    return recovered, hard_fails


def plot_grid(examples: dict, title: str, subtitle: str, out_path: str):
    """
    Plot a grid of examples: rows = emotions, cols = N_EXAMPLES images.
    """
    n_rows = len(EMOTION_LABELS)
    n_cols = N_EXAMPLES
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 1.8, n_rows * 1.8))
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")

    for ri, emotion in enumerate(EMOTION_LABELS):
        imgs = examples[emotion]
        for ci in range(n_cols):
            ax = axes[ri, ci]
            if ci < len(imgs):
                ax.imshow(imgs[ci], cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(np.ones((48, 48), dtype=np.uint8) * 200,
                          cmap="gray", vmin=0, vmax=255)
                ax.text(24, 24, "N/A", ha="center", va="center",
                        fontsize=8, color="grey")
            ax.axis("off")
            if ci == 0:
                ax.set_ylabel(emotion, fontsize=9, fontweight="bold",
                              rotation=0, labelpad=40, va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {os.path.basename(out_path)} → {OUT_DIR}/")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Collecting failure examples from train split ...")
    print("(this may take a few minutes)\n")

    recovered, hard_fails = collect_failures(split="train")

    for emotion in EMOTION_LABELS:
        print(f"  {emotion:<12}  recovered: {len(recovered[emotion])}  "
              f"hard fails: {len(hard_fails[emotion])}")

    plot_grid(
        recovered,
        title="Failures recovered by upscaling",
        subtitle=f"Failed at 48px — detected at {TARGET_SIZE}px",
        out_path=os.path.join(OUT_DIR, "failures_recovered.png")
    )

    plot_grid(
        hard_fails,
        title="Hard detection failures",
        subtitle=f"Failed at both 48px and {TARGET_SIZE}px",
        out_path=os.path.join(OUT_DIR, "failures_hard.png")
    )