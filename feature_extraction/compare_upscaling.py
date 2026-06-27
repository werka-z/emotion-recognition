"""
compare_upscaling.py

Ablation study: effect of upscaling factor on dlib face detection rate.

FER-2013 images are 48x48 pixels. This script tests how detection rate
changes as we upscale to larger resolutions before running dlib, to
justify the choice of target size used in extract_features.py.

Upscale factors tested (resulting image sizes):
    1x  ->  48x48   (no upscaling, baseline)
    2x  ->  96x96
    3x  ->  144x144
    4x  ->  192x192
    5x  ->  240x240
    6x  ->  288x288  (diminishing returns expected here)

Environment: dlib (see environment.yml)

Usage:
    python compare_upscaling.py

Output:
    Detection rate table printed per upscale factor.
    Saves compare_upscaling.png to processed_final/.
"""

import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FER_ROOT       = os.path.join(BASE_DIR, "data")
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
OUT_DIR        = os.path.join(BASE_DIR, "processed_final")
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

UPSCALE_FACTORS = [1, 2, 3, 4, 5, 6]   # resulting sizes: 48, 96, 144, 192, 240, 288

os.makedirs(OUT_DIR, exist_ok=True)
detector = dlib.get_frontal_face_detector()


def count_detections(split_dir: str, upscale_factor: int) -> dict:
    """
    Walk split_dir/<Emotion>/*.jpg and count how many images
    dlib successfully detects a face in at the given upscale factor.
    Returns {emotion: {total, detected}}.
    """
    stats = {e: {"total": 0, "detected": 0} for e in EMOTION_LABELS}

    for emotion in EMOTION_LABELS:
        class_dir = os.path.join(split_dir, emotion)
        if not os.path.isdir(class_dir):
            continue
        for fn in sorted(os.listdir(class_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(class_dir, fn), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48))

            if upscale_factor > 1:
                target = 48 * upscale_factor
                img = cv2.resize(img, (target, target),
                                 interpolation=cv2.INTER_CUBIC)

            stats[emotion]["total"] += 1
            rects = detector(img, 0)
            if not rects:
                rects = detector(img, 1)
            if rects:
                stats[emotion]["detected"] += 1

    return stats


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}   # upscale_factor -> split -> stats

    for factor in UPSCALE_FACTORS:
        size = 48 * factor
        print(f"\nUpscale {factor}x  ({size}x{size} px)")
        results[factor] = {}
        for split in ("train", "test"):
            print(f"  {split} ...", flush=True)
            stats    = count_detections(os.path.join(FER_ROOT, split), factor)
            results[factor][split] = stats
            total    = sum(s["total"]    for s in stats.values())
            detected = sum(s["detected"] for s in stats.values())
            print(f"  → detected {detected}/{total}  ({100 * detected / total:.1f}%)")

    # ── Print summary table ───────────────────────────────────────────────────
    for split in ("train", "test"):
        print(f"\n{split.upper()} split — detection rate (%) by upscale factor")
        print("-" * 55)
        header = "Emotion       " + "".join(f"  {f}x ({48*f}px)" for f in UPSCALE_FACTORS)
        print(header)
        print("-" * len(header))
        for emotion in EMOTION_LABELS:
            row = f"{emotion:<12}  "
            for factor in UPSCALE_FACTORS:
                s   = results[factor][split][emotion]
                pct = 100 * s["detected"] / s["total"] if s["total"] > 0 else 0
                row += f"  {pct:>9.1f}%"
            print(row)
        print("-" * len(header))
        total_row = "TOTAL         "
        for factor in UPSCALE_FACTORS:
            s   = results[factor][split]
            tot = sum(v["total"]    for v in s.values())
            det = sum(v["detected"] for v in s.values())
            total_row += f"  {100 * det / tot:>9.1f}%"
        print(total_row)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Detection rate vs upscale factor\n(dlib frontal face detector)")

    sizes = [48 * f for f in UPSCALE_FACTORS]

    for ax, split in zip(axes, ("train", "test")):
        for emotion in EMOTION_LABELS:
            rates = []
            for factor in UPSCALE_FACTORS:
                s   = results[factor][split][emotion]
                pct = 100 * s["detected"] / s["total"] if s["total"] > 0 else 0
                rates.append(pct)
            ax.plot(sizes, rates, marker="o", label=emotion)

        # overall detection rate
        overall = []
        for factor in UPSCALE_FACTORS:
            s   = results[factor][split]
            tot = sum(v["total"]    for v in s.values())
            det = sum(v["detected"] for v in s.values())
            overall.append(100 * det / tot)
        ax.plot(sizes, overall, marker="o", linewidth=2.5,
                linestyle="--", label="Overall", color="black")

        ax.axvline(224, ls=":", label="224px (extract_features.py)")
        ax.set_title(f"{split.capitalize()} split")
        ax.set_xlabel("Image size (px)")
        ax.set_ylabel("Detection rate (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(sizes)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "compare_upscaling.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved compare_upscaling.png → {OUT_DIR}/")

    np.save(os.path.join(OUT_DIR, "upscaling_results.npy"),
            results, allow_pickle=True)
    print(f"Saved upscaling_results.npy → {OUT_DIR}/")