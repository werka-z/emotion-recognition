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
    python ablation_studies/compare_upscaling.py

Output:
    Detection rate table printed per upscale factor.
    Saves upscaling_results.npy to ablation_results/upscaling/.
    Run plot_upscaling.py to generate plots from this data.
"""

import os
import sys
import cv2
import dlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import FER_ROOT, PREDICTOR_PATH, EMOTION_LABELS, ablation_results_subdir

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR = ablation_results_subdir("upscaling")

UPSCALE_FACTORS = [1, 2, 3, 4, 5, 6]   # resulting sizes: 48, 96, 144, 192, 240, 288

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

    # ── Save results ──────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "upscaling_results.npy"),
            results, allow_pickle=True)
    print(f"Saved upscaling_results.npy → {OUT_DIR}/")
    print("Run plot_upscaling.py and plot_upscaling_best.py to generate plots.")