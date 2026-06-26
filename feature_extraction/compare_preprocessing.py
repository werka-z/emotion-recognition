"""
compare_preprocessing.py

Ablation study: effect of upscaling and CLAHE on dlib face detection rate.

FER-2013 images are 48x48 pixels which are too small for dlib's frontal face detector
which was trained on natural-sized images. This script tests whether upscaling
to 224x224 before detection recovers more faces, and whether adding contrast enhancement (CLAHE)
on top of that helps or hurts.

The 4 conditions tested:
  no_up_no_cl   raw 48x48, no contrast enhancement
  no_up_cl      raw 48x48, CLAHE applied
  up_no_cl      upscaled to 224x224                    (best condition)
  up_cl         upscaled to 224x224, CLAHE applied

Environment: dlib (see environment.yml)

Usage:
    python compare_preprocessing.py

Output:
    Per-emotion face detection rate table for each condition (train and test split).
    Saves detection_counts.npy to processed_final/ for use in plotting.
"""

import os
import cv2
import dlib
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FER_ROOT       = os.path.join(BASE_DIR, "data")
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
OUT_DIR        = os.path.join(BASE_DIR, "processed_final")
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

CONDITIONS = [
    ("no_up_no_cl", False, False),
    ("no_up_cl",    False, True),
    ("up_no_cl",    True,  False),   # best results
    ("up_cl",       True,  True),
]

os.makedirs(OUT_DIR, exist_ok=True)
detector = dlib.get_frontal_face_detector()
clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def count_detections(split_dir: str, use_upscale: bool, use_clahe: bool) -> dict:
    """
    Walk split_dir/<Emotion>/*.jpg and count how many images
    dlib successfully detects a face in.
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
            if use_upscale:
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            if use_clahe:
                img = clahe_op.apply(img)

            stats[emotion]["total"] += 1
            rects = detector(img, 0)
            if not rects:
                rects = detector(img, 1)
            if rects:
                stats[emotion]["detected"] += 1

    return stats


def print_comparison_table(all_counts: dict, split: str):
    """
    Prints the detection rates per emotion for each condition for a given split (train or test).
    """
    print(f"\n{split.upper()} split — detection rate (%) per condition")
    print("-" * 50)
    print(f"  {'Emotion':<12}", end="")
    for prefix, _, _ in CONDITIONS:
        print(f"  {prefix:>12}", end="")
    print()
    print("-" * 50)
    for emotion in EMOTION_LABELS:
        print(f"  {emotion:<12}", end="")
        for prefix, _, _ in CONDITIONS:
            s   = all_counts[prefix][split][emotion]
            pct = 100 * s["detected"] / s["total"] if s["total"] > 0 else 0
            print(f"  {pct:>11.1f}%", end="")
        print()
    print("-" * 50)
    print(f"  {'TOTAL':<12}", end="")
    for prefix, _, _ in CONDITIONS:
        s   = all_counts[prefix][split]
        tot = sum(v["total"]    for v in s.values())
        det = sum(v["detected"] for v in s.values())
        print(f"  {100 * det / tot:>11.1f}%", end="")
    print()


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_counts = {}

    for prefix, use_upscale, use_clahe in CONDITIONS:
        label = f"upscale={use_upscale}  clahe={use_clahe}"
        print(f"\n{prefix}  ({label})")
        all_counts[prefix] = {}

        for split in ("train", "test"):
            print(f"  {split} ...", flush=True)
            stats    = count_detections(os.path.join(FER_ROOT, split),
                                        use_upscale, use_clahe)
            all_counts[prefix][split] = stats
            total    = sum(s["total"]    for s in stats.values())
            detected = sum(s["detected"] for s in stats.values())
            print(f"  → detected {detected}/{total}  ({100 * detected / total:.1f}%)")

    for split in ("train", "test"):
        print_comparison_table(all_counts, split)

    out_path = os.path.join(OUT_DIR, "detection_counts.npy")
    np.save(out_path, all_counts, allow_pickle=True)
    print(f"\nSaved detection_counts.npy → {OUT_DIR}/")