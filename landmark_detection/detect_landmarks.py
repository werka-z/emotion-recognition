"""Test facial-landmark detection coverage on FER-2013 with dlib and mediapipe.

Samples N images (stratified by class) from data/train and reports, for each
detector/setting, how many images had a face (and landmarks) found. Failure
file paths are written out so they can be inspected manually.

Usage:
    python detect_landmarks.py --n 1000 --upscale 4
"""
import argparse
import glob
import os
import random
import time

import cv2
import dlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT_DIR, "data", "train")
SHAPE_PREDICTOR_PATH = os.path.join(HERE, "models", "shape_predictor_68_face_landmarks.dat")
FACE_LANDMARKER_PATH = os.path.join(HERE, "models", "face_landmarker.task")
OUT_DIR = os.path.join(HERE, "results")


def rel(path):
    """Path relative to the emotion-recognition project root, for portable logs."""
    return os.path.relpath(path, ROOT_DIR)

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def sample_files(n, seed=42):
    """Stratified sample of n image paths across CLASSES, proportional to class size."""
    rng = random.Random(seed)
    all_by_class = {c: sorted(glob.glob(os.path.join(DATA_DIR, c, "*"))) for c in CLASSES}
    total = sum(len(v) for v in all_by_class.values())

    sampled = []
    for c in CLASSES:
        files = all_by_class[c]
        k = round(n * len(files) / total)
        k = min(k, len(files))
        sampled.extend(rng.sample(files, k))
    rng.shuffle(sampled)
    return sampled[:n]


def run_dlib(files, upscale=1):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    failures = []
    t0 = time.time()
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if upscale > 1:
            img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        rects = detector(img, 1)
        if len(rects) == 0:
            failures.append(f)
        # Note: if a face IS found, dlib's shape_predictor essentially always
        # returns 68 points for that box (it doesn't "fail" separately) so
        # face-detection success is the real bottleneck for landmarks.
    elapsed = time.time() - t0
    return {
        "name": f"dlib_upscale{upscale}x",
        "display_name": f"dlib (upscale={upscale}x)",
        "n": len(files),
        "n_failed": len(failures),
        "failures": failures,
        "elapsed_s": elapsed,
    }


def run_mediapipe(files):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)

    failures = []
    t0 = time.time()
    for f in files:
        img = cv2.imread(f)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)
        if len(result.face_landmarks) == 0:
            failures.append(f)
    elapsed = time.time() - t0
    return {
        "name": "mediapipe_native",
        "display_name": "mediapipe (native res)",
        "n": len(files),
        "n_failed": len(failures),
        "failures": failures,
        "elapsed_s": elapsed,
    }


def report(result):
    n, n_failed = result["n"], result["n_failed"]
    n_ok = n - n_failed
    rate = 100 * n_ok / n
    print(f"\n=== {result['display_name']} ===")
    print(f"  detected: {n_ok}/{n} ({rate:.1f}%)")
    print(f"  failed:   {n_failed}/{n} ({100 - rate:.1f}%)")
    print(f"  time:     {result['elapsed_s']:.2f}s ({result['elapsed_s']/n*1000:.2f} ms/img)")


def write_failures(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"failures_{result['name']}.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(rel(f) for f in result["failures"]))
    print(f"  failures written to {rel(path)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of images to sample")
    ap.add_argument("--upscale", type=int, default=4, help="upscale factor for the second dlib pass")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sample_files(args.n, seed=args.seed)
    print(f"Sampled {len(files)} images from {DATA_DIR}")

    results = []
    results.append(run_dlib(files, upscale=1))
    results.append(run_dlib(files, upscale=args.upscale))
    results.append(run_mediapipe(files))

    for r in results:
        report(r)
        write_failures(r, OUT_DIR)

    # Images that failed under every method -> the real "last resort, look manually" set.
    common_failures = set(results[0]["failures"])
    for r in results[1:]:
        common_failures &= set(r["failures"])
    print(f"\n=== Failed under ALL methods (dlib@1x, dlib@{args.upscale}x, mediapipe) ===")
    print(f"  {len(common_failures)}/{len(files)} images")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "failures_ALL_METHODS.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(sorted(rel(f) for f in common_failures)))
    print(f"  written to {rel(out_path)}")


if __name__ == "__main__":
    main()
