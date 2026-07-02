"""
extract_ablation.py  —  run in your DLIB environment
──────────────────────────────────────────────────────
Runs feature extraction for all 4 upscaling × CLAHE conditions:

  upscale=False  clahe=False  →  prefix: no_up_no_cl
  upscale=False  clahe=True   →  prefix: no_up_cl
  upscale=True   clahe=False  →  prefix: up_no_cl
  upscale=True   clahe=True   →  prefix: up_cl

Saves {prefix}_X_train/test.npy, {prefix}_y_train/test.npy,
and {prefix}_image_paths_train/test.npy
into ablation_results/features/ for use by downstream SVM scripts.

Image paths are saved alongside features so misclassified samples
can be traced back to their original images.

Usage:
    python ablation_studies/extract_ablation.py
"""

import os
import sys
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import FER_ROOT, PREDICTOR_PATH, EMOTION_LABELS, ablation_results_subdir

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR = ablation_results_subdir("features")

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
clahe_op  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

CONDITIONS = [
    # (prefix,        use_upscale, use_clahe)
    ("no_up_no_cl",  False,       False),
    ("no_up_cl",     False,       True),
    ("up_no_cl",     True,        False),
    ("up_cl",        True,        True),
]

# ── PIPELINE ──────────────────────────────────────────────────────────────────
def preprocess(gray_48: np.ndarray, use_upscale: bool, use_clahe: bool) -> np.ndarray:
    img = cv2.resize(gray_48, (144, 144), interpolation=cv2.INTER_CUBIC) \
          if use_upscale else gray_48.copy()
    return clahe_op.apply(img) if use_clahe else img

def extract_features(shape: np.ndarray):
    iod = dist.euclidean(shape[36], shape[45])
    if iod < 1e-3:
        return None
    f = {}

    def ear(eye):
        A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3]); return (A + B) / (2.0 * C)

    ear_r, ear_l          = ear(shape[36:42]), ear(shape[42:48])
    f["ear_avg"]          = (ear_r + ear_l) / 2
    f["ear_asymmetry"]    = abs(ear_r - ear_l)
    f["cheek_raise"]      = dist.euclidean(shape[37], shape[41]) / (dist.euclidean(shape[36], shape[39]) + 1e-6)
    f["eye_open_left"]    = dist.euclidean(shape[37], shape[41]) / iod
    f["eye_open_right"]   = dist.euclidean(shape[43], shape[47]) / iod
    f["eye_open_diff"]    = abs(f["eye_open_left"] - f["eye_open_right"])
    f["eye_corner_slant"] = ((shape[45,1]-shape[42,1]) + (shape[39,1]-shape[36,1])) / (2 * iod)
    f["brow_furrow"]      = dist.euclidean(shape[21], shape[22])
    r_brow_y = np.mean([shape[i][1] for i in range(17, 22)])
    l_brow_y = np.mean([shape[i][1] for i in range(22, 27)])
    f["brow_raise_avg"]   = ((shape[36][1]-r_brow_y) + (shape[45][1]-l_brow_y)) / 2
    f["brow_slope_right"] = (shape[21][1]-shape[17][1]) / (shape[21][0]-shape[17][0]+1e-6)
    f["brow_slope_left"]  = (shape[22][1]-shape[26][1]) / (shape[26][0]-shape[22][0]+1e-6)
    brow_ys               = [shape[i][1] for i in range(17, 27)]
    f["brow_compression"] = max(brow_ys) - min(brow_ys)
    ref_y   = shape[27, 1]
    inner_y = (shape[19,1]+shape[20,1]+shape[23,1]+shape[24,1]) / 4
    outer_y = (shape[17,1]+shape[18,1]+shape[25,1]+shape[26,1]) / 4
    f["inner_brow_raise"] = (ref_y - inner_y) / iod
    f["outer_brow_raise"] = (ref_y - outer_y) / iod
    f["brow_oblique"]     = (inner_y - outer_y) / iod
    f["inner_brow_dist"]  = dist.euclidean(shape[21], shape[22]) / iod

    def mar(m):
        A = dist.euclidean(m[2], m[10]); B = dist.euclidean(m[4], m[8])
        C = dist.euclidean(m[0], m[6]);  return (A + B) / (2.0 * C)
    f["mar"]                 = mar(shape[48:68])
    mt_y = np.mean([shape[50][1], shape[51][1], shape[52][1]])
    f["lip_corner_pull_avg"] = ((mt_y-shape[48][1]) + (mt_y-shape[54][1])) / 2
    f["lip_corner_angle"]    = np.degrees(np.arctan2(-(shape[54][1]-shape[48][1]), shape[54][0]-shape[48][0]+1e-6))
    f["mouth_asymmetry"]     = abs(shape[48][1] - shape[54][1])
    f["lip_corner_depress"]  = ((shape[48,1]+shape[54,1])/2 - shape[51,1]) / iod
    f["upper_lip_raise"]     = (shape[33,1] - shape[51,1]) / iod
    f["mouth_width"]         = dist.euclidean(shape[48], shape[54]) / iod
    f["lip_gap"]             = dist.euclidean(shape[51], shape[57]) / iod
    f["nose_to_mouth"]       = dist.euclidean(shape[33], shape[51])
    f["jaw_drop"]            = dist.euclidean(shape[8],  shape[27])
    f["chin_to_mouth"]       = (shape[8,1] - shape[57,1]) / iod
    fw = dist.euclidean(shape[0], shape[16]) / iod
    fh = dist.euclidean(shape[8], shape[27]) / iod
    f["face_ratio"]          = fh / (fw + 1e-5)
    return {k: v / iod for k, v in f.items()}


FEATURE_NAMES = None

def detect_and_extract(gray_48: np.ndarray, use_upscale: bool, use_clahe: bool):
    global FEATURE_NAMES
    img   = preprocess(gray_48, use_upscale, use_clahe)
    rects = detector(img, 0)
    if not rects:
        rects = detector(img, 1)
    if not rects:
        return None
    shape = face_utils.shape_to_np(predictor(img, rects[0]))
    feats = extract_features(shape)
    if feats is None:
        return None
    if FEATURE_NAMES is None:
        FEATURE_NAMES = list(feats.keys())
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)


def process_split(split_dir: str, use_upscale: bool, use_clahe: bool):
    X_rows, y_rows, path_rows = [], [], []
    stats = {e: {"total": 0, "dropped": 0} for e in EMOTION_LABELS}
    for label_idx, emotion in enumerate(EMOTION_LABELS):
        class_dir = os.path.join(split_dir, emotion)
        if not os.path.isdir(class_dir):
            continue
        img_paths = [os.path.join(class_dir, fn)
                     for fn in sorted(os.listdir(class_dir))
                     if fn.lower().endswith((".jpg", ".jpeg", ".png"))]
        for path in img_paths:
            stats[emotion]["total"] += 1
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                stats[emotion]["dropped"] += 1; continue
            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48))
            feats = detect_and_extract(img, use_upscale, use_clahe)
            if feats is None:
                stats[emotion]["dropped"] += 1; continue
            X_rows.append(feats)
            y_rows.append(label_idx)
            path_rows.append(path)   # ← save path of successfully detected image
    total   = sum(s["total"]   for s in stats.values())
    dropped = sum(s["dropped"] for s in stats.values())
    print(f"    kept {total-dropped}/{total}  (drop {100*dropped/total:.1f}%)")
    return (np.array(X_rows, dtype=np.float32),
            np.array(y_rows, dtype=np.int32),
            np.array(path_rows))


# ── RUN ALL CONDITIONS ────────────────────────────────────────────────────────
if __name__ == "__main__":
    for prefix, use_upscale, use_clahe in CONDITIONS:
        label = f"upscale={use_upscale}  clahe={use_clahe}"
        print(f"\n{'='*55}\n  {prefix}  ({label})\n{'='*55}")
        for split in ("train", "test"):
            print(f"  {split}:")
            X, y, paths = process_split(os.path.join(FER_ROOT, split),
                                        use_upscale, use_clahe)
            np.save(os.path.join(OUT_DIR, f"{prefix}_X_{split}.npy"), X)
            np.save(os.path.join(OUT_DIR, f"{prefix}_y_{split}.npy"), y)
            np.save(os.path.join(OUT_DIR, f"{prefix}_image_paths_{split}.npy"),
                    paths, allow_pickle=True)

    print(f"\nAll done. Files in '{OUT_DIR}/':")
    for fn in sorted(os.listdir(OUT_DIR)):
        kb = os.path.getsize(os.path.join(OUT_DIR, fn)) / 1024
        print(f"  {fn}  ({kb:.1f} KB)")