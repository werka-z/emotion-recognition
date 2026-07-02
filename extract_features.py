"""
extract_features.py

Facial emotion recognition geometric feature extraction pipeline.

Extracts 28 normalised geometric features from 68 dlib facial landmarks
for successfully detected images in the FER-2013 dataset, then selects the top 26 by
one-way analysis of variance (ANOVA) F-statistic computed on the training set.

Best preprocessing configuration (from ablation study):
    upscale=True, clahe=False
    - Upscaling to 144x144 improves detection rate by ~9pp over raw 48x48
    - No CLAHE: degrades landmark quality despite improving contrast

Environment: fer_feature_extraction (see environment.yml)

Usage:
    python extract_features.py                        # best config: upscale, no CLAHE, top-26
    python extract_features.py --all-features         # upscale, no CLAHE, all 28 features
    python extract_features.py --clahe                # adds CLAHE on top of upscaling
    python extract_features.py --no-upscale           # skips upscaling (not recommended)
    python extract_features.py --no-upscale --clahe   # raw 48x48 with CLAHE

Outputs saved to features/:
    X_train.npy                unscaled train features       (n_train, 26)
    y_train.npy                train labels                  (n_train,)
    X_test.npy                 unscaled test features        (n_test, 26)
    y_test.npy                 test labels                   (n_test,)
    image_paths_train.npy      file paths of detected train images  (n_train,)
    image_paths_test.npy       file paths of detected test images   (n_test,)
    feature_names.npy          selected feature names        (26,)
    feature_names_full.npy     all 28 feature names          (28,)
    f_stats.npy                F-statistic per feature       (28,)
    top26_idx.npy              indices of top-26             (26,)
    scaler_mean.npy            train mean for inference      (26,)
    scaler_std.npy             train std  for inference      (26,)
    feature_distributions.npy  per-emotion raw feature arrays
    figB_feature_heatmap.png   median feature heatmap across emotions
    figC_feature_ranking.png   F-statistic bar chart
"""

import os
import sys
import bz2
import argparse
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils
from scipy.spatial import distance as dist
from scipy.stats import f_oneway

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FER_ROOT, PREDICTOR_PATH, EMOTION_LABELS, features_dir

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR     = features_dir()
TARGET_SIZE = 144
TOP_N       = 26

# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="FER-2013 geometric feature extraction")
parser.add_argument("--no-upscale", action="store_true",
                    help="Skip upscaling to 144x144 (not recommended — reduces detection rate by ~9pp)")
parser.add_argument("--clahe", action="store_true",
                    help="Apply CLAHE contrast enhancement before detection (ablation showed marginal effect)")
parser.add_argument("--all-features", action="store_true",
                    help="Keep all 28 features instead of selecting top 26 by ANOVA F-statistic")
args = parser.parse_args()

USE_UPSCALE   = not args.no_upscale
USE_CLAHE     = args.clahe
USE_ALL_FEATS = args.all_features

print(f"Preprocessing config:  upscale={USE_UPSCALE}  clahe={USE_CLAHE}  all_features={USE_ALL_FEATS}")
if not USE_UPSCALE:
    print("  [WARNING] Running without upscaling — expect lower detection rate")
if USE_ALL_FEATS:
    print("  [INFO] Keeping all 28 features — skipping feature selection")

# ── DECOMPRESS PREDICTOR IF NEEDED ────────────────────────────────────────────
bz2_path = PREDICTOR_PATH + ".bz2"
if not os.path.exists(PREDICTOR_PATH) and os.path.exists(bz2_path):
    print(f"Decompressing {bz2_path} ...")
    with bz2.BZ2File(bz2_path) as fr, open(PREDICTOR_PATH, "wb") as fw:
        fw.write(fr.read())
    print("Done.")

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
clahe_op  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
print("dlib loaded successfully.")

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def preprocess_image(gray_48: np.ndarray) -> np.ndarray:
    """
    Optionally upscale 48x48 -> 144x144 (cubic) and apply CLAHE.
    Controlled by USE_UPSCALE and USE_CLAHE flags.
    """
    img = gray_48.copy()
    if USE_UPSCALE:
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE),
                         interpolation=cv2.INTER_CUBIC)
    if USE_CLAHE:
        img = clahe_op.apply(img)
    return img


# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────
def extract_features(shape: np.ndarray) -> dict | None:
    """
    Compute 28 geometric features from 68 dlib landmarks.
    All distances are normalised by inter-ocular distance (IOD)
    to make features scale-invariant across face sizes.
    Returns None if its a failed landmark fit.
    """
    iod = dist.euclidean(shape[36], shape[45])
    if iod < 1e-3:
        return None

    features = {}

    # Eye features
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    ear_right = eye_aspect_ratio(shape[36:42])
    ear_left  = eye_aspect_ratio(shape[42:48])

    features["ear_avg"]          = (ear_right + ear_left) / 2
    features["ear_asymmetry"]    = abs(ear_right - ear_left)
    features["cheek_raise"]      = (dist.euclidean(shape[37], shape[41]) /
                                    (dist.euclidean(shape[36], shape[39]) + 1e-6))
    features["eye_open_left"]    = dist.euclidean(shape[37], shape[41]) / iod
    features["eye_open_right"]   = dist.euclidean(shape[43], shape[47]) / iod
    features["eye_open_diff"]    = abs(features["eye_open_left"] -
                                       features["eye_open_right"])
    features["eye_corner_slant"] = ((shape[45,1] - shape[42,1]) +
                                    (shape[39,1] - shape[36,1])) / (2 * iod)

    # Brow features
    features["brow_furrow"]      = dist.euclidean(shape[21], shape[22])

    r_brow_y = np.mean([shape[i][1] for i in range(17, 22)])
    l_brow_y = np.mean([shape[i][1] for i in range(22, 27)])
    features["brow_raise_avg"]   = ((shape[36][1] - r_brow_y) +
                                    (shape[45][1] - l_brow_y)) / 2

    features["brow_slope_right"] = ((shape[21][1] - shape[17][1]) /
                                    (shape[21][0] - shape[17][0] + 1e-6))
    features["brow_slope_left"]  = ((shape[22][1] - shape[26][1]) /
                                    (shape[26][0] - shape[22][0] + 1e-6))

    brow_ys = [shape[i][1] for i in range(17, 27)]
    features["brow_compression"] = max(brow_ys) - min(brow_ys)

    ref_y   = shape[27, 1]
    inner_y = (shape[19,1] + shape[20,1] + shape[23,1] + shape[24,1]) / 4
    outer_y = (shape[17,1] + shape[18,1] + shape[25,1] + shape[26,1]) / 4
    features["inner_brow_raise"] = (ref_y - inner_y) / iod
    features["outer_brow_raise"] = (ref_y - outer_y) / iod
    features["brow_oblique"]     = (inner_y - outer_y) / iod
    features["inner_brow_dist"]  = dist.euclidean(shape[21], shape[22]) / iod

    # Mouth features
    def mouth_aspect_ratio(mouth):
        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[4], mouth[8])
        C = dist.euclidean(mouth[0], mouth[6])
        return (A + B) / (2.0 * C)

    features["mar"] = mouth_aspect_ratio(shape[48:68])

    mouth_top_y = np.mean([shape[50][1], shape[51][1], shape[52][1]])
    features["lip_corner_pull_avg"] = ((mouth_top_y - shape[48][1]) +
                                       (mouth_top_y - shape[54][1])) / 2

    dY = shape[54][1] - shape[48][1]
    dX = shape[54][0] - shape[48][0] + 1e-6
    features["lip_corner_angle"]    = np.degrees(np.arctan2(-dY, dX))
    features["mouth_asymmetry"]     = abs(shape[48][1] - shape[54][1])
    features["lip_corner_depress"]  = ((shape[48,1] + shape[54,1]) / 2 -
                                        shape[51,1]) / iod
    features["upper_lip_raise"]     = (shape[33,1] - shape[51,1]) / iod
    features["mouth_width"]         = dist.euclidean(shape[48], shape[54]) / iod
    features["lip_gap"]             = dist.euclidean(shape[51], shape[57]) / iod

    # Nose / jaw features
    features["nose_to_mouth"] = dist.euclidean(shape[33], shape[51])
    features["jaw_drop"]      = dist.euclidean(shape[8],  shape[27])
    features["chin_to_mouth"] = (shape[8,1] - shape[57,1]) / iod

    face_width  = dist.euclidean(shape[0],  shape[16]) / iod
    face_height = dist.euclidean(shape[8],  shape[27]) / iod
    features["face_ratio"] = face_height / (face_width + 1e-5)

    # Global IOD normalisation
    return {k: v / iod for k, v in features.items()}


FEATURE_NAMES = None

def detect_and_extract(gray_48: np.ndarray) -> np.ndarray | None:
    """
    Full single-image pipeline:
        preprocess -> detect -> extract features
    Returns a 1-D float32 array or None if no face is detected.
    """
    global FEATURE_NAMES

    img   = preprocess_image(gray_48)
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


# ── DATASET PROCESSING ────────────────────────────────────────────────────────
def process_split(split_dir: str):
    """
    Iterate over split_dir/<Emotion>/*.jpg and extract features for every image.
    Saves the file path for each successfully detected image alongside features,
    so misclassified samples can be traced back to their original images.
    Returns X (n_samples, 28), y (n_samples,), paths (n_samples,).
    """
    X_rows, y_rows, path_rows = [], [], []
    drop_stats = {e: {"total": 0, "dropped": 0} for e in EMOTION_LABELS}

    for label_idx, emotion in enumerate(EMOTION_LABELS):
        class_dir = os.path.join(split_dir, emotion)
        if not os.path.isdir(class_dir):
            print(f"  [WARNING] {class_dir} not found — skipping.")
            continue

        img_paths = [os.path.join(class_dir, fn)
                     for fn in sorted(os.listdir(class_dir))
                     if fn.lower().endswith((".jpg", ".jpeg", ".png"))]

        for path in img_paths:
            drop_stats[emotion]["total"] += 1
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                drop_stats[emotion]["dropped"] += 1
                continue
            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48))

            feats = detect_and_extract(img)
            if feats is None:
                drop_stats[emotion]["dropped"] += 1
                continue

            X_rows.append(feats)
            y_rows.append(label_idx)
            path_rows.append(path)   # ← save path of successfully detected image

    print("\nEmotion       Total    Dropped    Drop%")
    print("-" * 40)
    total_all = dropped_all = 0
    for emotion in EMOTION_LABELS:
        s   = drop_stats[emotion]
        pct = 100 * s["dropped"] / s["total"] if s["total"] > 0 else 0
        print(f"{emotion}:  {s['total']} total,  {s['dropped']} dropped  ({pct:.1f}%)")
        total_all   += s["total"]
        dropped_all += s["dropped"]
    print("-" * 40)
    pct_all = 100 * dropped_all / total_all
    print(f"TOTAL:  {total_all} total,  {dropped_all} dropped  ({pct_all:.1f}%)")

    return (np.array(X_rows, dtype=np.float32),
            np.array(y_rows, dtype=np.int32),
            np.array(path_rows))


# ── FEATURE SELECTION ─────────────────────────────────────────────────────────
def select_top_features(X_train: np.ndarray, y_train: np.ndarray, n: int = TOP_N):
    """
    Rank all 28 features by one-way ANOVA F-statistic across emotion classes.
    Computed on training set only to prevent data leakage.
    Returns f_stats (28,) and top_idx (n,).
    """
    f_stats = []
    for fi in range(X_train.shape[1]):
        groups  = [X_train[y_train == li, fi]
                   for li in range(7) if (y_train == li).sum() > 1]
        stat, _ = f_oneway(*groups)
        f_stats.append(float(stat) if np.isfinite(stat) else 0.0)

    f_stats    = np.array(f_stats)
    ranked_idx = np.argsort(f_stats)[::-1]
    top_idx    = ranked_idx[:n]

    print(f"\nTop-{n} features  (F cutoff: {f_stats[top_idx[-1]]:.1f})\n")
    print(f"Rank   Feature                    F-stat")
    print("-" * 40)
    for rank, fi in enumerate(top_idx, 1):
        print(f"{rank}.  {FEATURE_NAMES[fi]}:  {f_stats[fi]:.1f}")

    print(f"\nDropped  (F < {f_stats[top_idx[-1]]:.1f}):")
    for fi in ranked_idx[n:]:
        print(f"  {FEATURE_NAMES[fi]}:  F = {f_stats[fi]:.1f}")

    return f_stats, top_idx


# ── FEATURE LABELS ────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "ear_avg": "EAR avg", "ear_asymmetry": "EAR asymmetry",
    "cheek_raise": "Cheek raise", "eye_open_left": "Eye open (L)",
    "eye_open_right": "Eye open (R)", "eye_open_diff": "Eye open diff",
    "eye_corner_slant": "Eye corner slant", "brow_furrow": "Brow furrow",
    "brow_raise_avg": "Brow raise avg", "brow_slope_right": "Brow slope (R)",
    "brow_slope_left": "Brow slope (L)", "brow_compression": "Brow compression",
    "inner_brow_raise": "Inner brow raise", "outer_brow_raise": "Outer brow raise",
    "brow_oblique": "Brow oblique", "inner_brow_dist": "Inner brow dist",
    "mar": "MAR", "lip_corner_pull_avg": "Lip corner pull",
    "lip_corner_angle": "Lip corner angle", "mouth_asymmetry": "Mouth asymmetry",
    "lip_corner_depress": "Lip corner depress", "upper_lip_raise": "Upper lip raise",
    "mouth_width": "Mouth width", "lip_gap": "Lip gap",
    "nose_to_mouth": "Nose-to-mouth", "jaw_drop": "Jaw drop",
    "chin_to_mouth": "Chin-to-mouth", "face_ratio": "Face ratio",
}
def flabel(n): return FEATURE_LABELS.get(n, n)


# ── PLOTS ─────────────────────────────────────────────────────────────────────
def plot_feature_ranking(f_stats: np.ndarray, top_idx: np.ndarray):
    ranked_idx  = np.argsort(f_stats)[::-1]
    F_THRESHOLD = f_stats[top_idx[-1]]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([flabel(FEATURE_NAMES[i]) for i in ranked_idx],
            [f_stats[i]               for i in ranked_idx])
    ax.axvline(F_THRESHOLD, ls="--",
               label=f"Top-{TOP_N} cutoff  (F = {F_THRESHOLD:.1f})")
    ax.invert_yaxis()
    ax.set_xlabel("F-statistic")
    ax.set_title("Feature discriminability ranking\n"
                 "(one-way ANOVA F-statistic across 7 emotions)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figC_feature_ranking.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figC_feature_ranking.png → {OUT_DIR}/")


def plot_feature_heatmap(emotion_arrays: dict,
                          f_stats: np.ndarray,
                          feature_names: list):
    n_features = len(feature_names)
    ranked_idx = np.argsort(f_stats)[::-1]

    all_data  = np.vstack(list(emotion_arrays.values()))
    feat_mean = all_data.mean(axis=0)
    feat_std  = all_data.std(axis=0) + 1e-8
    scaled    = {e: (arr - feat_mean) / feat_std for e, arr in emotion_arrays.items()}

    median_matrix = np.array([
        [np.median(scaled[e][:, fi]) for e in EMOTION_LABELS]
        for fi in ranked_idx])

    fig, ax = plt.subplots(figsize=(10, 14))
    fig.suptitle("Median Feature Value per Emotion\n"
                 "(features ranked by F-statistic, z-scored across emotions)",
                 fontsize=13, fontweight="bold")
    im = ax.imshow(median_matrix, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="z-score of median")
    ax.set_xticks(range(len(EMOTION_LABELS)))
    ax.set_xticklabels(EMOTION_LABELS, fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(
        [f"{flabel(feature_names[i])}  (F={f_stats[i]:.0f})" for i in ranked_idx],
        fontsize=8)
    ax.set_xlabel("Emotion", fontsize=11)
    ax.set_ylabel("Feature  (F-statistic)", fontsize=11)
    for r in range(n_features):
        for c in range(len(EMOTION_LABELS)):
            v = median_matrix[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 1.2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figB_feature_heatmap.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved figB_feature_heatmap.png → {OUT_DIR}/")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Processing TRAIN split ===")
    X_train, y_train, paths_train = process_split(os.path.join(FER_ROOT, "train"))

    print("\n=== Processing TEST split ===")
    X_test, y_test, paths_test = process_split(os.path.join(FER_ROOT, "test"))

    print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}  |  "
          f"Features: {X_train.shape[1]}")

    # Feature selection (train set only — no leakage)
    f_stats, top_idx = select_top_features(X_train, y_train, n=TOP_N)

    if USE_ALL_FEATS:
        X_train_sel    = X_train
        X_test_sel     = X_test
        names_selected = FEATURE_NAMES
        print(f"\nKeeping all {len(FEATURE_NAMES)} features.")
    else:
        X_train_sel    = X_train[:, top_idx]
        X_test_sel     = X_test[:,  top_idx]
        names_selected = [FEATURE_NAMES[i] for i in top_idx]

    scaler_mean = X_train_sel.mean(axis=0)
    scaler_std  = X_train_sel.std(axis=0) + 1e-8

    np.save(os.path.join(OUT_DIR, "X_train.npy"),            X_train_sel)
    np.save(os.path.join(OUT_DIR, "y_train.npy"),            y_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"),             X_test_sel)
    np.save(os.path.join(OUT_DIR, "y_test.npy"),             y_test)
    np.save(os.path.join(OUT_DIR, "image_paths_train.npy"),  paths_train,  allow_pickle=True)
    np.save(os.path.join(OUT_DIR, "image_paths_test.npy"),   paths_test,   allow_pickle=True)
    np.save(os.path.join(OUT_DIR, "feature_names.npy"),      np.array(names_selected), allow_pickle=True)
    np.save(os.path.join(OUT_DIR, "feature_names_full.npy"), np.array(FEATURE_NAMES),  allow_pickle=True)
    np.save(os.path.join(OUT_DIR, "f_stats.npy"),            f_stats)
    np.save(os.path.join(OUT_DIR, "top26_idx.npy"),          top_idx)
    np.save(os.path.join(OUT_DIR, "scaler_mean.npy"),        scaler_mean)
    np.save(os.path.join(OUT_DIR, "scaler_std.npy"),         scaler_std)

    emotion_arrays = {}
    for label_idx, emotion in enumerate(EMOTION_LABELS):
        mask = y_train == label_idx
        if mask.sum() > 0:
            emotion_arrays[emotion] = X_train[mask]

    np.save(os.path.join(OUT_DIR, "feature_distributions.npy"),
            emotion_arrays, allow_pickle=True)

    print(f"\nSaved to '{OUT_DIR}/':")
    for fn in sorted(os.listdir(OUT_DIR)):
        kb = os.path.getsize(os.path.join(OUT_DIR, fn)) / 1024
        print(f"  {fn}  ({kb:.1f} KB)")

    plot_feature_ranking(f_stats, top_idx)
    plot_feature_heatmap(emotion_arrays, f_stats, FEATURE_NAMES)