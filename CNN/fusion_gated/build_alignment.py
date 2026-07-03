"""Re-extract landmark features per image (dlib 20.0.1) and cache the alignment.

WHY RE-EXTRACT (instead of replaying feature_data/)
---------------------------------------------------
feature_data/ was produced with dlib 19.24.2, which does not build on this
machine (modern CMake dropped support for its bundled pybind11). dlib 20.0.1
produces slightly different landmarks AND a different success set, so the saved
rows cannot be mapped back to specific images. We therefore re-run the SAME
26-feature geometric pipeline with the installed dlib, directly in
torchvision.ImageFolder order, giving every image its own feature vector and an
exact per-image mask (1.0 = dlib detected a face, 0.0 = failed).

The feature DEFINITIONS are identical to feature_extraction/extract_features.py
(copied below — see note there). Only the dlib version differs. The scaler
(mean/std) is RE-FIT on the freshly extracted train set, because the saved
scaler_mean/std belong to the 19.24.2 value distribution; using it on 20.0.1
values would mis-normalise. The re-fit stats are cached alongside the features.

LABEL SPACE: we iterate ImageFolder order (alphabetical: angry,disgust,fear,
happy,neutral,sad,surprise) so labels are already in the CNN's label space.
No remapping needed.

OUTPUT (per split, CNN/fusion_gated/cache/align_<split>.npz):
  lm    (N,26) float32  z-score-normalised landmark vector (zeros where missing)
  mask  (N,)   float32  1.0 present / 0.0 missing
  label (N,)   int64    ImageFolder label
N == len(ImageFolder), aligned 1:1 with ImageFolder sample order.
Also writes cache/scaler.npz with mean,std,feature_names (the re-fit scaler).
"""

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

FEATURE_DATA = os.path.join(_REPO_ROOT, "feature_data")
DATA_DIR = os.path.join(_REPO_ROOT, "data")
CACHE_DIR = os.path.join(_HERE, "cache")
PREDICTOR_PATH = os.path.join(_REPO_ROOT, "landmark_detection", "models",
                              "shape_predictor_68_face_landmarks.dat")

# torchvision.ImageFolder order (alphabetical lowercase dir names).
IMAGEFOLDER_ORDER = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# The 26 selected feature names, in feature_data column order — we reproduce the
# same column order so the fusion model's lm_dim and feature semantics match the
# project's documented 26-feature set.
SELECTED_NAMES = [str(s) for s in np.load(
    os.path.join(FEATURE_DATA, "feature_names.npy"), allow_pickle=True)]

TARGET_SIZE = 224  # extract_features.py best config: upscale=True, clahe=False


class Extractor:
    """Self-contained copy of extract_features.py's pure pipeline.

    Identical feature math to feature_extraction/extract_features.py; only the
    dlib version differs. Any upstream change to the feature definitions must be
    mirrored here.
    """

    def __init__(self):
        import dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

    def preprocess(self, gray_48):
        import cv2
        return cv2.resize(gray_48, (TARGET_SIZE, TARGET_SIZE),
                          interpolation=cv2.INTER_CUBIC)

    def align_face(self, image, shape):
        import cv2
        lec = shape[36:42].mean(axis=0)
        rec = shape[42:48].mean(axis=0)
        angle = np.degrees(np.arctan2(rec[1] - lec[1], rec[0] - lec[0]))
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    @staticmethod
    def extract_features(shape):
        from scipy.spatial import distance as dist
        iod = dist.euclidean(shape[36], shape[45])
        if iod < 1e-3:
            return None
        features = {}

        def eye_aspect_ratio(eye):
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            C = dist.euclidean(eye[0], eye[3])
            return (A + B) / (2.0 * C)

        ear_right = eye_aspect_ratio(shape[36:42])
        ear_left = eye_aspect_ratio(shape[42:48])
        features["ear_avg"] = (ear_right + ear_left) / 2
        features["ear_asymmetry"] = abs(ear_right - ear_left)
        features["cheek_raise"] = (dist.euclidean(shape[37], shape[41]) /
                                   (dist.euclidean(shape[36], shape[39]) + 1e-6))
        features["eye_open_left"] = dist.euclidean(shape[37], shape[41]) / iod
        features["eye_open_right"] = dist.euclidean(shape[43], shape[47]) / iod
        features["eye_open_diff"] = abs(features["eye_open_left"] -
                                        features["eye_open_right"])
        features["eye_corner_slant"] = ((shape[45, 1] - shape[42, 1]) +
                                        (shape[39, 1] - shape[36, 1])) / (2 * iod)

        features["brow_furrow"] = dist.euclidean(shape[21], shape[22])
        r_brow_y = np.mean([shape[i][1] for i in range(17, 22)])
        l_brow_y = np.mean([shape[i][1] for i in range(22, 27)])
        features["brow_raise_avg"] = ((shape[36][1] - r_brow_y) +
                                      (shape[45][1] - l_brow_y)) / 2
        features["brow_slope_right"] = ((shape[21][1] - shape[17][1]) /
                                        (shape[21][0] - shape[17][0] + 1e-6))
        features["brow_slope_left"] = ((shape[22][1] - shape[26][1]) /
                                       (shape[26][0] - shape[22][0] + 1e-6))
        brow_ys = [shape[i][1] for i in range(17, 27)]
        features["brow_compression"] = max(brow_ys) - min(brow_ys)
        ref_y = shape[27, 1]
        inner_y = (shape[19, 1] + shape[20, 1] + shape[23, 1] + shape[24, 1]) / 4
        outer_y = (shape[17, 1] + shape[18, 1] + shape[25, 1] + shape[26, 1]) / 4
        features["inner_brow_raise"] = (ref_y - inner_y) / iod
        features["outer_brow_raise"] = (ref_y - outer_y) / iod
        features["brow_oblique"] = (inner_y - outer_y) / iod
        features["inner_brow_dist"] = dist.euclidean(shape[21], shape[22]) / iod

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
        features["lip_corner_angle"] = np.degrees(np.arctan2(-dY, dX))
        features["mouth_asymmetry"] = abs(shape[48][1] - shape[54][1])
        features["lip_corner_depress"] = ((shape[48, 1] + shape[54, 1]) / 2 -
                                          shape[51, 1]) / iod
        features["upper_lip_raise"] = (shape[33, 1] - shape[51, 1]) / iod
        features["mouth_width"] = dist.euclidean(shape[48], shape[54]) / iod
        features["lip_gap"] = dist.euclidean(shape[51], shape[57]) / iod

        features["nose_to_mouth"] = dist.euclidean(shape[33], shape[51])
        features["jaw_drop"] = dist.euclidean(shape[8], shape[27])
        features["chin_to_mouth"] = (shape[8, 1] - shape[57, 1]) / iod
        face_width = dist.euclidean(shape[0], shape[16]) / iod
        face_height = dist.euclidean(shape[8], shape[27]) / iod
        features["face_ratio"] = face_height / (face_width + 1e-5)

        return {k: v / iod for k, v in features.items()}

    def vec(self, gray48):
        """Full pipeline -> 26-vector (selected cols) or None on dlib failure."""
        import cv2
        from imutils import face_utils
        img = self.preprocess(gray48)
        rects = self.detector(img, 0)
        if not rects:
            rects = self.detector(img, 1)
        if not rects:
            return None
        shape = face_utils.shape_to_np(self.predictor(img, rects[0]))
        aligned = self.align_face(img, shape)
        rects_aligned = self.detector(aligned, 0)
        if rects_aligned:
            shape = face_utils.shape_to_np(self.predictor(aligned, rects_aligned[0]))
        feats = self.extract_features(shape)
        if feats is None:
            return None
        return np.array([feats[n] for n in SELECTED_NAMES], dtype=np.float32)


def extract_split(split, fe):
    """Extract raw (unscaled) features for every image in ImageFolder order.

    Returns raw (N,26) with NaN rows where dlib failed, mask (N,), label (N,).
    """
    import cv2
    split_dir = os.path.join(DATA_DIR, split)
    raw_rows, mask_rows, label_rows = [], [], []
    per_class = {}
    for cnn_label, emotion in enumerate(IMAGEFOLDER_ORDER):
        class_dir = os.path.join(split_dir, emotion)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(class_dir)
        fns = sorted(fn for fn in os.listdir(class_dir)
                     if fn.lower().endswith((".jpg", ".jpeg", ".png")))
        present = 0
        for fn in fns:
            img = cv2.imread(os.path.join(class_dir, fn), cv2.IMREAD_GRAYSCALE)
            vec = None
            if img is not None:
                if img.shape != (48, 48):
                    img = cv2.resize(img, (48, 48))
                vec = fe.vec(img)
            if vec is None:
                raw_rows.append(np.full(len(SELECTED_NAMES), np.nan, dtype=np.float32))
                mask_rows.append(0.0)
            else:
                raw_rows.append(vec)
                mask_rows.append(1.0)
                present += 1
            label_rows.append(cnn_label)
        per_class[emotion] = (present, len(fns))
        print(f"  [{split}/{emotion}] {present}/{len(fns)} detected "
              f"({100*present/len(fns):.1f}%)")
    raw = np.stack(raw_rows)
    mask = np.array(mask_rows, dtype=np.float32)
    label = np.array(label_rows, dtype=np.int64)
    return raw, mask, label, per_class


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Predictor: {PREDICTOR_PATH}")
    fe = Extractor()

    print("\n=== TRAIN ===")
    raw_tr, mask_tr, label_tr, pc_tr = extract_split("train", fe)
    print("\n=== TEST ===")
    raw_te, mask_te, label_te, pc_te = extract_split("test", fe)

    # Re-fit scaler on the PRESENT train rows only (z-score, train stats only).
    present_tr = raw_tr[mask_tr == 1.0]
    mean = present_tr.mean(axis=0)
    std = present_tr.std(axis=0) + 1e-8

    def normalise(raw, mask):
        lm = np.zeros_like(raw)
        present = mask == 1.0
        lm[present] = (raw[present] - mean) / std
        return lm.astype(np.float32)

    lm_tr = normalise(raw_tr, mask_tr)
    lm_te = normalise(raw_te, mask_te)

    np.savez(os.path.join(CACHE_DIR, "align_train.npz"),
             lm=lm_tr, mask=mask_tr, label=label_tr)
    np.savez(os.path.join(CACHE_DIR, "align_test.npz"),
             lm=lm_te, mask=mask_te, label=label_te)
    np.savez(os.path.join(CACHE_DIR, "scaler.npz"),
             mean=mean, std=std, feature_names=np.array(SELECTED_NAMES))

    n_tr_present = int(mask_tr.sum())
    n_te_present = int(mask_te.sum())
    print("\n--- Summary ---")
    print(f"train: {n_tr_present}/{len(mask_tr)} present "
          f"({len(mask_tr)-n_tr_present} missing, mask=0)")
    print(f"test:  {n_te_present}/{len(mask_te)} present "
          f"({len(mask_te)-n_te_present} missing, mask=0)")
    # Sanity: label dist == ImageFolder folder counts (full set, both splits).
    print(f"train label counts: {np.bincount(label_tr).tolist()}")
    print(f"test  label counts: {np.bincount(label_te).tolist()}")
    print(f"\nCached to {CACHE_DIR}/ (align_train.npz, align_test.npz, scaler.npz)")


if __name__ == "__main__":
    main()
