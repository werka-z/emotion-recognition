# Feature Extraction

This folder contains the dlib-based feature extraction pipeline for the facial emotion recognition project. It must be run before the `svm/` training pipeline.

## Folder Structure

```
feature_extraction/
├── environment.yml           # conda environment for this folder
├── compare_preprocessing.py  # ablation: upscaling vs CLAHE detection rates
├── extract_features.py       # main extraction pipeline
└── processed_final/          # generated outputs (created on first run)
    ├── X_train.npy           # train features  (n_train, 20)
    ├── y_train.npy           # train labels    (n_train,)
    ├── X_test.npy            # test features   (n_test, 20)
    ├── y_test.npy            # test labels     (n_test,)
    ├── feature_names.npy     # selected feature names  (20,)
    ├── feature_names_full.npy# all 28 feature names   (28,)
    ├── f_stats.npy           # ANOVA F-statistic per feature (28,)
    ├── top20_idx.npy         # indices of selected features  (20,)
    ├── scaler_mean.npy       # train mean for inference      (20,)
    └── scaler_std.npy        # train std  for inference      (20,)
```

## Downloads

### Data

Download FER-2013 from Kaggle:
https://www.kaggle.com/datasets/msambare/fer2013

Place the extracted folders so the repo structure looks like:
```
data/
├── train/
│   ├── Angry/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Sad/
│   ├── Surprise/
│   └── Neutral/
└── test/
    └── (same structure)
```

### Landmark Predictor

Download the dlib facial landmark predictor:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Place `shape_predictor_68_face_landmarks.dat.bz2` in the repo root.
`extract_features.py` will decompress it automatically on first run.

## Environment Setup

Tested on MacBook with Apple M5 chip.

```bash
conda env create -f environment.yml
conda activate fer_feature_extraction
```

## Usage

**Step 1 (optional):** Verify that upscaling improves dlib detection rate on your data:
```bash
python compare_preprocessing.py
```

**Step 2:** Run the main extraction pipeline:
```bash
python extract_features.py                        # best config: upscale, no CLAHE, top-20
python extract_features.py --all-features         # upscale, no CLAHE, all 28 features
python extract_features.py --clahe                # adds CLAHE on top of upscaling
python extract_features.py --no-upscale           # skips upscaling (not recommended)
python extract_features.py --no-upscale --clahe   # raw 48x48 with CLAHE
```

This will:
1. Extract 28 geometric features from images using dlib landmarks
2. Select the top 20 features by analysis of variance (ANOVA) F-statistic (computed on train set only)
3. Save the results to `processed_final/`

Typical runtime: ~5-10 minutes for the full FER-2013 dataset.

## Best Preprocessing Configuration for dlib

Determined by tests across upscaling × CLAHE conditions:

| Condition | Detection Rate (train) | Notes |
|---|---|---|
| 48px, no CLAHE | ~71% | baseline |
| 48px, CLAHE | ~73% | marginal gain |
| **224px, no CLAHE** | **~80%** | **best — used in pipeline** |
| 224px, CLAHE | ~79% | CLAHE slightly hurts landmark quality |

Upscaling from 48px to 224px recovers images that dlib would otherwise miss.
CLAHE was removed because it degrades landmark fitting quality despite improving contrast.
