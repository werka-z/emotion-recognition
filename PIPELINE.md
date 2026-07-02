# emotion-recognition

Facial emotion recognition on FER-2013 using dlib 68-landmark geometric
features and SVMs (linear + RBF).

## Repo structure

```
.
├── paths.py                        # shared path resolution — all scripts import from here
├── ablation_studies/                # ablation pipeline, run in order (see below)
│   ├── compare_upscaling.py
│   ├── plot_upscaling.py
│   ├── show_detection_failures.py
│   ├── compare_all_preprocessing.py
│   ├── plot_clahe.py
│   ├── extract_ablation.py
│   ├── linear_svm_ablation.py
│   └── rbf_parameters.py
├── ablation_results/                # ablation outputs, one subfolder per study
│   ├── upscaling/
│   ├── clahe/
│   ├── features/
│   ├── linear_svm_ablation/
│   └── rbf_grid_search/
├── extract_features.py              # final single-condition feature extraction
│                                     # (best condition/params from the ablations above) —
│                                     # shared input for every model below
├── features/                        # output of extract_features.py — X/y/paths .npy files,
│                                     # read by both models/svm/ and models/cnn/
├── models/
│   ├── svm/                         # final trained SVM scripts (not ablations)
│   └── cnn/                         # CNN model scripts
└── model_results/                   # final model outputs, mirrors ablation_results/ pattern
    ├── svm/
    └── cnn/
```

`data/` and `shape_predictor_68_face_landmarks.dat` are gitignored (too
large / not ours to redistribute). To run anything, set them up locally:

- **data/** — place the FER-2013 split as:
  ```
  data/
  ├── train/<Emotion>/*.jpg
  └── test/<Emotion>/*.jpg
  ```
  where `<Emotion>` is one of: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

- **shape_predictor_68_face_landmarks.dat** — download from dlib's model
  zoo and place at repo root.

- **courselib (AppliedML repo)** — required by `linear_svm_ablation.py`
  and `rbf_parameters.py`. By default, `paths.py` expects it cloned as a
  sibling folder next to this repo:
  ```
  parent_folder/
  ├── emotion-recognition/      (this repo)
  └── AppliedML/                (courselib)
  ```
  If yours lives elsewhere, set the `COURSELIB_PATH` environment
  variable before running those scripts:
  ```bash
  export COURSELIB_PATH="/path/to/AppliedML"
  ```

## Environments

Two conda environments are used, matching each script's header comment:

- **fer_feature_extraction** — `compare_upscaling.py`, `plot_upscaling.py`,
  `show_detection_failures.py`, `compare_all_preprocessing.py`,
  `plot_clahe.py`, `extract_ablation.py`
- **aml** — `linear_svm_ablation.py`, `rbf_parameters.py`

See `environment.yml` for `fer_feature_extraction` and
`environment_aml.yml` for `aml` (create with
`conda env create -f <file>`).

## Running the ablation pipeline

Scripts must be run in this order, since later ones read the outputs of
earlier ones. All paths resolve via `paths.py` (repo-root-anchored), so
run every command from the repo root:

```bash
conda activate fer_feature_extraction
python ablation_studies/compare_upscaling.py
python ablation_studies/plot_upscaling.py
python ablation_studies/show_detection_failures.py
python ablation_studies/compare_all_preprocessing.py
python ablation_studies/plot_clahe.py
python ablation_studies/extract_ablation.py

conda activate aml
python ablation_studies/linear_svm_ablation.py
python ablation_studies/rbf_parameters.py
```

**Note:** `paths.py` locates the repo root by walking up to the nearest
`.git` folder. This means it only works inside an actual git repo — run
`git init` (or clone this repo) before running any script. Running from
a plain folder with no `.git` will raise a `RuntimeError`.

## Results

Each script writes to its own subfolder under `ablation_results/`:

| Script | Output folder | Contents |
|---|---|---|
| `compare_upscaling.py` | `ablation_results/upscaling/` | `upscaling_results.npy` |
| `plot_upscaling.py` | `ablation_results/upscaling/` | `compare_upscaling.png`, `compare_upscaling_best.png` |
| `show_detection_failures.py` | `ablation_results/upscaling/` | `failures_recovered.png`, `failures_hard.png` |
| `compare_all_preprocessing.py` | `ablation_results/clahe/` | `detection_counts.npy` |
| `plot_clahe.py` | `ablation_results/clahe/` | `compare_clahe.png` |
| `extract_ablation.py` | `ablation_results/features/` | per-condition `X`/`y`/`image_paths` `.npy` files |
| `linear_svm_ablation.py` | `ablation_results/linear_svm_ablation/` | feature sweep + full ablation figures/results |
| `rbf_parameters.py` | `ablation_results/rbf_grid_search/` | grid search figures/results |

## Final feature extraction & models

`extract_features.py` (repo root) runs feature extraction once, using
the best condition/parameters found by the ablations above, and saves
to `features/` — shared input data for every model in `models/`.

Scripts in `models/svm/` and `models/cnn/` should import from `paths.py`
the same way the ablation scripts do:

```python
from paths import features_dir, model_results_subdir

IN_DIR  = features_dir()
OUT_DIR = model_results_subdir("svm", "linear_svm_final")   # or "cnn", ...
```

This keeps `features/` as a single shared source of truth (no
duplicated extraction per model), and keeps each model's results
separated under `model_results/<model_type>/<name>/`.