# emotion-recognition — SVM pipeline

The classical/SVM side of the project: FER-2013 emotion recognition from dlib
68-landmark geometric features with SVMs (linear + RBF). This file documents the
SVM scripts, their run order, and the environments they need. For the CNN models
, see `CNN/README.md` and `CNN_notebook.ipynb`.

## Repository structure:

**courselib (AppliedML repo)** is required by `linear_svm_ablation.py`
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

Two conda environments cover the whole project:

- **fer_feature_extraction** (dlib) — the feature-extraction / preprocessing scripts
- **appliedml** — everything else

Create them with `conda env create -f <file>`: `environment.yml` builds
`fer_feature_extraction`, `environment_appliedml.yml` builds `appliedml`.

### data
`data/` and `shape_predictor_68_face_landmarks.dat` need to be set up locally:

Download the data from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data) and place it in the `data` folder of the `emotion-recognition-test` repo. Also download dlib's detector from dlib's model zoo at [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), click on the `.bz2` file to turn it into a `.dat` file, and place the `.dat` file at the repo root.

It should look like this:

```
parent_folder/
├── AppliedML/                                     (courselib)
└── emotion-recognition-test/                       (this repo)
    ├── data/                                       (downloaded data)
    |     ├── train/<Emotion>/*.jpg
    |     └── test/<Emotion>/*.jpg
    └── shape_predictor_68_face_landmarks.dat        (downloaded detector)
```

Jupyter notebook in your IDE should warn you about when to switch kernels, and when it does, click on the "change kernel" button and select the appropriate kernel.

## Running the pipeline (with ablation)

If you want to run them manually:

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

conda activate appliedml
python ablation_studies/linear_svm_ablation.py
python ablation_studies/rbf_parameters.py

python models/svm/final_models.py
python models/svm/final_models_sklearn.py
```

To skip the ablation, just run:

```bash
conda activate fer_feature_extraction

python ablation_studies/extract_ablation.py

conda activate appliedml

python models/svm/final_models.py
python models/svm/final_models_sklearn.py
```

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
| `final_models.py` | `model_results/svm/final_models/` | final trained courselib SVM scripts (not ablations) |
| `final_models_sklearn.py` | `model_results/svm/final_models_sklearn/` | final trained sklearn SVM script (not ablations) |

## Final feature extraction & models

`extract_features.py` runs feature extraction once, using
the best condition/parameters found by the ablations above, and saves
to `features/` — shared input data for every model in `models/`.
