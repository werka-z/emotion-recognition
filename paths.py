"""
paths.py

Central path definitions for the emotion-recognition repo. Every script
should import from here instead of computing BASE_DIR locally — this
means scripts can move/nest freely without breaking path resolution.

Repo root is located by walking upward from this file until a marker
(.git) is found, so this only works correctly if paths.py itself stays
at the repo root.
"""

import os

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def find_repo_root(marker: str = ".git") -> str:
    d = os.path.abspath(os.path.dirname(__file__))
    while not os.path.exists(os.path.join(d, marker)):
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                f"Could not locate repo root (no '{marker}' found above {__file__})"
            )
        d = parent
    return d


BASE_DIR       = find_repo_root()
FER_ROOT       = os.path.join(BASE_DIR, "data")
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
ABLATION_RESULTS_DIR = os.path.join(BASE_DIR, "ablation_results")
FEATURES_DIR         = os.path.join(BASE_DIR, "features")
MODEL_RESULTS_DIR    = os.path.join(BASE_DIR, "model_results")

# Path to the course's `courselib` package (from the AppliedML repo).
# Override by setting the COURSELIB_PATH environment variable if it's
# not cloned as a sibling folder next to this repo, e.g.:
#   export COURSELIB_PATH="/path/to/AppliedML"
COURSELIB_PATH = os.environ.get(
    "COURSELIB_PATH",
    os.path.join(os.path.dirname(BASE_DIR), "AppliedML")
)


def ablation_results_subdir(name: str) -> str:
    """Return ablation_results/<name>/, creating it if needed."""
    d = os.path.join(ABLATION_RESULTS_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def features_dir() -> str:
    """Return features/, creating it if needed. Shared input data for
    all final model scripts (SVM, CNN, etc.)."""
    os.makedirs(FEATURES_DIR, exist_ok=True)
    return FEATURES_DIR


def model_results_subdir(model_type: str, name: str) -> str:
    """Return model_results/<model_type>/<name>/, creating it if needed.

    model_type is e.g. "svm" or "cnn", matching the models/ subfolder
    the calling script lives in.
    """
    d = os.path.join(MODEL_RESULTS_DIR, model_type, name)
    os.makedirs(d, exist_ok=True)
    return d