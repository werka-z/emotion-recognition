"""
final_models_sklearn.py  —  run in your aml environment
─────────────────────────────────────────────────────────
Trains the best RBF SVM configuration on the FULL FER-2013 training set
using sklearn's SVC, which uses libsvm's SMO implementation internally.
Unlike courselib's BinaryKernelSVM (cvxopt QP solver), sklearn's SVC never
constructs the full N×N kernel matrix — it computes kernel values on demand,
making full-dataset training feasible on consumer hardware.

This script is intended as a reference/upper-bound result alongside the
courselib result (trained on 10,000 samples due to memory constraints).

Best config from rbf_parameters.py ablation:
    condition = up_no_cl, scale = clip2, C = 10, sigma = 2.8958
    sigma → sklearn gamma:  gamma = 1 / (2 * sigma^2) = 0.06125

Usage:
    python models/svm/final_models_sklearn.py

Reads from:  features/                       (from extract_features.py)
             model_results/svm/final_models/ (for linear SVM CM comparison)
Saves to:    model_results/svm/final_models_sklearn/
    fig_rbf_sklearn_cm.png
    fig_rbf_sklearn_misclassifications.png
    fig_all_models_comparison.png
    final_results_summary.txt
"""

import sys
import os
import time
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from paths import EMOTION_LABELS, features_dir, model_results_subdir

# ── CONFIG ────────────────────────────────────────────────────────────────────
IN_DIR    = features_dir()
OUT_DIR   = model_results_subdir("svm", "final_models_sklearn")
N_CLASSES = 7
CLIP      = 2.0
N_FEATURES = 26

# Best RBF config from rbf_parameters.py
# sigma = 2.8958  →  gamma = 1 / (2 * sigma^2)
RBF_C     = 10.0
RBF_SIGMA = 2.8958
RBF_GAMMA = 1.0 / (2.0 * RBF_SIGMA ** 2)

os.makedirs(OUT_DIR, exist_ok=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def macro_f1(y_true, y_pred, n_classes=N_CLASSES):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1s.append(2 * precision * recall / (precision + recall + 1e-8))
    return np.mean(f1s)


def cm_from_pred(y_true, y_pred, n_classes=N_CLASSES):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def apply_scale(Xtr, Xte, clip):
    mean  = Xtr.mean(axis=0)
    std   = Xtr.std(axis=0)
    std   = np.where(std < 1e-6, 1.0, std)
    Xtr_s = np.clip((Xtr - mean) / std, -clip, clip)
    Xte_s = np.clip((Xte - mean) / std, -clip, clip)
    return Xtr_s, Xte_s


def compute_f_stats(Xtr, ytr):
    f_stats = []
    for fi in range(Xtr.shape[1]):
        groups  = [Xtr[ytr == li, fi] for li in range(7) if (ytr == li).sum() > 1]
        from scipy.stats import f_oneway as fowa
        stat, _ = fowa(*groups)
        f_stats.append(float(stat) if np.isfinite(stat) else 0.0)
    return np.array(f_stats)


def plot_cm(cm, title, fname, acc=None, mac=None):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for r in range(N_CLASSES):
        for c in range(N_CLASSES):
            v = cm_norm[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if v > 0.5 else "black")
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(EMOTION_LABELS, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    sub = f"\nacc={acc:.4f}  mac={mac:.4f}" if acc is not None else ""
    ax.set_title(title + sub, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {fname}")


def plot_misclassifications(y_true, y_pred, image_paths, fname, model_label,
                            n_per_pair=3, top_n_pairs=6):
    import cv2
    confused_pairs = []
    for true_c in range(N_CLASSES):
        for pred_c in range(N_CLASSES):
            if true_c == pred_c:
                continue
            indices = np.where((y_true == true_c) & (y_pred == pred_c))[0]
            if len(indices) >= n_per_pair:
                confused_pairs.append((true_c, pred_c, indices))
    confused_pairs.sort(key=lambda x: -len(x[2]))
    top_pairs = confused_pairs[:top_n_pairs]

    if not top_pairs:
        print(f"Not enough misclassification examples for {model_label}.")
        return

    n_pairs = len(top_pairs)
    fig, axes = plt.subplots(n_pairs, n_per_pair + 1,
                             figsize=((n_per_pair + 1) * 2.2, n_pairs * 2.4),
                             gridspec_kw={"width_ratios": [1.8] + [1] * n_per_pair})
    fig.suptitle(f"Most Common Misclassifications — {model_label}\n"
                 "(True label → Predicted label)",
                 fontsize=11, fontweight="bold")
    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    for row, (true_c, pred_c, indices) in enumerate(top_pairs):
        axes[row, 0].axis("off")
        axes[row, 0].text(0.5, 0.5,
                          f"True: {EMOTION_LABELS[true_c]}\n"
                          f"→ Pred: {EMOTION_LABELS[pred_c]}\n"
                          f"({len(indices)} samples)",
                          ha="center", va="center", fontsize=9,
                          fontweight="bold", transform=axes[row, 0].transAxes)
        shown = 0
        for idx in indices[:n_per_pair * 3]:
            if shown >= n_per_pair:
                break
            img = cv2.imread(image_paths[idx], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            axes[row, shown + 1].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[row, shown + 1].axis("off")
            shown += 1
        for col in range(shown + 1, n_per_pair + 1):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {fname}")


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print(f"Loading features from {IN_DIR}/...")
Xtr_raw = np.load(os.path.join(IN_DIR, "X_train.npy"))
ytr     = np.load(os.path.join(IN_DIR, "y_train.npy"))
Xte_raw = np.load(os.path.join(IN_DIR, "X_test.npy"))
yte     = np.load(os.path.join(IN_DIR, "y_test.npy"))
print(f"Full dataset:  Train={Xtr_raw.shape}  Test={Xte_raw.shape}")

if Xtr_raw.shape[1] != N_FEATURES:
    print(f"  [INFO] re-selecting top {N_FEATURES} features...")
    f_stats = compute_f_stats(Xtr_raw, ytr)
    top_idx = np.argsort(f_stats)[::-1][:N_FEATURES]
    Xtr_raw = Xtr_raw[:, top_idx]
    Xte_raw = Xte_raw[:, top_idx]

Xtr_sc, Xte_sc = apply_scale(Xtr_raw, Xte_raw, CLIP)

paths_path  = os.path.join(IN_DIR, "image_paths_test.npy")
image_paths = np.load(paths_path, allow_pickle=True) if os.path.exists(paths_path) else None

print(f"\nClass counts (train): {np.bincount(ytr)}")
print(f"gamma = 1/(2*sigma^2) = 1/(2*{RBF_SIGMA}^2) = {RBF_GAMMA:.6f}")


# ── TRAIN sklearn SVC ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  RBF SVM — sklearn SVC (libsvm SMO, full dataset)")
print(f"{'='*60}")
print(f"C={RBF_C}  sigma={RBF_SIGMA}  gamma={RBF_GAMMA:.6f}")
print(f"N_train={len(ytr)} (full dataset)")
print("\nNote: sklearn SVC uses libsvm's SMO internally — never builds the full")
print("N×N kernel matrix, so full-dataset training is feasible.")

t0  = time.time()
svc = SVC(
    kernel="rbf",
    C=RBF_C,
    gamma=RBF_GAMMA,        # equivalent to sigma=2.8958
    decision_function_shape="ovr",
    random_state=42,
    verbose=True
)
svc.fit(Xtr_sc, ytr)
elapsed = time.time() - t0
print(f"\nTraining time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Support vectors per class: {svc.n_support_}")
print(f"Total support vectors: {sum(svc.n_support_)}")

y_pred = svc.predict(Xte_sc)
acc    = np.mean(y_pred == yte)
mac    = macro_f1(yte, y_pred)
cm     = cm_from_pred(yte, y_pred)

print(f"\nFinal:  acc={acc:.4f}  mac={mac:.4f}")
print(f"\nPer-class recall:")
for i, emo in enumerate(EMOTION_LABELS):
    mask = yte == i
    print(f"  {emo:<10} recall={np.mean(y_pred[mask] == i):.3f}")


# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
summary_lines = [
    "RBF SVM — sklearn SVC (libsvm SMO, full dataset)",
    "=" * 50,
    f"C={RBF_C}  sigma={RBF_SIGMA}  gamma={RBF_GAMMA:.6f}",
    f"N_train={len(ytr)} (full FER-2013 training set)",
    f"Solver: sklearn SVC (libsvm SMO — no full kernel matrix)",
    f"Training time: {elapsed:.1f}s",
    f"Total support vectors: {sum(svc.n_support_)}",
    "",
    f"acc={acc:.4f}  mac={mac:.4f}",
    "",
]
for i, emo in enumerate(EMOTION_LABELS):
    mask = yte == i
    summary_lines.append(f"  {emo:<10} recall={np.mean(y_pred[mask] == i):.3f}")

with open(os.path.join(OUT_DIR, "final_results_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))
print(f"\nSaved → {OUT_DIR}/final_results_summary.txt")


# ── PLOTS ─────────────────────────────────────────────────────────────────────
plot_cm(cm,
        f"RBF SVM — sklearn SVC (C={RBF_C}, σ={RBF_SIGMA})\nFull dataset (N={len(ytr)})",
        os.path.join(OUT_DIR, "fig_rbf_sklearn_cm.png"),
        acc=acc, mac=mac)

if image_paths is not None:
    plot_misclassifications(yte, y_pred, image_paths,
                            os.path.join(OUT_DIR, "fig_rbf_sklearn_misclassifications.png"),
                            model_label=f"RBF SVM — sklearn SVC (full dataset)")



print(f"\n{'='*60}")
print("  DONE")
print(f"{'='*60}")
print(f"RBF SVM (sklearn, full dataset):  acc={acc:.4f}  mac={mac:.4f}")
print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")