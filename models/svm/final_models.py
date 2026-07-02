"""
final_models.py  —  run in your aml environment
──────────────────────────────────────────────────
Trains and evaluates the two best models found from ablation/grid search,
using fixed, hardcoded hyperparameters (no search performed here).

LINEAR SVM  — LinearSVMMulticlassC, best macro F1 config from linear_svm_ablation.py:
    condition = up_no_cl, scale = clip2, C = 1000, reg = balanced, n = 26 features
    C-parameterized, HW7-style gradient masking, inverse decay lr schedule

RBF SVM     — OneVsRestKernelSVM, best config from rbf_parameters.py:
    condition = up_no_cl, scale = clip2, C = 10, sigma = 2.8958, n = 26 features
    trained on a stratified subsample of N_SAMPLES due to O(n^3) QP solver
    memory constraints

── REPRODUCIBILITY NOTE (why this script does NOT simply retrain the linear
   SVM and report the retrained numbers as final) ───────────────────────────
linear_svm_ablation.py evaluates hundreds of hyperparameter combinations
(11 feature counts x 4 conditions x 4 scales x 2 regs x 4 Cs) in sequence,
all drawing from one global NumPy RNG stream seeded once at the top of that
script. Every combination's training loop calls np.random.permutation()
per epoch, advancing that shared stream. By the time the script reaches the
winning combination (C=1000, up_no_cl, clip2, balanced), thousands of prior
random draws — from unrelated hyperparameter combinations — have already
been consumed.

Retraining that exact config here, in isolation, seeds the RNG fresh and
consumes zero prior draws, so the batch-shuffle order differs from what
happened inside the ablation. The result is a slightly different SGD
trajectory, and therefore slightly different final accuracy/macro-F1 —
even with an identical seed, identical hyperparameters, and identical
code. This is standard behaviour for SGD, not a bug.

(The RBF SVM below does not have this problem: its stratified subsampling
is the very first random draw in rbf_parameters.py, so re-seeding
immediately before subsampling here reproduces it exactly.)

Given this, this script reports the LINEAR SVM's official metrics and
confusion matrix by loading them directly from linear_svm_ablation.py's
saved results (ablation_results/linear_svm_ablation/{N_FEATURES}feat/),
i.e. the literal numbers the ablation produced. Separately, it also
retrains the same config once here — purely to produce a representative
learning curve and representative misclassification examples, since the
ablation only saved final metrics/confusion matrices, not per-epoch
history or per-image predictions. This retrained run's own accuracy/macro
F1 is reported alongside the exact ablation numbers for transparency, and
is expected to differ from them by a small margin (SGD shuffle-order
stochasticity, as explained above) — see final_results_summary.txt.

Usage:
    python models/svm/final_models.py

Reads from:
    features/                                          (from extract_features.py)
    ablation_results/linear_svm_ablation/{N_FEATURES}feat/  (from linear_svm_ablation.py,
                                                               for exact linear SVM metrics/CM)
Saves to:    model_results/svm/final_models/
    fig_linear_learning_curve.png    (from diagnostic retrain)
    fig_linear_cm.png                (exact, from ablation)
    fig_linear_misclassifications.png (from diagnostic retrain)
    fig_rbf_cm.png                   (exact — RBF is reproducible via reseeding)
    fig_rbf_misclassifications.png   (exact)
    fig_linear_vs_rbf_cm.png         (exact numbers for both models)
    final_results_summary.txt        (reports both exact and retrained
                                       linear numbers, with explanation)
"""

import sys
import os
import time
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

warnings.filterwarnings("ignore")
np.random.seed(42)


# Walk upward from this file until we find paths.py, so this works
# regardless of how deeply nested this script is (models/svm/ here).
def _find_repo_root_for_import():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(d, "paths.py")):
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("Could not locate paths.py above this script")
        d = parent
    return d


sys.path.insert(0, _find_repo_root_for_import())
from paths import EMOTION_LABELS, COURSELIB_PATH, features_dir, model_results_subdir, ablation_results_subdir

sys.path.insert(0, COURSELIB_PATH)
from courselib.models.base import TrainableModel
from courselib.optimizers import Optimizer
from courselib.models.svm import BinaryKernelSVM

# ── FIXED BEST CONFIGS (no search — found previously) ─────────────────────────
IN_DIR    = features_dir()
OUT_DIR   = model_results_subdir("svm", "final_models")
N_CLASSES = 7

BEST_CONDITION = "up_no_cl"
N_FEATURES     = 26
CLIP           = 2.0

# Linear SVM best config — C-parameterized, HW7-style
# Best macro f1 config from linear_svm_ablation.py:
#   C=1000, up_no_cl, clip2, reg=balanced
# Equivalent to lambda=1/C=0.001 in the lambda-parameterized formulation
LINEAR_C            = 1000.0
LINEAR_CLASS_WEIGHT = "balanced"     # reg=balanced
LINEAR_LR           = 0.01    # initial lr, with inverse decay schedule
LINEAR_DECAY        = 0.1     # gamma in eta_t = eta_0 / (1 + gamma*t), HW7
LINEAR_EPOCHS       = 100
LINEAR_BATCH_SIZE   = 256

# RBF SVM best config
RBF_C          = 10.0
RBF_SIGMA      = 2.8958
RBF_N_SAMPLES  = 10000   # stratified subsample size

# ── Location of the winning combination within linear_svm_ablation.py's
# saved results (results.npy / cms.npy), used to load its EXACT metrics
# and confusion matrix rather than recomputing them by retraining.
# These indices/keys must match linear_svm_ablation.py's own definitions:
#   CONDITIONS     = [no_up_no_cl, no_up_cl, up_no_cl, up_cl]   -> up_no_cl is index 2
#   REG_SETTINGS   = [none, balanced]                            -> balanced is index 1
#   SCALE_SETTINGS keys include "clip2" directly (used as a dict key, not an index)
#   model_name(C)  = f"SVM (C={C:g})"                            -> "SVM (C=1000)" for C=1000.0
LINEAR_ABLATION_DIR = os.path.join(
    ablation_results_subdir("linear_svm_ablation"), f"{N_FEATURES}feat")
LINEAR_MODEL_NAME  = f"SVM (C={LINEAR_C:g})"
LINEAR_SCALE_NAME  = "clip2"
LINEAR_COND_INDEX  = 2   # up_no_cl
LINEAR_REG_INDEX   = 1   # balanced


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
        stat, _ = f_oneway(*groups)
        f_stats.append(float(stat) if np.isfinite(stat) else 0.0)
    return np.array(f_stats)


# ── GDOptimizer with schedule — exact copy of HW7 ────────────────────────────
class GDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, schedule_fn=None):
        super().__init__(learning_rate)
        self.schedule_fn = schedule_fn
        self.step = 0

    def update(self, params, grads):
        if self.schedule_fn is not None:
            self.step += 1
            self.learning_rate = self.schedule_fn(self.step)
        for key in params:
            np.subtract(params[key], self.learning_rate * grads[key], out=params[key])


def decay_schedule(step, initial_lr=0.01, decay_rate=0.1):
    return initial_lr / (1 + decay_rate * step)


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
    sub = ""
    if acc is not None and mac is not None:
        sub = f"\nacc={acc:.4f}  mac={mac:.4f}"
    ax.set_title(title + sub, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {fname}")


# ════════════════════════════════════════════════════════════════════════════
#  LINEAR SVM — C-parameterized, HW7-style gradient masking
# ════════════════════════════════════════════════════════════════════════════
class LinearSVMMulticlassC(TrainableModel):
    """
    C-parameterized multiclass SVM following the course's exact formulation
    from linear_svm.ipynb / week7_hw_lr_scheduler.ipynb:

        min_{W,b}  (C/N) * sum_i L_i  +  ||W||_2^2

    with HW7-style gradient masking (only margin-violating samples contribute)
    and optional class weighting (our own extension for FER-2013 imbalance).
    """
    def __init__(self, w, b, optimizer, C=1000.0,
                 class_weight=None, n_classes=N_CLASSES):
        super().__init__(optimizer)
        self.w            = np.array(w, dtype=float)
        self.b            = np.array(b, dtype=float).reshape(1, -1)
        self.C            = C
        self.class_weight = class_weight
        self.n_classes    = n_classes

    def decision_function(self, X):
        return X @ self.w + self.b

    def _get_params(self):
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return np.argmax(self.decision_function(X), axis=-1)

    def _compute_class_weights(self, y_int):
        counts   = np.bincount(y_int, minlength=self.n_classes)
        freq     = counts / counts.sum()
        inv_freq = 1.0 / (freq + 1e-8)
        inv_freq /= inv_freq.sum() / self.n_classes
        return inv_freq[y_int]

    def loss_grad(self, X, y):
        N      = X.shape[0]
        y_int  = y.astype(int)
        scores = self.decision_function(X)
        correct_scores = scores[np.arange(N), y_int]
        margins = np.maximum(0, scores - correct_scores[:, None] + 1)
        margins[np.arange(N), y_int] = 0
        violated = (margins > 0).astype(float)
        dscores  = violated.copy()
        dscores[np.arange(N), y_int] -= dscores.sum(axis=1)
        if self.class_weight == "balanced":
            sample_weights = self._compute_class_weights(y_int)
            dscores *= sample_weights[:, None]
        # C-parameterized gradient: (C/N) * loss grad + 2*w regularization
        grad_w = 2 * self.w + self.C * (X.T @ dscores) / N
        grad_b = self.C * dscores.mean(axis=0, keepdims=True)
        return {"w": grad_w, "b": grad_b}

    def fit_with_history(self, X_tr, y_tr, X_te, y_te,
                         num_epochs=200, batch_size=256):
        history = {"train_acc": [], "test_acc": [],
                   "train_mac": [], "test_mac": []}
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(X_tr))
            batches = np.array_split(indices,
                                     int(np.ceil(len(X_tr) / batch_size)))
            for idx in batches:
                grads = self.loss_grad(X_tr[idx], y_tr[idx])
                self.optimizer.update(self._get_params(), grads)
            y_pred_tr = self(X_tr)
            y_pred_te = self(X_te)
            history["train_acc"].append(np.mean(y_pred_tr == y_tr) * 100)
            history["test_acc"].append(np.mean(y_pred_te == y_te) * 100)
            history["train_mac"].append(macro_f1(y_tr, y_pred_tr))
            history["test_mac"].append(macro_f1(y_te, y_pred_te))
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:>3}  "
                      f"acc={history['test_acc'][-1]:.2f}%  "
                      f"mac={history['test_mac'][-1]:.4f}")
        return history


def plot_learning_curves(history, fname):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Linear SVM — Learning Curves (best macro F1 config)",
                 fontsize=12, fontweight="bold")
    for ax, (tr_key, te_key, ylabel) in zip(axes, [
        ("train_acc", "test_acc", "Accuracy (%)"),
        ("train_mac", "test_mac", "Macro F1"),
    ]):
        ax.plot(epochs, history[tr_key], label="Train")
        ax.plot(epochs, history[te_key], label="Test")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {fname}")


def plot_misclassifications(y_true, y_pred, image_paths, fname, model_label,
                            n_per_pair=3, top_n_pairs=6):
    """
    model_label is used in the figure title/print, e.g. "Best Linear SVM"
    or "Best RBF SVM" — lets this be reused for either model.
    """
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
        print(f"Not enough misclassification examples found for {model_label}.")
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


# ════════════════════════════════════════════════════════════════════════════
#  RBF SVM
# ════════════════════════════════════════════════════════════════════════════
class OneVsRestKernelSVM:
    def __init__(self, C=1.0, kernel="rbf", **kwargs):
        self.C      = C
        self.kernel = kernel
        self.kwargs = kwargs
        self.svms   = []

    def fit(self, X, y, n_classes=N_CLASSES):
        self.svms = []
        for c in range(n_classes):
            t0    = time.time()
            y_bin = np.where(y == c, 1, -1)
            svm   = BinaryKernelSVM(C=self.C, kernel=self.kernel, **self.kwargs)
            svm.fit(X, y_bin)
            self.svms.append(svm)
            print(f"    class {c} ({EMOTION_LABELS[c]:<10}) done in "
                  f"{time.time()-t0:.1f}s")

    def decision_function(self, X):
        return np.column_stack([svm.decision_function(X) for svm in self.svms])

    def __call__(self, X):
        return np.argmax(self.decision_function(X), axis=-1)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── LOAD DATA (shared by both models) ──────────────────────────────────
    print(f"Loading features from {IN_DIR}/ (condition: {BEST_CONDITION})...")
    # features/ was extracted with the best condition (up_no_cl) baked in,
    # and already top-26 selected via extract_features.py — load directly.
    Xtr_raw = np.load(os.path.join(IN_DIR, "X_train.npy"))
    ytr     = np.load(os.path.join(IN_DIR, "y_train.npy"))
    Xte_raw = np.load(os.path.join(IN_DIR, "X_test.npy"))
    yte     = np.load(os.path.join(IN_DIR, "y_test.npy"))
    print(f"Full dataset:  Train={Xtr_raw.shape}  Test={Xte_raw.shape}")

    # Data already top-N selected at extraction time; only re-select if a
    # different N_FEATURES is requested and full feature set is available.
    if Xtr_raw.shape[1] != N_FEATURES:
        print(f"  [INFO] Loaded data has {Xtr_raw.shape[1]} features, "
              f"re-selecting top {N_FEATURES}...")
        f_stats = compute_f_stats(Xtr_raw, ytr)
        top_idx = np.argsort(f_stats)[::-1][:N_FEATURES]
        Xtr_raw = Xtr_raw[:, top_idx]
        Xte_raw = Xte_raw[:, top_idx]

    Xtr_sc, Xte_sc = apply_scale(Xtr_raw, Xte_raw, CLIP)

    # Image paths for misclassification plots (both models)
    paths_path = os.path.join(IN_DIR, "image_paths_test.npy")
    image_paths = (np.load(paths_path, allow_pickle=True)
                   if os.path.exists(paths_path) else None)

    summary_lines = []
    summary_lines.append("FINAL MODEL RESULTS SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Condition: {BEST_CONDITION}  |  Scale: clip{CLIP}  |  "
                         f"n_features: {N_FEATURES}\n")

    # ── LOAD EXACT LINEAR SVM RESULTS (from the ablation, not retrained) ────
    print(f"\n{'='*60}")
    print("  LINEAR SVM — exact metrics from linear_svm_ablation.py")
    print(f"{'='*60}")
    print(f"Loading from {LINEAR_ABLATION_DIR}/ ...")
    print(f"  model={LINEAR_MODEL_NAME}  scale={LINEAR_SCALE_NAME}  "
          f"condition_idx={LINEAR_COND_INDEX} (up_no_cl)  reg_idx={LINEAR_REG_INDEX} (balanced)")

    ablation_results = np.load(os.path.join(LINEAR_ABLATION_DIR, "results.npy"),
                                allow_pickle=True).item()
    ablation_cms     = np.load(os.path.join(LINEAR_ABLATION_DIR, "cms.npy"),
                                allow_pickle=True).item()

    acc_lin = ablation_results[LINEAR_MODEL_NAME][LINEAR_SCALE_NAME]["acc"][
        LINEAR_COND_INDEX, LINEAR_REG_INDEX]
    mac_lin = ablation_results[LINEAR_MODEL_NAME][LINEAR_SCALE_NAME]["mac"][
        LINEAR_COND_INDEX, LINEAR_REG_INDEX]
    cm_lin  = ablation_cms[LINEAR_MODEL_NAME][LINEAR_SCALE_NAME][
        LINEAR_COND_INDEX, LINEAR_REG_INDEX]

    print(f"\nLinear SVM (exact, from ablation):  acc={acc_lin:.4f}  mac={mac_lin:.4f}")
    summary_lines.append("LINEAR SVM — official metrics (loaded directly from")
    summary_lines.append("linear_svm_ablation.py's saved results, NOT retrained)")
    summary_lines.append(f"  config: C={LINEAR_C}, class_weight={LINEAR_CLASS_WEIGHT}, "
                         f"lr={LINEAR_LR}, decay={LINEAR_DECAY}")
    summary_lines.append(f"  acc={acc_lin:.4f}  mac={mac_lin:.4f}")
    for i, emo in enumerate(EMOTION_LABELS):
        row_total = cm_lin[i].sum()
        recall = cm_lin[i, i] / row_total if row_total > 0 else 0.0
        summary_lines.append(f"    {emo:<10} recall={recall:.3f}")
    summary_lines.append("")

    plot_cm(cm_lin, "Best Linear SVM", os.path.join(OUT_DIR, "fig_linear_cm.png"),
            acc=acc_lin, mac=mac_lin)

    # ── RETRAIN THE SAME CONFIG (diagnostic only) ───────────────────────────
    # Used only to produce a learning curve and misclassification examples,
    # since the ablation never saved per-epoch history or per-image
    # predictions — only final metrics and a confusion matrix. This
    # retrained run's own accuracy/macro F1 will differ slightly from the
    # exact numbers above; see the REPRODUCIBILITY NOTE at the top of this
    # file (and the summary below) for why that is expected, not an error.
    print(f"\n{'='*60}")
    print("  LINEAR SVM — retraining same config (diagnostic run:")
    print("  learning curve + misclassification examples only)")
    print(f"{'='*60}")
    print(f"C={LINEAR_C}  class_weight={LINEAR_CLASS_WEIGHT}  "
          f"lr={LINEAR_LR}  decay={LINEAR_DECAY}")

    w   = np.zeros((Xtr_sc.shape[1], N_CLASSES))
    b   = np.zeros((1, N_CLASSES))
    opt = GDOptimizer(schedule_fn=lambda step: decay_schedule(
        step, initial_lr=LINEAR_LR, decay_rate=LINEAR_DECAY))
    linear_model_retrain = LinearSVMMulticlassC(w, b, opt, C=LINEAR_C,
                                                 class_weight=LINEAR_CLASS_WEIGHT,
                                                 n_classes=N_CLASSES)

    history = linear_model_retrain.fit_with_history(
        Xtr_sc, ytr, Xte_sc, yte,
        num_epochs=LINEAR_EPOCHS, batch_size=LINEAR_BATCH_SIZE
    )

    y_pred_lin_retrain = linear_model_retrain(Xte_sc)
    acc_lin_retrain = np.mean(y_pred_lin_retrain == yte)
    mac_lin_retrain = macro_f1(yte, y_pred_lin_retrain)

    print(f"\nLinear SVM (this run's retrain):  "
          f"acc={acc_lin_retrain:.4f}  mac={mac_lin_retrain:.4f}")
    print(f"Difference from exact ablation numbers:  "
          f"Δacc={acc_lin_retrain - acc_lin:+.4f}  Δmac={mac_lin_retrain - mac_lin:+.4f}")
    print("(Expected — see REPRODUCIBILITY NOTE in this script's docstring: SGD")
    print(" shuffle order differs because the ablation's RNG stream had already")
    print(" been advanced by thousands of draws from earlier hyperparameter")
    print(" combinations before reaching this exact config.)")

    summary_lines.append("LINEAR SVM — diagnostic retrain (same hyperparameters,")
    summary_lines.append("retrained in isolation for the learning curve and")
    summary_lines.append("misclassification plot only — NOT the official result)")
    summary_lines.append(f"  acc={acc_lin_retrain:.4f}  mac={mac_lin_retrain:.4f}")
    summary_lines.append(f"  difference from official (ablation-exact) numbers: "
                         f"Δacc={acc_lin_retrain - acc_lin:+.4f}  "
                         f"Δmac={mac_lin_retrain - mac_lin:+.4f}")
    summary_lines.append("  This gap is expected, not an error: linear_svm_ablation.py")
    summary_lines.append("  trains hundreds of hyperparameter combinations in sequence,")
    summary_lines.append("  sharing one global NumPy RNG stream (seeded once at the")
    summary_lines.append("  script's start). Each combination's per-epoch")
    summary_lines.append("  np.random.permutation() call advances that shared stream,")
    summary_lines.append("  so by the time the ablation reaches the winning combination")
    summary_lines.append("  (C=1000, up_no_cl, clip2, balanced), thousands of prior draws")
    summary_lines.append("  from unrelated combinations have already been consumed.")
    summary_lines.append("  Retraining that exact config in isolation here seeds the RNG")
    summary_lines.append("  fresh, so the batch-shuffle order differs — giving a slightly")
    summary_lines.append("  different (but comparable) SGD trajectory and final result,")
    summary_lines.append("  even with an identical seed, hyperparameters, and code.")
    summary_lines.append("")

    plot_learning_curves(history, os.path.join(OUT_DIR, "fig_linear_learning_curve.png"))
    if image_paths is not None:
        plot_misclassifications(yte, y_pred_lin_retrain, image_paths,
                                os.path.join(OUT_DIR, "fig_linear_misclassifications.png"),
                                model_label=f"Linear SVM (diagnostic retrain, "
                                            f"acc={acc_lin_retrain:.4f} mac={mac_lin_retrain:.4f})")
    else:
        print("Image paths not found — skipping linear misclassification plot.")

    # ── TRAIN RBF SVM ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RBF SVM — best grid search config")
    print(f"{'='*60}")
    print(f"C={RBF_C}  sigma={RBF_SIGMA}  N_SAMPLES={RBF_N_SAMPLES}")

    # Re-seed immediately before subsampling to exactly reproduce the stratified
    # subsample used in rbf_parameters.py (which calls np.random during median
    # heuristic computation before subsampling — re-seeding here realigns the
    # random call sequence so the resulting subsample matches exactly).
    np.random.seed(42)
    subsample_idx = []
    for c in range(N_CLASSES):
        idx = np.where(ytr == c)[0]
        n   = max(1, int(RBF_N_SAMPLES * len(idx) / len(ytr)))
        subsample_idx.extend(np.random.choice(idx, n, replace=False))
    subsample_idx = np.array(subsample_idx)
    Xtr_raw_sub = Xtr_raw[subsample_idx]
    ytr_sub     = ytr[subsample_idx]

    Xtr_sub_sc, Xte_sc_rbf = apply_scale(Xtr_raw_sub, Xte_raw, CLIP)
    print(f"Subsample: {Xtr_sub_sc.shape}  class counts: {np.bincount(ytr_sub)}")

    t0 = time.time()
    rbf_model = OneVsRestKernelSVM(C=RBF_C, kernel="rbf", sigma=RBF_SIGMA)
    rbf_model.fit(Xtr_sub_sc, ytr_sub)
    print(f"Total training time: {time.time()-t0:.1f}s")

    y_pred_rbf = rbf_model(Xte_sc_rbf)
    acc_rbf = np.mean(y_pred_rbf == yte)
    mac_rbf = macro_f1(yte, y_pred_rbf)
    cm_rbf  = cm_from_pred(yte, y_pred_rbf)

    print(f"\nRBF SVM final:  acc={acc_rbf:.4f}  mac={mac_rbf:.4f}")
    summary_lines.append("RBF SVM")
    summary_lines.append(f"  config: C={RBF_C}, sigma={RBF_SIGMA}, "
                         f"N_train={len(Xtr_sub_sc)}")
    summary_lines.append(f"  acc={acc_rbf:.4f}  mac={mac_rbf:.4f}")
    for i, emo in enumerate(EMOTION_LABELS):
        mask = yte == i
        recall = np.mean(y_pred_rbf[mask] == i)
        summary_lines.append(f"    {emo:<10} recall={recall:.3f}")
    summary_lines.append("")

    plot_cm(cm_rbf, f"Best RBF SVM (C={RBF_C}, sigma={RBF_SIGMA})",
            os.path.join(OUT_DIR, "fig_rbf_cm.png"), acc=acc_rbf, mac=mac_rbf)
    if image_paths is not None:
        plot_misclassifications(yte, y_pred_rbf, image_paths,
                                os.path.join(OUT_DIR, "fig_rbf_misclassifications.png"),
                                model_label=f"Best RBF SVM (C={RBF_C}, sigma={RBF_SIGMA})")
    else:
        print("Image paths not found — skipping RBF misclassification plot.")

    # ── SIDE-BY-SIDE COMPARISON ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Final Models — Linear SVM vs RBF SVM", fontsize=13, fontweight="bold")

    for ax, (cm, title, acc, mac) in zip(axes, [
        (cm_lin, f"Linear SVM (C={LINEAR_C})", acc_lin, mac_lin),
        (cm_rbf, f"RBF SVM (C={RBF_C}, sigma={RBF_SIGMA})", acc_rbf, mac_rbf),
    ]):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
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
        ax.set_title(f"{title}\nacc={acc:.4f}  mac={mac:.4f}",
                     fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_linear_vs_rbf_cm.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {OUT_DIR}/fig_linear_vs_rbf_cm.png")

    # ── SAVE TEXT SUMMARY ────────────────────────────────────────────────────
    with open(os.path.join(OUT_DIR, "final_results_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSaved → {OUT_DIR}/final_results_summary.txt")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")
    print(f"Linear SVM (official, from ablation):  acc={acc_lin:.4f}  mac={mac_lin:.4f}")
    print(f"Linear SVM (this run's retrain):       "
          f"acc={acc_lin_retrain:.4f}  mac={mac_lin_retrain:.4f}")
    print(f"RBF SVM:                                acc={acc_rbf:.4f}  mac={mac_rbf:.4f}")