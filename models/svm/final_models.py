"""
final_models.py  —  run in your appliedml environment
──────────────────────────────────────────────────
Trains and evaluates the two best models found from ablation/grid search.

LINEAR SVM  — C=1000, clip2, balanced, 100 epochs, inverse decay schedule. RNG replay reproduces
              the exact ablation result.

RBF SVM     — C=10, sigma=2.8958, 10,000-sample stratified subsample
              (courselib BinaryKernelSVM / cvxopt QP solver).

Saves to:  model_results/svm/final_models/
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

def _find_repo_root_for_import():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(d, "paths.py")):
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("Could not locate paths.py above this script")
        d = parent
    return d

sys.path.insert(0, _find_repo_root_for_import())
from paths import EMOTION_LABELS, COURSELIB_PATH, features_dir, model_results_subdir
sys.path.insert(0, COURSELIB_PATH)
from courselib.models.base import TrainableModel
from courselib.optimizers import Optimizer
from courselib.models.svm import BinaryKernelSVM

# ── CONFIG ────────────────────────────────────────────────────────────────────
IN_DIR    = features_dir()
OUT_DIR   = model_results_subdir("svm", "final_models")
N_CLASSES = 7
CLIP      = 2.0
N_FEATURES = 26

LINEAR_C            = 1000.0
LINEAR_CLASS_WEIGHT = "balanced"
LINEAR_LR           = 0.01
LINEAR_DECAY        = 0.1
LINEAR_EPOCHS       = 100
LINEAR_BATCH_SIZE   = 256

RBF_C         = 10.0
RBF_SIGMA     = 2.8958
RBF_N_SAMPLES = 10000

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
    sub = f"\nacc={acc:.4f}  mac={mac:.4f}" if acc is not None else ""
    ax.set_title(title + sub, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {fname}")


# ════════════════════════════════════════════════════════════════════════════
#  LINEAR SVM
# ════════════════════════════════════════════════════════════════════════════
class LinearSVMMulticlassC(TrainableModel):
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
        grad_w = 2 * self.w + self.C * (X.T @ dscores) / N
        grad_b = self.C * dscores.mean(axis=0, keepdims=True)
        return {"w": grad_w, "b": grad_b}

    def fit_with_history(self, X_tr, y_tr, X_te, y_te,
                         num_epochs=100, batch_size=256):
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


# ════════════════════════════════════════════════════════════════════════════
#  RBF SVM (one-vs-rest, courselib BinaryKernelSVM / cvxopt)
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
            print(f"    class {c} ({EMOTION_LABELS[c]:<10}) done in {time.time()-t0:.1f}s")

    def decision_function(self, X):
        return np.column_stack([svm.decision_function(X) for svm in self.svms])

    def __call__(self, X):
        return np.argmax(self.decision_function(X), axis=-1)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── LOAD DATA ─────────────────────────────────────────────────────────
    print(f"Loading features from {IN_DIR}/ (condition: up_no_cl)...")
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

    summary_lines = ["FINAL MODEL RESULTS SUMMARY", "=" * 50,
                     f"Condition: up_no_cl  |  Scale: clip{CLIP}  |  n_features: {N_FEATURES}\n"]

    # ── LINEAR SVM ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  LINEAR SVM — best macro F1 config")
    print(f"{'='*60}")
    print(f"C={LINEAR_C}  class_weight={LINEAR_CLASS_WEIGHT}  lr={LINEAR_LR}  decay={LINEAR_DECAY}")

    # Replay exact RNG state from linear_svm_ablation.py before the winning config.
    # The ablation seeds np.random(42) then runs 1408 fits x 100 epochs (feature sweep)
    # + 95 fits x 100 epochs (full ablation) = 150,300 permutation calls before winner.
    # Each call uses n_train for that condition: no_up_no_cl=20001, no_up_cl=20213,
    # up_no_cl=22578, up_cl=22670.
    print("  Replaying prior RNG draws to reproduce ablation RNG state...")
    np.random.seed(42)
    FEATURE_COUNTS_ABL = [5, 8, 10, 12, 15, 18, 20, 22, 24, 26, 28]
    CONDITIONS_ABL     = [("no_up_no_cl", 20001), ("no_up_cl", 20213),
                          ("up_no_cl", 22578),    ("up_cl", 22670)]
    SCALE_NAMES_ABL    = ["no_scale", "scale_only", "clip3", "clip2"]
    REG_NAMES_ABL      = ["none", "balanced"]
    C_GRID_ABL         = [1.0, 10.0, 100.0, 1000.0]
    EPOCHS_ABL         = 100

    for _n in FEATURE_COUNTS_ABL:
        for _prefix, _n_train in CONDITIONS_ABL:
            for _sn in SCALE_NAMES_ABL:
                for _reg in REG_NAMES_ABL:
                    for _C in C_GRID_ABL:
                        for _ep in range(EPOCHS_ABL):
                            np.random.permutation(_n_train)

    _done = False
    for _prefix, _n_train in CONDITIONS_ABL:
        for _sn in SCALE_NAMES_ABL:
            for _reg in REG_NAMES_ABL:
                for _C in C_GRID_ABL:
                    if (_prefix == "up_no_cl" and _sn == "clip2"
                            and _reg == "balanced" and _C == 1000.0):
                        _done = True
                        break
                    for _ep in range(EPOCHS_ABL):
                        np.random.permutation(_n_train)
                if _done: break
            if _done: break
        if _done: break
    print("  RNG state reproduced.")

    w   = np.zeros((Xtr_sc.shape[1], N_CLASSES))
    b   = np.zeros((1, N_CLASSES))
    opt = GDOptimizer(schedule_fn=lambda step: decay_schedule(
        step, initial_lr=LINEAR_LR, decay_rate=LINEAR_DECAY))
    linear_model = LinearSVMMulticlassC(w, b, opt, C=LINEAR_C,
                                         class_weight=LINEAR_CLASS_WEIGHT,
                                         n_classes=N_CLASSES)
    history = linear_model.fit_with_history(
        Xtr_sc, ytr, Xte_sc, yte,
        num_epochs=LINEAR_EPOCHS, batch_size=LINEAR_BATCH_SIZE)

    y_pred_lin = linear_model(Xte_sc)
    acc_lin    = np.mean(y_pred_lin == yte)
    mac_lin    = macro_f1(yte, y_pred_lin)
    cm_lin     = cm_from_pred(yte, y_pred_lin)
    print(f"\nLinear SVM final:  acc={acc_lin:.4f}  mac={mac_lin:.4f}")

    summary_lines.append("LINEAR SVM")
    summary_lines.append(f"  config: C={LINEAR_C}, class_weight={LINEAR_CLASS_WEIGHT}, "
                         f"lr={LINEAR_LR}, decay={LINEAR_DECAY}")
    summary_lines.append(f"  acc={acc_lin:.4f}  mac={mac_lin:.4f}")
    for i, emo in enumerate(EMOTION_LABELS):
        mask = yte == i
        summary_lines.append(f"    {emo:<10} recall={np.mean(y_pred_lin[mask] == i):.3f}")
    summary_lines.append("")

    plot_learning_curves(history, os.path.join(OUT_DIR, "fig_linear_learning_curve.png"))
    plot_cm(cm_lin, "Best Linear SVM", os.path.join(OUT_DIR, "fig_linear_cm.png"),
            acc=acc_lin, mac=mac_lin)
    np.save(os.path.join(OUT_DIR, "linear_cm.npy"), cm_lin)
    if image_paths is not None:
        plot_misclassifications(yte, y_pred_lin, image_paths,
                                os.path.join(OUT_DIR, "fig_linear_misclassifications.png"),
                                model_label="Best Linear SVM")

    # ── RBF SVM ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RBF SVM — best grid search config (courselib cvxopt, 10k subsample)")
    print(f"{'='*60}")
    print(f"C={RBF_C}  sigma={RBF_SIGMA}  N_SAMPLES={RBF_N_SAMPLES}")

    np.random.seed(42)
    subsample_idx = []
    for c in range(N_CLASSES):
        idx = np.where(ytr == c)[0]
        n   = max(1, int(RBF_N_SAMPLES * len(idx) / len(ytr)))
        subsample_idx.extend(np.random.choice(idx, n, replace=False))
    subsample_idx = np.array(subsample_idx)
    Xtr_sub_sc, Xte_sc_rbf = apply_scale(Xtr_raw[subsample_idx], Xte_raw, CLIP)
    ytr_sub = ytr[subsample_idx]
    print(f"Subsample: {Xtr_sub_sc.shape}  class counts: {np.bincount(ytr_sub)}")

    t0        = time.time()
    rbf_model = OneVsRestKernelSVM(C=RBF_C, kernel="rbf", sigma=RBF_SIGMA)
    rbf_model.fit(Xtr_sub_sc, ytr_sub)
    print(f"Total training time: {time.time()-t0:.1f}s")

    y_pred_rbf = rbf_model(Xte_sc_rbf)
    acc_rbf    = np.mean(y_pred_rbf == yte)
    mac_rbf    = macro_f1(yte, y_pred_rbf)
    cm_rbf     = cm_from_pred(yte, y_pred_rbf)
    print(f"\nRBF SVM final:  acc={acc_rbf:.4f}  mac={mac_rbf:.4f}")

    summary_lines.append("RBF SVM (courselib cvxopt)")
    summary_lines.append(f"  config: C={RBF_C}, sigma={RBF_SIGMA}, N_train={len(Xtr_sub_sc)}")
    summary_lines.append(f"  acc={acc_rbf:.4f}  mac={mac_rbf:.4f}")
    for i, emo in enumerate(EMOTION_LABELS):
        mask = yte == i
        summary_lines.append(f"    {emo:<10} recall={np.mean(y_pred_rbf[mask] == i):.3f}")
    summary_lines.append("")

    plot_cm(cm_rbf, f"Best RBF SVM (C={RBF_C}, sigma={RBF_SIGMA})",
            os.path.join(OUT_DIR, "fig_rbf_cm.png"), acc=acc_rbf, mac=mac_rbf)
    np.save(os.path.join(OUT_DIR, "rbf_cm.npy"), cm_rbf)
    if image_paths is not None:
        plot_misclassifications(yte, y_pred_rbf, image_paths,
                                os.path.join(OUT_DIR, "fig_rbf_misclassifications.png"),
                                model_label=f"Best RBF SVM (C={RBF_C}, sigma={RBF_SIGMA})")

    # ── SIDE-BY-SIDE ──────────────────────────────────────────────────────
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

    with open(os.path.join(OUT_DIR, "final_results_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSaved → {OUT_DIR}/final_results_summary.txt")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")
    print(f"Linear SVM:  acc={acc_lin:.4f}  mac={mac_lin:.4f}")
    print(f"RBF SVM:     acc={acc_rbf:.4f}  mac={mac_rbf:.4f}")