"""
linear_svm_ablation.py  —  run in your aml environment
─────────────────────────────────────────────────────────
Full ablation for the C-parameterized multiclass SVM

Sweeps across:
  - 4 preprocessing conditions (upscale x CLAHE)
  - 4 scaling settings (no_scale, scale_only, clip3, clip2)
  - 2 regularization settings (none, balanced) — our own extension,
    not covered in the course material, added due to FER-2013's severe
    class imbalance
  - C in [1, 10, 100, 1000]             
  - learning rate fixed

Produces the figures:
  - macro F1 heatmap (condition x reg, per scale x C)
  - best confusion matrix (best macro F1 and best accuracy)
  - feature-count sweep (optional, see RUN_FEATURE_SWEEP flag)

Usage:
    python ablation_studies/linear_svm_ablation.py

Reads from:  ablation_results/features/  (written by extract_ablation.py)
Saves to:    ablation_results/linear_svm_ablation/

Requires the `courselib` package from the course's AppliedML repo. By
default this is expected as a sibling folder next to this repo:
    parent_folder/
    ├── emotion-recognition/      (this repo)
    └── AppliedML/                (courselib)
If yours lives elsewhere, set the COURSELIB_PATH environment variable
before running, e.g.:
    export COURSELIB_PATH="/path/to/AppliedML"
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import f_oneway

warnings.filterwarnings("ignore")
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import EMOTION_LABELS, COURSELIB_PATH, ablation_results_subdir

sys.path.insert(0, COURSELIB_PATH)
from courselib.models.base import TrainableModel
from courselib.optimizers import Optimizer

# ── CONFIG ────────────────────────────────────────────────────────────────────
IN_DIR         = ablation_results_subdir("features")
OUT_DIR        = ablation_results_subdir("linear_svm_ablation")
N_CLASSES      = 7
FEATURE_COUNTS = [5, 8, 10, 12, 15, 18, 20, 22, 24, 26, 28]
NUM_EPOCHS     = 100
BATCH_SIZE     = 256

CONDITIONS = [
    ("no_up_no_cl", "No upscale\nNo CLAHE"),
    ("no_up_cl",    "No upscale\nCLAHE"),
    ("up_no_cl",    "Upscale\nNo CLAHE"),
    ("up_cl",       "Upscale\nCLAHE"),
]
C_GRID         = [1.0, 10.0, 100.0, 1000.0] 
INITIAL_LR     = 0.01 
DECAY_RATE     = 0.1
USE_SCHEDULE   = True
REG_SETTINGS   = [("none", None), ("balanced", "balanced")]
SCALE_SETTINGS = [
    ("no_scale",   None,  None),
    ("scale_only", True,  None),
    ("clip3",      True,  3.0),
    ("clip2",      True,  2.0),
]

metric_keys = ["acc", "mac", "wf1"]


# ── GDOptimizer with schedule ───────────
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


def decay_schedule(step, initial_lr=0.1, decay_rate=0.01):
    return initial_lr / (1 + decay_rate * step)


# ── MODEL ─────────────────────────────────────────────────────────────────────
class LinearSVMMulticlassC(TrainableModel):
    def __init__(self, w, b, optimizer, C=10.0, class_weight=None,
                 n_classes=N_CLASSES):
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

    def fit(self, X_tr, y_tr, num_epochs=100, batch_size=256):
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(X_tr))
            batches = np.array_split(indices,
                                     int(np.ceil(len(X_tr) / batch_size)))
            for idx in batches:
                grads = self.loss_grad(X_tr[idx], y_tr[idx])
                self.optimizer.update(self._get_params(), grads)


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


def weighted_f1(y_true, y_pred, n_classes=N_CLASSES):
    f1s, weights = [], []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1s.append(2 * precision * recall / (precision + recall + 1e-8))
        weights.append(np.sum(y_true == c))
    weights = np.array(weights) / len(y_true)
    return float(np.dot(f1s, weights))


def cm_from_pred(y_true, y_pred, n_classes=N_CLASSES):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def apply_scale(Xtr, Xte, do_scale, clip):
    if not do_scale:
        return Xtr.copy(), Xte.copy()
    mean  = Xtr.mean(axis=0)
    std   = Xtr.std(axis=0)
    std   = np.where(std < 1e-6, 1.0, std)
    Xtr_s = (Xtr - mean) / std
    Xte_s = (Xte - mean) / std
    if clip is not None:
        Xtr_s = np.clip(Xtr_s, -clip, clip)
        Xte_s = np.clip(Xte_s, -clip, clip)
    return Xtr_s, Xte_s


def compute_f_stats(Xtr, ytr):
    f_stats = []
    for fi in range(Xtr.shape[1]):
        groups  = [Xtr[ytr == li, fi] for li in range(7) if (ytr == li).sum() > 1]
        stat, _ = f_oneway(*groups)
        f_stats.append(float(stat) if np.isfinite(stat) else 0.0)
    return np.array(f_stats)


def get_top_idx(f_stats, n):
    return np.argsort(f_stats)[::-1][:n]


def train_and_eval(Xtr, ytr, Xte, yte, C, cw):
    n_feat = Xtr.shape[1]
    w      = np.zeros((n_feat, N_CLASSES))
    b      = np.zeros((1, N_CLASSES))
    if USE_SCHEDULE:
        opt = GDOptimizer(schedule_fn=lambda step: decay_schedule(
            step, initial_lr=INITIAL_LR, decay_rate=DECAY_RATE))
    else:
        opt = GDOptimizer(learning_rate=INITIAL_LR)
    model = LinearSVMMulticlassC(w, b, opt, C=C, class_weight=cw,
                                  n_classes=N_CLASSES)
    try:
        model.fit(Xtr, ytr, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        if not np.all(np.isfinite(model.w)):
            raise FloatingPointError("Diverged")
        y_pred = model(Xte)
        return (float(np.mean(y_pred == yte)),
                macro_f1(yte, y_pred),
                weighted_f1(yte, y_pred),
                cm_from_pred(yte, y_pred))
    except (FloatingPointError, OverflowError):
        nan_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        return np.nan, np.nan, np.nan, nan_cm


def model_name(C): return f"SVM (C={C:g})"
def all_model_names(): return [model_name(C) for C in C_GRID]


# ── PLOTTING ──────────────────────────────────────────────────────────────────
def find_best(results, metric="mac"):
    best_mn, best_sn, best_ci, best_ri = None, None, -1, -1
    best_val = -1
    for mn in all_model_names():
        for sn, _, _ in SCALE_SETTINGS:
            for ci in range(len(CONDITIONS)):
                for ri in range(len(REG_SETTINGS)):
                    v = results[mn][sn][metric][ci, ri]
                    if not np.isnan(v) and v > best_val:
                        best_val = v
                        best_mn, best_sn, best_ci, best_ri = mn, sn, ci, ri
    return best_mn, best_sn, best_ci, best_ri, best_val


def plot_all_figures(results, cms, out_dir, n_feats):
    cond_labels_short = [cl.replace("\n", " ") for _, cl in CONDITIONS]
    reg_labels_short  = [rl for rl, _ in REG_SETTINGS]
    scale_names       = [sn for sn, _, _ in SCALE_SETTINGS]
    mnames            = all_model_names()

    # Macro F1 heatmap — rows = C, cols = scale
    fig, axes = plt.subplots(len(C_GRID), len(SCALE_SETTINGS),
                              figsize=(5 * len(SCALE_SETTINGS), 4 * len(C_GRID)))
    fig.suptitle(f"C-parameterized SVM (HW7-style) — Macro F1 ({n_feats} features)\n"
                 "(rows = C, cols = scaling, best over conditions x reg)",
                 fontsize=12, fontweight="bold", y=1.01)

    global_max = max(
        np.nanmax(results[mn][sn]["mac"]) for mn in mnames for sn in scale_names
    )

    for ci_row, C in enumerate(C_GRID):
        mn = model_name(C)
        for si, (sn, _, _) in enumerate(SCALE_SETTINGS):
            ax  = axes[ci_row, si]
            mat = results[mn][sn]["mac"]
            im  = ax.imshow(mat, cmap="YlGn", vmin=0.1, vmax=0.45, aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    v = mat[r, c]
                    txt = "div" if np.isnan(v) else f"{v:.3f}"
                    star = " *" if (not np.isnan(v) and v == global_max) else ""
                    ax.text(c, r, f"{txt}{star}", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if (not np.isnan(v) and v > 0.3) else "black")
            ax.set_xticks([0, 1]); ax.set_xticklabels(reg_labels_short, fontsize=8)
            ax.set_yticks(range(4)); ax.set_yticklabels(cond_labels_short, fontsize=7)
            ax.set_xlabel("Class weight", fontsize=8)
            ax.set_title(f"C={C:g}  scale={sn}", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_macro_f1_heatmap.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out_dir}/fig_macro_f1_heatmap.png")

    # Best CMs: macro F1 and accuracy side by side
    best_mn_mac, best_sn_mac, best_ci_mac, best_ri_mac, best_mac = find_best(results, "mac")
    best_mn_acc, best_sn_acc, best_ci_acc, best_ri_acc, best_acc_v = find_best(results, "acc")

    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
    fig2.suptitle(f"C-parameterized SVM — Best Configurations ({n_feats} features)\n"
                  "Left: best macro F1   Right: best accuracy",
                  fontsize=12, fontweight="bold")

    for ax2, (best_mn, best_sn, best_ci, best_ri, metric_val, metric_name) in zip(axes2, [
        (best_mn_mac, best_sn_mac, best_ci_mac, best_ri_mac, best_mac,   "Macro F1"),
        (best_mn_acc, best_sn_acc, best_ci_acc, best_ri_acc, best_acc_v, "Accuracy"),
    ]):
        cm      = cms[best_mn][best_sn][best_ci, best_ri].astype(float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        acc     = results[best_mn][best_sn]["acc"][best_ci, best_ri]
        mac     = results[best_mn][best_sn]["mac"][best_ci, best_ri]

        im = ax2.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        for r in range(N_CLASSES):
            for c in range(N_CLASSES):
                v = cm_norm[r, c]
                ax2.text(c, r, f"{v:.2f}", ha="center", va="center",
                         fontsize=9, color="white" if v > 0.5 else "black")
        ax2.set_xticks(range(N_CLASSES))
        ax2.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right", fontsize=10)
        ax2.set_yticks(range(N_CLASSES))
        ax2.set_yticklabels(EMOTION_LABELS, fontsize=10)
        ax2.set_xlabel("Predicted", fontsize=11)
        ax2.set_ylabel("True", fontsize=11)
        ax2.set_title(f"Best {metric_name}: {best_mn}\n"
                      f"{CONDITIONS[best_ci][0]} | scale={best_sn} | "
                      f"reg={REG_SETTINGS[best_ri][0]}\n"
                      f"acc={acc:.4f}  mac={mac:.4f}",
                      fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_best_cm.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out_dir}/fig_best_cm.png")

    print(f"\nFULL SUMMARY ({n_feats} features)")
    print("-" * 80)
    print(f"Best macro F1: {best_mac:.4f}")
    print(f"  {best_mn_mac} | {CONDITIONS[best_ci_mac][0]} | "
          f"scale={best_sn_mac} | reg={REG_SETTINGS[best_ri_mac][0]}")
    print(f"  acc={results[best_mn_mac][best_sn_mac]['acc'][best_ci_mac, best_ri_mac]:.4f}")
    print(f"\nBest accuracy: {best_acc_v:.4f}")
    print(f"  {best_mn_acc} | {CONDITIONS[best_ci_acc][0]} | "
          f"scale={best_sn_acc} | reg={REG_SETTINGS[best_ri_acc][0]}")
    print(f"  mac={results[best_mn_acc][best_sn_acc]['mac'][best_ci_acc, best_ri_acc]:.4f}")


# ── ABLATION RUN (single feature count) ───────────────────────────────────────
def run_ablation(out_dir, top_idx_by_condition, n_feats):
    """
    top_idx_by_condition: dict {condition_prefix: top_idx array}, since
    feature selection (ANOVA F-stat) is computed per-condition.
    """
    os.makedirs(out_dir, exist_ok=True)
    mnames  = all_model_names()
    results = {mn: {sn: {mk: np.zeros((4, 2)) for mk in metric_keys}
                    for sn, _, _ in SCALE_SETTINGS}
               for mn in mnames}
    cms     = {mn: {sn: np.zeros((4, 2, N_CLASSES, N_CLASSES), dtype=int)
                    for sn, _, _ in SCALE_SETTINGS}
               for mn in mnames}

    print(f"\nTraining all combinations ({n_feats} features, "
          f"lr={INITIAL_LR}, schedule={USE_SCHEDULE})...\n")

    for ci, (prefix, cond_label) in enumerate(CONDITIONS):
        Xtr_raw = np.load(os.path.join(IN_DIR, f"{prefix}_X_train.npy"))
        ytr     = np.load(os.path.join(IN_DIR, f"{prefix}_y_train.npy"))
        Xte_raw = np.load(os.path.join(IN_DIR, f"{prefix}_X_test.npy"))
        yte     = np.load(os.path.join(IN_DIR, f"{prefix}_y_test.npy"))

        top_idx = top_idx_by_condition[prefix]
        Xtr_raw = Xtr_raw[:, top_idx]
        Xte_raw = Xte_raw[:, top_idx]

        for sn, do_scale, clip in SCALE_SETTINGS:
            Xtr_sc, Xte_sc = apply_scale(Xtr_raw, Xte_raw, do_scale, clip)
            for ri, (reg_label, cw) in enumerate(REG_SETTINGS):
                for C in C_GRID:
                    mn = model_name(C)
                    acc, mac, wf1, cm = train_and_eval(Xtr_sc, ytr, Xte_sc, yte, C, cw)
                    results[mn][sn]["acc"][ci, ri] = acc
                    results[mn][sn]["mac"][ci, ri] = mac
                    results[mn][sn]["wf1"][ci, ri] = wf1
                    cms[mn][sn][ci, ri]            = cm
                    cl = cond_label.replace("\n", " ")
                    status = "DIVERGED" if np.isnan(acc) else f"acc={acc:.4f} mac={mac:.4f}"
                    print(f"  [{cl}] [scale={sn}] [reg={reg_label}] [C={C:g}]  {status}")

    plot_all_figures(results, cms, out_dir, n_feats)
    np.save(os.path.join(out_dir, "results.npy"), results, allow_pickle=True)
    np.save(os.path.join(out_dir, "cms.npy"), cms, allow_pickle=True)
    print(f"\nSaved results.npy and cms.npy -> {out_dir}/")
    return results, cms


# ── FEATURE COUNT SWEEP ───────────────────────────────────────────────────────
def run_feature_sweep():
    SWEEP_OUT_DIR = os.path.join(OUT_DIR, "feature_sweep")
    os.makedirs(SWEEP_OUT_DIR, exist_ok=True)
    mnames = all_model_names()

    print(f"\n{'='*60}")
    print(f"  Feature count sweep over {FEATURE_COUNTS}")
    print(f"{'='*60}\n")

    sweep_best     = {mn: {"acc": [], "mac": []} for mn in mnames}
    best_ci_global = -1
    best_mac_global = -1

    for n in FEATURE_COUNTS:
        print(f"\n  n={n} features")
        best_per_model = {mn: {"acc": -1, "mac": -1, "ci_acc": -1, "ci_mac": -1}
                          for mn in mnames}

        for ci, (prefix, _) in enumerate(CONDITIONS):
            Xtr_raw = np.load(os.path.join(IN_DIR, f"{prefix}_X_train.npy"))
            ytr     = np.load(os.path.join(IN_DIR, f"{prefix}_y_train.npy"))
            Xte_raw = np.load(os.path.join(IN_DIR, f"{prefix}_X_test.npy"))
            yte     = np.load(os.path.join(IN_DIR, f"{prefix}_y_test.npy"))

            f_stats = compute_f_stats(Xtr_raw, ytr)
            top_idx = get_top_idx(f_stats, n)
            Xtr_raw = Xtr_raw[:, top_idx]
            Xte_raw = Xte_raw[:, top_idx]

            for sn, do_scale, clip in SCALE_SETTINGS:
                Xtr_sc, Xte_sc = apply_scale(Xtr_raw, Xte_raw, do_scale, clip)
                for ri, (_, cw) in enumerate(REG_SETTINGS):
                    for C in C_GRID:
                        mn = model_name(C)
                        acc, mac, wf1, cm = train_and_eval(Xtr_sc, ytr, Xte_sc, yte, C, cw)
                        if not np.isnan(acc) and acc > best_per_model[mn]["acc"]:
                            best_per_model[mn]["acc"]    = acc
                            best_per_model[mn]["ci_acc"] = ci
                        if not np.isnan(mac) and mac > best_per_model[mn]["mac"]:
                            best_per_model[mn]["mac"]    = mac
                            best_per_model[mn]["ci_mac"] = ci

        for mn in mnames:
            sweep_best[mn]["acc"].append(best_per_model[mn]["acc"])
            sweep_best[mn]["mac"].append(best_per_model[mn]["mac"])
            print(f"    [{mn}]  best acc={best_per_model[mn]['acc']:.4f}"
                  f"  best mac={best_per_model[mn]['mac']:.4f}")

        best_mac_n = max(best_per_model[mn]["mac"] for mn in mnames)
        best_mn_n  = max(mnames, key=lambda mn: best_per_model[mn]["mac"])
        if best_mac_n > best_mac_global:
            best_mac_global = best_mac_n
            best_ci_global  = best_per_model[best_mn_n]["ci_mac"]

    np.save(os.path.join(SWEEP_OUT_DIR, "sweep_results.npy"),
            sweep_best, allow_pickle=True)
    np.save(os.path.join(SWEEP_OUT_DIR, "feature_counts.npy"),
            np.array(FEATURE_COUNTS))
    print(f"\nSaved sweep_results.npy -> {SWEEP_OUT_DIR}/")

    best_mac_per_n = [max(sweep_best[mn]["mac"][i] for mn in mnames)
                      for i in range(len(FEATURE_COUNTS))]
    best_acc_per_n = [max(sweep_best[mn]["acc"][i] for mn in mnames)
                      for i in range(len(FEATURE_COUNTS))]
    optimal_n_mac = FEATURE_COUNTS[int(np.argmax(best_mac_per_n))]
    optimal_n_acc = FEATURE_COUNTS[int(np.argmax(best_acc_per_n))]
    print(f"Optimal feature count (best macro F1): {optimal_n_mac}")
    print(f"Optimal feature count (best accuracy):  {optimal_n_acc}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("C-parameterized SVM — Performance vs number of features\n"
                 "(best result per C across all conditions x scaling x reg)",
                 fontsize=12, fontweight="bold")
    for ax, (mkey, mlabel, opt_n) in zip(axes, [
        ("acc", "Accuracy", optimal_n_acc),
        ("mac", "Macro F1", optimal_n_mac),
    ]):
        for mn in mnames:
            ax.plot(FEATURE_COUNTS, sweep_best[mn][mkey], marker="o", label=mn)
        ax.axvline(opt_n, ls="--", color="grey", alpha=0.7,
                   label=f"Optimal (n={opt_n})")
        ax.set_xlabel("Number of features")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.set_xticks(FEATURE_COUNTS)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out_path = os.path.join(SWEEP_OUT_DIR, "fig_feature_count_sweep.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out_path}")

    return optimal_n_mac, best_ci_global


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Step 1: Feature count sweep")
    print(f"{'='*60}")
    optimal_n, best_ci = run_feature_sweep()

    print(f"\n{'='*60}")
    print(f"  Step 2: {optimal_n}-feature full ablation")
    print(f"{'='*60}")

    top_idx_by_condition = {}
    for prefix, _ in CONDITIONS:
        Xtr_raw_c = np.load(os.path.join(IN_DIR, f"{prefix}_X_train.npy"))
        ytr_c     = np.load(os.path.join(IN_DIR, f"{prefix}_y_train.npy"))
        f_stats_c = compute_f_stats(Xtr_raw_c, ytr_c)
        top_idx_by_condition[prefix] = get_top_idx(f_stats_c, optimal_n)

    results, cms = run_ablation(
        out_dir=os.path.join(OUT_DIR, f"{optimal_n}feat"),
        top_idx_by_condition=top_idx_by_condition,
        n_feats=optimal_n
    )