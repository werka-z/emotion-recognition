"""
rbf_grid_search.py  —  run in your aml environment
────────────────────────────────────────────────────
Tests BinaryKernelSVM wrapped in one-vs-rest multiclass scheme
on a stratified subsample of the 26-feature FER-2013 data.
Runs a grid search over C and sigma using the best preprocessing
condition found from the linear ablation (up_no_cl, clip2).

Subsampling is necessary because of memory concerns.

RBF sigma baseline is set using the median heuristic:
    sigma = median(pairwise distances between training samples)
Grid search then evaluates sigma around this baseline.

Usage:
    python ablation_studies/rbf_grid_search.py

Reads from:
    ablation_results/features/                       (from extract_ablation.py)
    ablation_results/linear_svm_ablation/             (from linear_svm_ablation.py)

Saves to:
    ablation_results/rbf_grid_search/
        grid_search_results.npy
        fig_rbf_grid.png
        fig_rbf_best_cm.png
        fig_rbf_sample_size.png
        fig_linear_vs_rbf_cm.png
"""

import sys
import os
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import f_oneway

warnings.filterwarnings("ignore")
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import EMOTION_LABELS, COURSELIB_PATH, ablation_results_subdir

sys.path.insert(0, COURSELIB_PATH)
from courselib.models.svm import BinaryKernelSVM

# ── CONFIG ────────────────────────────────────────────────────────────────────
IN_DIR     = ablation_results_subdir("features")
LINEAR_DIR = ablation_results_subdir("linear_svm_ablation")
SWEEP_DIR  = os.path.join(LINEAR_DIR, "feature_sweep")
OUT_DIR    = ablation_results_subdir("rbf_grid_search")

N_CLASSES      = 7
N_SAMPLES       = 6000
C_GRID          = [0.1, 1.0, 10.0, 100.0]
SIGMA_GRID      = None   # set after median heuristic

SCALE_NAMES     = ["no_scale", "scale_only", "clip3", "clip2"]
CLIP_VALUES     = [None, None, 3.0, 2.0]
CONDITIONS_KEYS = ["no_up_no_cl", "no_up_cl", "up_no_cl", "up_cl"]
REG_SETTINGS    = [("none", None), ("balanced", "balanced")]


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


def scale(Xtr, Xte, clip=3.0):
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


def median_heuristic(X: np.ndarray) -> float:
    if len(X) > 1000:
        idx = np.random.choice(len(X), 1000, replace=False)
        X   = X[idx]
    return float(np.median(pdist(X)))


# ── ONE-VS-REST WRAPPER ───────────────────────────────────────────────────────
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


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
# Find optimal feature count from the feature sweep summary
sweep_results  = np.load(os.path.join(SWEEP_DIR, "sweep_results.npy"),
                         allow_pickle=True).item()
feature_counts = np.load(os.path.join(SWEEP_DIR, "feature_counts.npy")).tolist()
mnames         = list(sweep_results.keys())
best_mac_per_n = [max(sweep_results[mn]["mac"][i] for mn in mnames)
                  for i in range(len(feature_counts))]
optimal_n      = feature_counts[int(np.argmax(best_mac_per_n))]
print(f"Optimal feature count from C ablation: {optimal_n}")

# Find best condition and scaling from the full ablation results at optimal_n
# (reads ablation_results/linear_svm_ablation/{optimal_n}feat/results.npy,
# which has the full condition x scaling x reg breakdown, unlike the sweep
# which only stores best-per-model values)
FULL_ABLATION_DIR = os.path.join(LINEAR_DIR, f"{optimal_n}feat")
n_results_sw = np.load(os.path.join(FULL_ABLATION_DIR, "results.npy"),
                        allow_pickle=True).item()
n_cms_sw     = np.load(os.path.join(FULL_ABLATION_DIR, "cms.npy"),
                        allow_pickle=True).item()
best_mac_sw  = -1
best_mn_sw   = None
best_sn_sw   = None
best_ci_sw   = -1
best_ri_sw   = -1
for mn in n_results_sw.keys():
    for si, sn in enumerate(SCALE_NAMES):
        if sn not in n_results_sw[mn]:
            continue
        for ci in range(len(CONDITIONS_KEYS)):
            for ri in range(2):
                mac = n_results_sw[mn][sn]["mac"][ci, ri]
                if not np.isnan(mac) and mac > best_mac_sw:
                    best_mac_sw = mac
                    best_mn_sw  = mn
                    best_sn_sw  = sn
                    best_ci_sw  = ci
                    best_ri_sw  = ri

BEST_CONDITION = CONDITIONS_KEYS[best_ci_sw]
CLIP           = CLIP_VALUES[SCALE_NAMES.index(best_sn_sw)]
if CLIP is None:
    CLIP = 0.0   # no_scale or scale_only — use no clip

print(f"Best condition from sweep: {BEST_CONDITION}")
print(f"Best scaling from sweep:   {best_sn_sw} (clip={CLIP})")
print(f"Best C from sweep:         {best_mn_sw}")

print(f"\nLoading {BEST_CONDITION} features...")
Xtr_raw = np.load(os.path.join(IN_DIR, f"{BEST_CONDITION}_X_train.npy"))
ytr     = np.load(os.path.join(IN_DIR, f"{BEST_CONDITION}_y_train.npy"))
Xte_raw = np.load(os.path.join(IN_DIR, f"{BEST_CONDITION}_X_test.npy"))
yte     = np.load(os.path.join(IN_DIR, f"{BEST_CONDITION}_y_test.npy"))

print(f"Full dataset:  Train={Xtr_raw.shape}  Test={Xte_raw.shape}")

# Select top features
f_stats = compute_f_stats(Xtr_raw, ytr)
top_idx = np.argsort(f_stats)[::-1][:optimal_n]
Xtr_raw = Xtr_raw[:, top_idx]
Xte_raw = Xte_raw[:, top_idx]

# Stratified subsample
subsample_idx = []
for c in range(N_CLASSES):
    idx = np.where(ytr == c)[0]
    n   = max(1, int(N_SAMPLES * len(idx) / len(ytr)))
    subsample_idx.extend(np.random.choice(idx, n, replace=False))

subsample_idx = np.array(subsample_idx)
X_sub = Xtr_raw[subsample_idx]
y_sub = ytr[subsample_idx]

print(f"Subsample:     {X_sub.shape}  (stratified from {N_SAMPLES} target)")
print(f"Class counts:  {np.bincount(y_sub)}")

if CLIP > 0:
    X_sub_sc, X_test_sc = scale(X_sub, Xte_raw, clip=CLIP)
else:
    # no clipping — just standardize
    mean = X_sub.mean(axis=0)
    std  = X_sub.std(axis=0)
    std  = np.where(std < 1e-6, 1.0, std)
    X_sub_sc  = (X_sub   - mean) / std
    X_test_sc = (Xte_raw - mean) / std

# ── MEDIAN HEURISTIC ──────────────────────────────────────────────────────────
print("\nComputing median heuristic sigma...")
sigma_median = median_heuristic(X_sub_sc)
print(f"Median heuristic sigma: {sigma_median:.4f}")
print(f"Equivalent sklearn gamma: {1/(2*sigma_median**2):.6f}")

# Build sigma grid around median heuristic
SIGMA_GRID = [round(sigma_median * 0.5, 4),
              round(sigma_median, 4),
              round(sigma_median * 2.0, 4)]
print(f"Sigma grid: {SIGMA_GRID}")

# ── GRID SEARCH ───────────────────────────────────────────────────────────────
grid_results_path = os.path.join(OUT_DIR, "grid_search_results.npy")
if os.path.exists(grid_results_path):
    print("\nLoading existing grid search results...")
    saved = np.load(grid_results_path, allow_pickle=True).item()
    grid_acc  = saved["acc"]
    grid_mac  = saved["mac"]
    grid_cms  = saved["cms"]
    C_GRID    = saved["C"]
    SIGMA_GRID = saved["sigma"]
    print(f"  C grid: {C_GRID}")
    print(f"  Sigma grid: {SIGMA_GRID}")
    print("  Skipping grid search — delete grid_search_results.npy to rerun.")
else:
    print(f"\n{'='*60}")
    print(f"  Grid search: C × sigma")
    print(f"{'='*60}\n")

    grid_acc = np.zeros((len(C_GRID), len(SIGMA_GRID)))
    grid_mac = np.zeros((len(C_GRID), len(SIGMA_GRID)))
    grid_cms = np.zeros((len(C_GRID), len(SIGMA_GRID), N_CLASSES, N_CLASSES), dtype=int)

    for ci, C in enumerate(C_GRID):
        for si, sigma in enumerate(SIGMA_GRID):
            print(f"\nC={C}  sigma={sigma}")
            t0    = time.time()
            model = OneVsRestKernelSVM(C=C, kernel="rbf", sigma=sigma)
            model.fit(X_sub_sc, y_sub)
            y_pred = model(X_test_sc)

            acc = np.mean(y_pred == yte) * 100
            mac = macro_f1(yte, y_pred)
            cm  = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
            for t, p in zip(yte, y_pred):
                cm[t, p] += 1

            grid_acc[ci, si] = acc
            grid_mac[ci, si] = mac
            grid_cms[ci, si] = cm

            print(f"  Total time: {time.time()-t0:.1f}s")
            print(f"  acc={acc:.2f}%  mac={mac:.4f}")
            for i, emo in enumerate(EMOTION_LABELS):
                mask = yte == i
                print(f"    {emo:<10} {np.mean(y_pred[mask]==i)*100:.1f}%")

    # Save results
    grid_results = {"C": C_GRID, "sigma": SIGMA_GRID,
                    "acc": grid_acc, "mac": grid_mac, "cms": grid_cms,
                    "sigma_median": sigma_median, "N_SAMPLES": N_SAMPLES,
                    "optimal_n": optimal_n, "condition": BEST_CONDITION}
    np.save(os.path.join(OUT_DIR, "grid_search_results.npy"),
            grid_results, allow_pickle=True)
    print(f"\nSaved grid_search_results.npy → {OUT_DIR}/")

# ── PLOT: grid heatmaps ───────────────────────────────────────────────────────
sigma_labels = [f"{s:.2f}" for s in SIGMA_GRID]
C_labels     = [str(c) for c in C_GRID]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"RBF SVM Grid Search — C × sigma\n"
             f"({BEST_CONDITION}, clip={CLIP}, n={optimal_n} features, "
             f"N_train={len(X_sub)})",
             fontsize=12, fontweight="bold")

for ax, (mat, title) in zip(axes, [(grid_acc, "Accuracy (%)"),
                                    (grid_mac, "Macro F1")]):
    vmin = mat.min() - 0.5
    vmax = mat.max() + 0.5
    im   = ax.imshow(mat, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for r in range(len(C_GRID)):
        for c in range(len(SIGMA_GRID)):
            v    = mat[r, c]
            star = " ★" if v == mat.max() else ""
            ax.text(c, r, f"{v:.2f}{star}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if v > (vmin + vmax) / 2 else "black")
    ax.set_xticks(range(len(SIGMA_GRID)))
    ax.set_xticklabels([f"σ={s}" for s in sigma_labels], fontsize=9)
    ax.set_yticks(range(len(C_GRID)))
    ax.set_yticklabels([f"C={c}" for c in C_labels], fontsize=9)
    ax.set_xlabel("sigma", fontsize=10)
    ax.set_ylabel("C", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_rbf_grid.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved fig_rbf_grid.png → {OUT_DIR}/")

# ── PLOT: best CM ─────────────────────────────────────────────────────────────
best_flat  = int(np.argmax(grid_mac))
best_ci    = best_flat // len(SIGMA_GRID)
best_si    = best_flat  % len(SIGMA_GRID)
best_C     = C_GRID[best_ci]
best_sigma = SIGMA_GRID[best_si]

cm      = grid_cms[best_ci, best_si].astype(float)
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

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
ax.set_title(f"Best RBF SVM: C={best_C}  sigma={best_sigma}\n"
             f"{BEST_CONDITION}, clip={CLIP}, n={optimal_n} features, "
             f"N_train={len(X_sub)}\n"
             f"acc={grid_acc[best_ci,best_si]:.2f}%  "
             f"mac={grid_mac[best_ci,best_si]:.4f}",
             fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_rbf_best_cm.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved fig_rbf_best_cm.png → {OUT_DIR}/")

# ── PLOT: performance vs sample size ─────────────────────────────────────────
print(f"\nRunning performance vs sample size sweep...")
SAMPLE_SIZES = [1000, 2000, 3000, 4000, 5000, 6000]
size_acc, size_mac = [], []

for n_s in SAMPLE_SIZES:
    sub_idx = []
    for c in range(N_CLASSES):
        idx = np.where(ytr == c)[0]   # sample from full training data
        n   = max(1, int(n_s * len(idx) / len(ytr)))
        sub_idx.extend(np.random.choice(idx, n, replace=False))
    sub_idx = np.array(sub_idx)
    Xs_raw  = Xtr_raw[sub_idx]
    ys      = ytr[sub_idx]
    if CLIP > 0:
        mean_s = Xs_raw.mean(axis=0); std_s = Xs_raw.std(axis=0)
        std_s  = np.where(std_s < 1e-6, 1.0, std_s)
        Xs     = np.clip((Xs_raw - mean_s) / std_s, -CLIP, CLIP)
        Xt     = np.clip((Xte_raw - mean_s) / std_s, -CLIP, CLIP)
    else:
        mean_s = Xs_raw.mean(axis=0); std_s = Xs_raw.std(axis=0)
        std_s  = np.where(std_s < 1e-6, 1.0, std_s)
        Xs     = (Xs_raw  - mean_s) / std_s
        Xt     = (Xte_raw - mean_s) / std_s

    print(f"  n={n_s}...", flush=True)
    t0 = time.time()
    m  = OneVsRestKernelSVM(C=best_C, kernel="rbf", sigma=best_sigma)
    m.fit(Xs, ys)
    yp  = m(Xt)
    acc = np.mean(yp == yte) * 100
    mac = macro_f1(yte, yp)
    size_acc.append(acc)
    size_mac.append(mac)
    print(f"    acc={acc:.2f}%  mac={mac:.4f}  ({time.time()-t0:.1f}s)")

np.save(os.path.join(OUT_DIR, "size_sweep_results.npy"),
        {"sizes": SAMPLE_SIZES, "acc": size_acc, "mac": size_mac},
        allow_pickle=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f"RBF SVM — Performance vs Training Set Size\n"
             f"(C={best_C}, sigma={best_sigma}, {BEST_CONDITION}, clip={CLIP}, "
             f"n={optimal_n} features)",
             fontsize=11, fontweight="bold")
for ax, (vals, ylabel) in zip(axes, [(size_acc, "Accuracy (%)"),
                                      (size_mac, "Macro F1")]):
    ax.plot(SAMPLE_SIZES, vals, marker="o")
    ax.set_xlabel("Training samples")
    ax.set_ylabel(ylabel)
    ax.set_xticks(SAMPLE_SIZES)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_rbf_sample_size.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved fig_rbf_sample_size.png → {OUT_DIR}/")

# ── PLOT: side-by-side RBF vs Linear best CM ─────────────────────────────────
# Reuses the linear-ablation results already loaded above (n_results_sw,
# n_cms_sw, best_mn_sw/best_sn_sw/best_ci_sw/best_ri_sw) instead of
# re-reading them from disk under a different (stale) folder name.
lin_cm      = n_cms_sw[best_mn_sw][best_sn_sw][best_ci_sw, best_ri_sw].astype(float)
lin_cm_norm = lin_cm / lin_cm.sum(axis=1, keepdims=True).clip(min=1)
lin_acc     = n_results_sw[best_mn_sw][best_sn_sw]["acc"][best_ci_sw, best_ri_sw]

rbf_cm_norm = grid_cms[best_ci, best_si].astype(float)
rbf_cm_norm = rbf_cm_norm / rbf_cm_norm.sum(axis=1, keepdims=True).clip(min=1)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Linear SVM vs RBF SVM — Best Confusion Matrices",
             fontsize=13, fontweight="bold")

for ax, (cm_norm, title) in zip(axes, [
    (lin_cm_norm, f"Best Linear SVM\n{best_mn_sw} | n={optimal_n}\n"
                  f"acc={lin_acc:.3f}  mac={best_mac_sw:.4f}"),
    (rbf_cm_norm, f"Best RBF SVM\nC={best_C}  sigma={best_sigma} | n={optimal_n}\n"
                  f"acc={grid_acc[best_ci,best_si]:.2f}%  "
                  f"mac={grid_mac[best_ci,best_si]:.4f}  N_train={len(X_sub)}")
]):
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
    ax.set_title(title, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_linear_vs_rbf_cm.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved fig_linear_vs_rbf_cm.png → {OUT_DIR}/")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  GRID SEARCH SUMMARY")
print(f"{'='*60}")
print(f"Best config:  C={best_C}  sigma={best_sigma}")
print(f"acc={grid_acc[best_ci,best_si]:.2f}%  "
      f"mac={grid_mac[best_ci,best_si]:.4f}")
print(f"\nFull grid (Macro F1):")
print(f"{'':>8}", end="")
for s in SIGMA_GRID:
    print(f"  sigma={s:.2f}", end="")
print()
for ci, C in enumerate(C_GRID):
    print(f"C={C:<6}", end="")
    for si in range(len(SIGMA_GRID)):
        print(f"  {grid_mac[ci,si]:.4f}      ", end="")
    print()