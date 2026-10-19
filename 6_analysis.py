"""
6_analysis.py  --  ALL THE EXTRA FIGURES AND CHECKS FOR THE REPORT
===================================================================

What is this file?
    5_ml.py trained the models and gave us ONE number per model (accuracy).
    This script answers the "but is it really good? why? what does the model look at?"
    questions, and makes all the figures you can put in your report.

It makes these figures (saved in the folder  figures/ ):
    01_gallery.png            Example stamps: spirals (top) vs ellipticals (bottom)
    02_catalog_features.png   How the main catalog numbers differ between the two classes
    03_pca.png                What PCA keeps from the images ("eigen-galaxies")
    04_roc.png                ROC curves: how good is each model at ALL decision thresholds
    05_confusion.png          Confusion matrices (what gets mixed up with what)
    06_cv_accuracy.png        5-fold cross-validated accuracy (mean +- spread) = robust number
    07_learning_curve.png     Does MORE training data help?
    08_pca_components.png     How many PCA components do we really need?
    09_mistakes.png           The galaxies the best model was MOST sure about... and got wrong
    10_permutation_importance.png  Which catalog numbers matter most

It also saves figures/results.txt with all the numbers.

How to run:
    python 6_analysis.py
    (needs sample.csv, stamps.npy, stamps_done.npy and 5_ml.py in the same folder)
    It takes several minutes: it trains many models. Be patient :)
"""

import importlib
import os

import matplotlib
matplotlib.use("Agg")                     # draw to files, no window needed (works on a server)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, learning_curve,
                                     train_test_split)

# We reuse the functions we wrote in 5_ml.py.
# (A file name that starts with a number cannot be imported with the normal
#  "import" word, so we use importlib instead. It does the same thing.)
ml = importlib.import_module("5_ml")

SEED = ml.SEED
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

KINDS = ["catalog", "stamps", "both"]
COLORS = {"catalog": "tab:green", "stamps": "tab:orange", "both": "tab:purple"}

# A bigger "knob" grid than in 5_ml.py (there the best value was at the edge of the grid)
GRID = {"svc__C": [3, 10, 30, 100], "svc__gamma": ["scale", 0.003, 0.01]}

plt.rcParams.update({"font.size": 11, "xtick.direction": "in", "ytick.direction": "in",
                     "font.family": "serif"})

report = []                                # every number we print also goes here


def say(text=""):
    """Print a line AND remember it so it is saved in results.txt."""
    print(text)
    report.append(text)


# ============================================================================
# 0. LOAD THE DATA  (exactly as in 5_ml.py, so the train/test split is the SAME)
# ============================================================================
X_cat, X_img, y, feats = ml.load_data()
n_cat, n_img = X_cat.shape[1], X_img.shape[1]
X = np.hstack([X_cat, X_img])              # big table: [catalog | pixels]
N = len(y)

# We also need the COLOUR pictures (for the figures) and the catalog table
sample = pd.read_csv("sample.csv")
keep = np.where(np.load("stamps_done.npy"))[0]
sample = sample.iloc[keep].reset_index(drop=True)
rgb = np.asarray(np.load("stamps.npy", mmap_mode="r")[keep])

# Split by POSITION (index) instead of by values. With the same seed and the same
# stratify, this gives the same split as in 5_ml.py, and we can later find the
# original picture of every test galaxy.
idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.25, stratify=y, random_state=SEED)
Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)   # 5-fold cross-validation

say(f"N = {N} galaxies | train = {len(idx_tr)} | test = {len(idx_te)}")
say(f"Catalog features used: {feats}")


# ============================================================================
# 1. GALLERY: what do the pictures look like?
# ============================================================================
def fig_gallery(n=8):
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.6))
    for row, lab in enumerate(["spiral", "elliptical"]):
        ids = np.where(sample["label"] == lab)[0][:n]       # first n galaxies of this class
        for ax, i in zip(axes[row], ids):
            ax.imshow(rgb[i])
            ax.set_title(lab, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_gallery.png", dpi=200)
    plt.close(fig)


# ============================================================================
# 2. CATALOG FEATURES: how do the classes differ?
# ============================================================================
def fig_catalog_features():
    show = [f for f in ["concentration", "g_r", "u_g", "mag_r", "petror50_r", "fracdev_r"]
            if f in sample.columns]
    fig, axes = plt.subplots(1, len(show), figsize=(3.3 * len(show), 3.2))
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, show):
        # clip the extremes (1% - 99%) so a few crazy values do not squash the plot
        lo, hi = np.percentile(sample[f], [1, 99])
        for lab, c in [("spiral", "tab:blue"), ("elliptical", "tab:red")]:
            ax.hist(sample.loc[sample["label"] == lab, f], bins=40, range=(lo, hi),
                    histtype="step", color=c, lw=1.6, density=True, label=lab)
        ax.set_xlabel(f)
        ax.set_yticks([])
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_catalog_features.png", dpi=200)
    plt.close(fig)


# ============================================================================
# 3. PCA: what does it keep from the images?
# ============================================================================
def fig_pca():
    pca = PCA(n_components=64, random_state=SEED).fit(X_img[idx_tr])   # learn ONLY from train
    cum = np.cumsum(pca.explained_variance_ratio_)
    say(f"PCA: 64 components keep {100 * cum[-1]:.1f}% of the variance of the pictures")

    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(2, 6)
    ax = fig.add_subplot(gs[:, :2])
    ax.plot(np.arange(1, 65), cum, lw=2)
    ax.set_xlabel("number of PCA components")
    ax.set_ylabel("fraction of variance kept")
    ax.grid(alpha=0.3)

    # The "eigen-galaxies": each component is itself a 64x64 picture.
    # Every galaxy = a mix of these basic patterns.
    for k in range(8):
        a = fig.add_subplot(gs[k // 4, 2 + k % 4])
        a.imshow(pca.components_[k].reshape(64, 64), cmap="RdBu_r")
        a.set_title(f"PC {k + 1}", fontsize=9)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_pca.png", dpi=200)
    plt.close(fig)


# ============================================================================
# 4. TRAIN the three models with the bigger grid and collect their predictions
# ============================================================================
def train_all():
    models = {}
    say("\n=== Grid search (bigger grid) ===")
    for kind in KINDS:
        gs = GridSearchCV(ml.make_model(kind, n_cat, n_img), GRID, cv=3, n_jobs=-1).fit(Xtr, ytr)
        pred = gs.predict(Xte)
        acc = (pred == yte).mean()
        models[kind] = dict(model=gs.best_estimator_, params=gs.best_params_,
                            pred=pred, score=gs.decision_function(Xte), acc=acc)
        say(f"{kind:8s} best knobs {gs.best_params_}  test accuracy = {acc:.3f}")
    return models


# ---- ROC curves -------------------------------------------------------------
def fig_roc(models):
    # ROC: for every possible threshold, plot how many spirals we find (y axis)
    # against how many ellipticals we wrongly call spiral (x axis).
    # Closer to the top-left corner = better. AUC = area under the curve (1 = perfect).
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for kind in KINDS:
        fpr, tpr, _ = roc_curve(yte, models[kind]["score"])
        a = auc(fpr, tpr)
        say(f"AUC {kind:8s} = {a:.4f}")
        ax.plot(fpr, tpr, color=COLORS[kind], lw=2, label=f"{kind} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guess")
    ax.set_xlabel("False positive rate (ellipticals called spiral)")
    ax.set_ylabel("True positive rate (spirals found)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_roc.png", dpi=200)
    plt.close(fig)


# ---- Confusion matrices -------------------------------------------------------
def fig_confusion(models):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, kind in zip(axes, KINDS):
        # normalize="true": each ROW sums to 1 -> we read it as a percentage of that class
        ConfusionMatrixDisplay.from_predictions(
            yte, models[kind]["pred"], display_labels=["elliptical", "spiral"],
            normalize="true", values_format=".2f", ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{kind}: acc={models[kind]['acc']:.3f}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/05_confusion.png", dpi=200)
    plt.close(fig)


# ============================================================================
# 5. CROSS-VALIDATION: a more trustworthy accuracy
# ============================================================================
def fig_cv(models):
    # One train/test split can be lucky or unlucky. Cross-validation repeats the
    # experiment 5 times with different splits and gives an average and a spread.
    say("\n=== 5-fold cross-validated accuracy (whole sample) ===")
    means, stds = [], []
    for kind in KINDS:
        est = ml.make_model(kind, n_cat, n_img).set_params(**models[kind]["params"])
        s = cross_val_score(est, X, y, cv=cv, n_jobs=-1)
        means.append(s.mean())
        stds.append(s.std())
        say(f"{kind:8s} {s.mean():.4f} +- {s.std():.4f}")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(KINDS, means, yerr=stds, capsize=6, color=[COLORS[k] for k in KINDS])
    ax.set_ylim(min(means) - 0.05, 1.0)
    ax.set_ylabel("accuracy (5-fold CV)")
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 0.003, f"{m:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(f"{OUT}/06_cv_accuracy.png", dpi=200)
    plt.close(fig)


# ============================================================================
# 6. LEARNING CURVE: would more galaxies help?
# ============================================================================
def fig_learning_curve(models):
    # Train with 10%, 25%, 50%, 75%, 100% of the data and see how accuracy changes.
    # If the curve is still going UP at 100% -> more data would help.
    # If it is flat -> we need a better model or better features instead.
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for kind in KINDS:
        est = ml.make_model(kind, n_cat, n_img).set_params(**models[kind]["params"])
        sizes, _, test_scores = learning_curve(est, X, y, cv=cv, n_jobs=-1,
                                               train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0])
        m, s = test_scores.mean(1), test_scores.std(1)
        ax.plot(sizes, m, "o-", color=COLORS[kind], label=kind)
        ax.fill_between(sizes, m - s, m + s, color=COLORS[kind], alpha=0.2)
    ax.set_xlabel("number of training galaxies")
    ax.set_ylabel("validation accuracy")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/07_learning_curve.png", dpi=200)
    plt.close(fig)


# ============================================================================
# 7. HOW MANY PCA COMPONENTS?
# ============================================================================
def fig_pca_components(models):
    say("\n=== PCA components comparison (5-fold CV) ===")
    ks = [16, 32, 64, 128, 256]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for kind in ["stamps", "both"]:
        means, stds = [], []
        for k in ks:
            ml.N_PCA = k          # make_model reads this number every time it is called
            # The best gamma was tuned for 64 components. With more components the distances
            # between galaxies grow, so gamma must shrink in the same proportion
            # (gamma x number_of_features stays constant). At k=64 this is exactly the best gamma.
            params = dict(models[kind]["params"])
            if params["svc__gamma"] != "scale":           # "scale" already adapts by itself
                extra = n_cat if kind == "both" else 0    # catalog columns also count as features
                params["svc__gamma"] *= (64 + extra) / (k + extra)
            est = ml.make_model(kind, n_cat, n_img).set_params(**params)
            s = cross_val_score(est, X, y, cv=cv, n_jobs=-1)
            means.append(s.mean())
            stds.append(s.std())
            say(f"{kind:8s} PCA={k:3d}  {s.mean():.4f} +- {s.std():.4f}")
        ax.errorbar(ks, means, yerr=stds, fmt="o-", capsize=4, color=COLORS[kind], label=kind)
    ml.N_PCA = 64                 # put the original value back
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("number of PCA components")
    ax.set_ylabel("accuracy (5-fold CV)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/08_pca_components.png", dpi=200)
    plt.close(fig)


# ============================================================================
# 8. THE MISTAKES: which galaxies fool the best model?
# ============================================================================
def fig_mistakes(models, n=16):
    m = models["both"]
    wrong = np.where(m["pred"] != yte)[0]                  # positions (inside the test set) of errors
    # decision_function is the model's "confidence": far from 0 = very sure.
    # We show the errors where the model was MOST sure (they are the most interesting).
    order = wrong[np.argsort(-np.abs(m["score"][wrong]))][:n]
    say(f"\nBest model ('both') made {len(wrong)} mistakes out of {len(yte)} test galaxies")
    if len(wrong) == 0:                                    # nothing to draw
        return

    cols = 8
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.3 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, j in zip(axes.ravel(), order):
        i = idx_te[j]                                       # position in the full sample
        true = "S" if y[i] == 1 else "E"
        pred = "S" if m["pred"][j] == 1 else "E"
        ax.imshow(rgb[i])
        ax.set_title(f"true {true} / pred {pred}", fontsize=9)
    fig.suptitle("Most confident mistakes (S = spiral, E = elliptical)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/09_mistakes.png", dpi=200)
    plt.close(fig)


# ============================================================================
# 9. WHICH CATALOG NUMBERS MATTER?
# ============================================================================
def fig_importance(models):
    # Permutation importance: shuffle ONE feature at random (this destroys its
    # information) and see how much the accuracy drops. Big drop = important feature.
    est = ml.make_model("catalog", n_cat, 0).set_params(**models["catalog"]["params"])
    est.fit(X_cat[idx_tr], ytr)
    pi = permutation_importance(est, X_cat[idx_te], yte, n_repeats=10,
                                random_state=SEED, n_jobs=-1)
    order = np.argsort(pi.importances_mean)
    say("\n=== Permutation importance (drop in accuracy when the feature is shuffled) ===")
    for k in order[::-1]:
        say(f"{feats[k]:15s} {pi.importances_mean[k]:.4f} +- {pi.importances_std[k]:.4f}")
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh([feats[k] for k in order], pi.importances_mean[order],
            xerr=pi.importances_std[order], color="tab:green")
    ax.set_xlabel("accuracy drop when shuffled")
    fig.tight_layout()
    fig.savefig(f"{OUT}/10_permutation_importance.png", dpi=200)
    plt.close(fig)


# ============================================================================
# MAIN: run everything in order
# ============================================================================
if __name__ == "__main__":
    fig_gallery()
    fig_catalog_features()
    fig_pca()
    models = train_all()
    fig_roc(models)
    fig_confusion(models)
    fig_cv(models)
    fig_learning_curve(models)
    fig_pca_components(models)
    fig_mistakes(models)
    fig_importance(models)

    with open(f"{OUT}/results.txt", "w") as f:
        f.write("\n".join(report))
    print(f"\nDone! Figures and results.txt are in the folder '{OUT}/'")
