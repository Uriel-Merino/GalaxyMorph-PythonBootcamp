"""
5_ml.py  --  EXERCISE 5: TEACH A COMPUTER TO TELL SPIRALS FROM ELLIPTICALS
==========================================================================

What is this file?
    The "moment of truth": we train a machine-learning model that looks at a
    galaxy and answers "spiral" or "elliptical".

The big idea (Machine Learning in 4 sentences):
    1. We have examples where we KNOW the right answer (Galaxy Zoo labels).
    2. We hide 25% of them (the "test set") and show the model the other 75% (the "train set").
    3. The model learns patterns from the train set.
    4. We ask it to guess the hidden 25%: if it guesses right often, it really learned!

What are "features"?
    The numbers describing each galaxy that the model is allowed to look at.
    A table with one row per galaxy and one column per feature = "feature table" (X).
    The right answers (spiral=1, elliptical=0) = "target array" (y).

We try THREE different feature sets to see what helps most:
    - "catalog": only numbers from the catalog (magnitudes, colors, concentration...)
    - "stamps" : only the pixels of the pictures
    - "both"   : catalog + pictures together

WARNING (data leakage!):
    We do NOT use columns like p_el, p_cs, spiral, elliptical as features. They
    ARE the answer (volunteer votes). Using them = cheating, and the result would be fake.

How to run:
    python 5_ml.py     (needs sample.csv, stamps.npy, stamps_done.npy from 4_cutouts.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

SEED = 42      # fixed random seed -> same results every run
N_PCA = 64     # we shrink 4096 pixel-features down to 64 (see PCA below)

# Catalog columns the model may use (they were created in utils.py)
CAT_FEATURES = ["mag_u", "mag_g", "mag_r", "mag_i", "mag_z",   # brightness in each filter
                "u_g", "g_r", "r_i", "i_z",                    # colors
                "concentration",                               # how packed the light is
                "petror50_r", "petror90_r",                    # sizes (radii)
                "fracdev_r"]                                   # how much it looks like an elliptical profile


def load_data():
    """
    Build the feature tables and the target array.
    Returns: X_cat (catalog numbers), X_img (pixels), y (answers), feats (names used)
    """
    sample = pd.read_csv("sample.csv")
    stamps = np.load("stamps.npy", mmap_mode="r")    # the pictures
    done = np.load("stamps_done.npy")                # which pictures downloaded OK

    # Keep only galaxies whose picture exists (drop failed downloads)
    keep = np.where(done)[0]
    sample = sample.iloc[keep].reset_index(drop=True)
    stamps = np.asarray(stamps[keep])

    # STAMP -> FEATURES
    # 1) Colour picture -> grayscale. Each pixel = 0.299*R + 0.587*G + 0.114*B
    #    (the weights come from how the human eye sees brightness).
    gray = stamps[..., :3] @ np.array([0.299, 0.587, 0.114])
    # 2) Pixel values go from 0-255 to 0-1, and each 64x64 picture is flattened
    #    into ONE row of 4096 numbers (64*64). Now pictures are a table!
    X_img = (gray / 255.0).reshape(len(gray), -1)

    # Catalog features (only the ones that really exist in the table)
    feats = [c for c in CAT_FEATURES if c in sample.columns]
    X_cat = sample[feats].values

    # Target: 1 = spiral, 0 = elliptical
    y = (sample["label"] == "spiral").astype(int).values
    return X_cat, X_img, y, feats


def make_model(kind, n_cat, n_img):
    """
    Build the machine-learning "recipe" (a Pipeline = steps done one after another).

    kind: "catalog", "stamps" or "both".

    Ingredients:
      - StandardScaler: puts every feature on a similar scale (mean 0, spread 1).
            Without it, a big number like a pixel count would overpower a small one like a color.
      - PCA (Principal Component Analysis): 4096 pixels are too many and very repetitive
            (neighbouring pixels look alike). PCA squeezes them into 64 "summary" numbers
            that keep most of the information.
      - SVC (Support Vector Classifier): the actual classifier. It draws the best
            boundary between "spiral" and "elliptical" in the feature space.

    Why a Pipeline? PCA and the scaler must learn ONLY from the train set. Putting
    them inside the pipeline guarantees that (no peeking at the test set).
    """
    # The big table is [catalog columns | pixel columns]. These are their positions:
    cat_idx = list(range(n_cat))
    img_idx = list(range(n_cat, n_cat + n_img))

    cat_pipe = Pipeline([("scale", StandardScaler())])
    img_pipe = Pipeline([("pca", PCA(n_components=N_PCA, random_state=SEED)),
                         ("scale", StandardScaler())])

    # Pick which column groups this model is allowed to see
    parts = []
    if kind in ("catalog", "both"):
        parts.append(("cat", cat_pipe, cat_idx))
    if kind in ("stamps", "both"):
        parts.append(("img", img_pipe, img_idx))

    return Pipeline([("prep", ColumnTransformer(parts)),   # prepare the features
                     ("svc", SVC(kernel="rbf"))])           # then classify


def main():
    X_cat, X_img, y, feats = load_data()
    print(f"N galaxies = {len(y)}   catalog features = {len(feats)}   pixels = {X_img.shape[1]}")

    # MERGE FEATURES: glue catalog columns and pixel columns into one big table
    X = np.hstack([X_cat, X_img])

    # Split into train (75%) and test (25%).
    # stratify=y keeps the same spiral/elliptical proportion in both parts.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    # SVC "knobs" (hyperparameters). GridSearchCV tries every combination
    # and keeps the one that works best (checked with 3-fold cross validation:
    # train on 2/3 of the train set, check on the remaining 1/3, repeat 3 times).
    #   C     : how strict the model is about mistakes (big C = very strict)
    #   gamma : how "curvy" the decision boundary can be
    grid = {"svc__C": [1, 10], "svc__gamma": ["scale", 0.01]}

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, kind in zip(axes, ["catalog", "stamps", "both"]):
        model = make_model(kind, X_cat.shape[1], X_img.shape[1])

        # TRAIN (n_jobs=-1 -> use all CPU cores)
        search = GridSearchCV(model, grid, cv=3, n_jobs=-1).fit(Xtr, ytr)

        # PREDICT on the hidden test galaxies and measure the accuracy
        # (accuracy = fraction of correct answers, 1.0 = perfect)
        pred = search.predict(Xte)
        acc = (pred == yte).mean()
        results[kind] = acc

        print(f"\n=== {kind} | best knobs {search.best_params_} | test accuracy = {acc:.3f}")
        # precision / recall / f1 for each class
        print(classification_report(yte, pred, target_names=["elliptical", "spiral"]))

        # CONFUSION MATRIX: a small table of "what the model said" vs "the truth".
        # Diagonal = correct. Off-diagonal = mistakes.
        ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=["E", "S"],
                                                ax=ax, colorbar=False)
        ax.set_title(f"{kind}: acc={acc:.3f}")

    fig.tight_layout()
    fig.savefig("confusion_matrices.png", dpi=200)
    print("\nSummary:", results)


if __name__ == "__main__":
    main()
