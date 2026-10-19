"""
2_exploration.py  --  EXERCISE 2: DATA EXPLORATION
==================================================

What is this file?
    The "let's look at what we downloaded" step. Before doing any science we
    must understand our data: how many galaxies, which columns, which are good, which are bad.

What does it do?
    1. Opens ZooSpecPhoto.csv.
    2. Prints the column names, so you can read them.
    3. Counts how many galaxies are spiral / elliptical / uncertain.
    4. Looks for "strange" objects (stars, broken magnitudes).
    5. Cleans the sample (using the S/N criterion in utils.py).
    6. Saves the clean sample in clean_catalog.csv  (the next scripts use it).

How to run:
    python 2_exploration.py
"""

from utils import load_catalog, add_label, clean_catalog


def explore():
    # ---- 1. Open the file ---------------------------------------------------
    df = load_catalog()
    print(f"Rows (galaxies): {len(df)}   Columns: {df.shape[1]}")

    # ---- 2. Look at the column names ----------------------------------------
    # IMPORTANT: read this list! If 'ra' / 'dec' have a weird name (like 'ra_1'),
    # 4_cutouts.py needs to know it.
    print(df.columns.tolist())

    # Quick statistics (mean, min, max...) of a few columns
    print(df[["dr7objid", "nvote", "spiral", "elliptical", "uncertain"]].describe())

    # ---- 3. Classification of objects ----------------------------------------
    df = add_label(df)
    print("\nGalaxy Zoo classification (how many of each):")
    print(df["label"].value_counts())                         # counts
    print(df["label"].value_counts(normalize=True).round(3))  # fractions (0 to 1)

    # ---- 4. Strange objects --------------------------------------------------
    # 'type' tells if SDSS thinks the object is a galaxy (3) or a star (6).
    if "type" in df.columns:
        print("\nObject type (3 = galaxy, 6 = star):")
        print(df["type"].value_counts())

    # SDSS writes -9999 when a magnitude is missing. Count them per filter.
    for b in "ugriz":
        n_bad = (df[f"modelmag_{b}"] < -100).sum()
        print(f"Missing magnitude in filter {b}: {n_bad}")

    # ---- 5. Clean the sample ---------------------------------------------------
    print("\n--- Cleaning (signal-to-noise criterion) ---")
    clean = clean_catalog(df)
    print("\nFinal clean sample:")
    print(clean["label"].value_counts())

    # ---- 6. Save it ---------------------------------------------------------------
    clean.to_csv("clean_catalog.csv", index=False)
    print("Saved clean_catalog.csv")

    # Question from the slides: "Objects without any classification, do we use them?"
    # Answer: NOT for training. They have no trustworthy label, so the computer
    # would not know the "right answer" for them. (After training we could
    # use the model to GUESS their shape, though!)


if __name__ == "__main__":
    explore()
