"""
utils.py  --  SHARED HELPER FUNCTIONS
=====================================

What is this file?
    A small toolbox that the other scripts (2_exploration.py, 3_plot.py, ...)
    import so we do not copy-paste the same code everywhere.

What does it do?
    1. Reads the big CSV that 1_retrieval.py downloaded.
    2. Labels every galaxy as "spiral", "elliptical" or "uncertain".
    3. Cleans the sample using a SIGNAL-TO-NOISE (S/N) criterion.

Quick vocabulary (read this first!):
    - magnitude (mag): how bright an object is. SMALLER number = BRIGHTER.
    - magnitude error: how unsure we are about that brightness.
    - S/N (signal-to-noise): "how many times bigger is the signal than the noise".
        S/N = 1  -> the signal is as big as the noise (we basically see nothing).
        S/N = 10 -> a clean, trustworthy measurement.
    - extinction: the dust of our own Milky Way makes galaxies look dimmer and
      redder. We subtract it to see the "true" brightness.
    - color: the difference between two magnitudes, e.g. g - r. It tells us if a
      galaxy is blue (young stars, usually spirals) or red (old stars, usually ellipticals).
"""

import numpy as np
import pandas as pd

# Name of the CSV created by 1_retrieval.py
CSV = "/home/uri/estadistica/ESTADISTICA/BOOTCAMP/ZooSpecPhoto.csv"

# The five SDSS filters (colors of light the telescope measures):
# u = ultraviolet, g = green, r = red, i = infrared-ish, z = more infrared
BANDS = ["u", "g", "r", "i", "z"]

# ---------------------------------------------------------------------------
# SELECTION CRITERION (the important part!)
# ---------------------------------------------------------------------------
# Minimum signal-to-noise we ACCEPT in each filter.
#   g, r, i are the "good" filters -> we ask for S/N >= 10 (very reliable).
#   u and z are noisy filters (the camera is less sensitive there), so if we asked
#   for 10 we would throw away too many galaxies -> we ask for S/N >= 5 (acceptable).
# You can change these numbers if you want a stricter or a looser sample.
MIN_SN = {"u": 5, "g": 10, "r": 10, "i": 10, "z": 5}

# Magnitudes are a logarithmic scale, so S/N and magnitude error are related by:
#       S/N = 1.0857 / magnitude_error
# (1.0857 = 2.5 / ln(10)). Example: error 0.1 mag  ->  S/N ~ 10.9
MAG_TO_SN = 1.0857

# Faintest galaxy we keep in the "r" filter. 17.77 is the limit of the SDSS
# spectroscopic Main Galaxy Sample (fainter galaxies were not well classified).
MAG_R_LIMIT = 17.77


def load_catalog(csv=CSV):
    """
    Read the CSV file into a pandas table (a "DataFrame").
    We also turn ALL column names into lowercase, so we never have to worry
    about writing 'modelMag_r' vs 'modelmag_r'.
    """
    df = pd.read_csv(csv, low_memory=False)
    df.columns = df.columns.str.lower()
    return df


def add_label(df):
    """
    Add a new column called 'label' with the morphology (shape) of each galaxy.
    Galaxy Zoo volunteers voted, and the table already has simple 0/1 flags:
        spiral     == 1  -> clearly a spiral galaxy
        elliptical == 1  -> clearly an elliptical galaxy
        neither          -> 'uncertain' (people disagreed, we do not trust it)
    """
    df["label"] = np.where(df["spiral"] == 1, "spiral",
                  np.where(df["elliptical"] == 1, "elliptical", "uncertain"))
    return df


def signal_to_noise(mag_error):
    """
    Convert a magnitude error into a signal-to-noise ratio.
    Bigger result = better measurement.
    (We add a tiny number to avoid dividing by zero.)
    """
    return MAG_TO_SN / (mag_error + 1e-12)


def clean_catalog(df, min_sn=MIN_SN, mag_limit=MAG_R_LIMIT):
    """
    Throw away the galaxies we do NOT trust, and keep the good ones.

    Steps (each step prints how many galaxies are left):
        1. Keep only clear spirals and clear ellipticals (drop 'uncertain').
        2. Remove duplicated rows (same galaxy twice).
        3. Remove galaxies with broken magnitudes (SDSS writes -9999 when it has no value).
        4. Keep only galaxies with S/N >= our threshold in EVERY filter (see MIN_SN).
        5. Correct the magnitudes for dust in the Milky Way (extinction).
        6. Keep only galaxies brighter than mag_limit in the r filter.
        7. Compute colors and a 'concentration' number (see below).
    """
    df = add_label(df.copy())
    print(f"Start: {len(df)} galaxies")

    # --- Step 1: only clear classifications -------------------------------
    df = df[df["label"] != "uncertain"]
    print(f"1) Clear spiral/elliptical only:        {len(df)}")

    # --- Step 2: no duplicates ---------------------------------------------
    df = df.drop_duplicates(subset="dr7objid")
    print(f"2) After removing duplicates:           {len(df)}")

    # --- Step 3: magnitudes must be real numbers ----------------------------
    # Real magnitudes of galaxies are between ~10 and ~30. -9999 means "no data".
    good = np.ones(len(df), dtype=bool)          # start by accepting everyone...
    for b in BANDS:
        mag = df[f"modelmag_{b}"]
        err = df[f"modelmagerr_{b}"]
        good &= (mag > 0) & (mag < 40) & (err > 0)   # ...and reject the broken ones
    df = df[good].copy()
    print(f"3) Valid magnitudes in all 5 filters:   {len(df)}")

    # --- Step 4: SIGNAL-TO-NOISE CUT ---------------------------------------
    # For each filter compute S/N and keep only galaxies above the threshold.
    good = np.ones(len(df), dtype=bool)
    for b in BANDS:
        sn = signal_to_noise(df[f"modelmagerr_{b}"])
        df[f"sn_{b}"] = sn                       # keep S/N in the table, it can be handy
        good &= (sn >= min_sn[b])
    df = df[good].copy()
    print(f"4) S/N above threshold {min_sn}: {len(df)}")

    # --- Step 5: remove the Milky Way dust effect ---------------------------
    # New columns mag_u, mag_g, ... = corrected magnitudes. We use THESE from now on.
    for b in BANDS:
        df[f"mag_{b}"] = df[f"modelmag_{b}"] - df[f"extinction_{b}"]

    # --- Step 6: not too faint ---------------------------------------------
    df = df[df["mag_r"] < mag_limit].copy()
    print(f"5) Brighter than r < {mag_limit}:         {len(df)}")

    # --- Step 7: new useful columns -----------------------------------------
    # Colors = difference of two magnitudes (they do NOT depend on distance!).
    df["u_g"] = df["mag_u"] - df["mag_g"]
    df["g_r"] = df["mag_g"] - df["mag_r"]
    df["r_i"] = df["mag_r"] - df["mag_i"]
    df["i_z"] = df["mag_i"] - df["mag_z"]

    # Concentration = r90 / r50.
    #   r50 = radius that contains 50% of the light, r90 = radius with 90% of the light.
    #   Ellipticals are very "packed" in the center (big concentration),
    #   spirals have light spread out in a disk (smaller concentration).
    if {"petror50_r", "petror90_r"} <= set(df.columns):
        df = df[df["petror50_r"] > 0].copy()           # avoid dividing by zero
        df["concentration"] = df["petror90_r"] / df["petror50_r"]

    return df.reset_index(drop=True)


def pick(df, *names):
    """
    Return the FIRST column name from `names` that really exists in the table.
    Why? Both tables we joined have 'ra' and 'dec' columns, so SQL may have
    renamed one of them (e.g. 'ra_1'). This function finds whichever exists.
    ra and dec = the sky coordinates of the galaxy (like longitude and latitude).
    """
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"None of these columns exist: {names}")
