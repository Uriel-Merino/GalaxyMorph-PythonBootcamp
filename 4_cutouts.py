"""
4_cutouts.py  --  PART 2: DOWNLOAD SMALL IMAGES ("STAMPS") OF GALAXIES
======================================================================

What is this file?
    Our ML model needs to SEE the galaxies. SDSS has a web service that, given
    the sky position (ra, dec) of a galaxy, sends back a small picture of it.
    That small picture is called a "cutout" or "stamp".

What does it do?
    1. Picks a balanced sample: the same number of spirals and ellipticals.
    2. Downloads one stamp per galaxy (many at the same time = "in parallel").
    3. Saves ALL stamps inside ONE single file (stamps.npy).

Why those choices? (the HINTS from the slides)
    - How many images?  5000 spirals + 5000 ellipticals = 10,000.
        "Balanced" means 50/50, so the model does not learn to just always say
        "spiral" because spirals are more common.
    - How many pixels?  64 x 64. Big enough to see the shape, small enough to be fast.
    - Where to store?   ONE .npy file. 10,000 separate JPEGs in a folder is slow and messy.
    - Parallel download: 16 workers at once -> ~16x faster than one by one.
    - Safe against crashes: progress is saved, so if it stops you just run it again.

Files it creates:
    sample.csv        -> the list of chosen galaxies (same order as the stamps!)
    stamps.npy        -> the images, shape (N, 64, 64, 3)  [N galaxies, 64x64 pixels, 3 colors RGB]
    stamps_done.npy   -> for each galaxy: True if its image was downloaded OK

How to run:
    python 4_cutouts.py     (needs clean_catalog.csv)
"""

import io
import os
import numpy as np
import pandas as pd
import requests                      # to talk to the SDSS web service
from PIL import Image                # to open JPEG images
from concurrent.futures import ThreadPoolExecutor, as_completed   # parallel downloads

from utils import pick

# ---- Settings you can change ------------------------------------------------
URL = "https://skyserver.sdss.org/dr19/SkyServerWS/ImgCutout/getjpeg"  # SDSS cutout service
NPIX = 64            # picture size in pixels (64 x 64)
SCALE = 0.4          # arcseconds per pixel. 64 * 0.4 = ~25 arcsec wide picture.
                     # Increase it if the galaxies look cut off.
N_PER_CLASS = 5000   # how many spirals and how many ellipticals
WORKERS = 16         # how many downloads happen at the same time
SEED = 42            # fixed "random seed" -> same random choice every time (reproducible)


def build_sample(clean_csv="clean_catalog.csv", out="sample.csv"):
    """
    Choose N_PER_CLASS random spirals and N_PER_CLASS random ellipticals
    from the clean catalog, and save the list in sample.csv.
    """
    df = pd.read_csv(clean_csv)

    # If one class has fewer galaxies than we want, use that smaller number
    n = min(N_PER_CLASS, df["label"].value_counts().min())

    # For each class (spiral / elliptical) take n random galaxies,
    # glue them together, then shuffle so the classes are mixed.
    # (We loop over the classes by hand instead of using groupby.apply, because
    #  newer pandas versions silently DROP the 'label' column in that case.)
    parts = [g.sample(n, random_state=SEED) for _, g in df.groupby("label")]
    sample = (pd.concat(parts)
                .sample(frac=1, random_state=SEED)
                .reset_index(drop=True))

    # Find the real names of the coordinate columns and rename them 'ra_' / 'dec_'
    ra = pick(sample, "ra", "ra_1", "ra1")
    dec = pick(sample, "dec", "dec_1", "dec1")
    sample = sample.rename(columns={ra: "ra_", dec: "dec_"})

    sample.to_csv(out, index=False)
    print(f"Sample: {len(sample)} galaxies")
    print(sample["label"].value_counts())
    return sample


def fetch(ra, dec, retries=3):
    """
    Download ONE stamp from SDSS.
    Returns a numpy array of shape (64, 64, 3), or None if it failed 3 times
    (internet hiccups happen, so we try up to 3 times).
    """
    params = dict(ra=ra, dec=dec, scale=SCALE, width=NPIX, height=NPIX)
    for _ in range(retries):
        try:
            r = requests.get(URL, params=params, timeout=30)   # ask SDSS for the picture
            r.raise_for_status()                               # error if the server complained
            img = Image.open(io.BytesIO(r.content)).convert("RGB")  # bytes -> picture
            return np.asarray(img)                             # picture -> numbers
        except Exception:
            continue                                           # failed? try again
    return None


def download(sample, stamps_file="stamps.npy", done_file="stamps_done.npy"):
    """
    Download all the stamps in parallel and store them in stamps_file.
    If you run it again, it skips what is already downloaded.
    """
    N = len(sample)

    # Big empty "box" on the hard disk (memmap) that will hold all the pictures.
    # If the file already exists we re-open it ("r+") instead of erasing it ("w+").
    mode = "r+" if os.path.exists(stamps_file) else "w+"
    stamps = np.lib.format.open_memmap(stamps_file, mode=mode, dtype=np.uint8,
                                       shape=(N, NPIX, NPIX, 3))

    # 'done' remembers which galaxies already have their picture
    done = np.load(done_file) if os.path.exists(done_file) else np.zeros(N, dtype=bool)

    todo = np.where(~done)[0]          # indices of galaxies still missing
    print(f"To download: {len(todo)} / {N}")

    ra, dec = sample["ra_"].values, sample["dec_"].values

    # Launch many downloads at the same time
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(fetch, ra[i], dec[i]): i for i in todo}

        # as_completed gives us each result as soon as it is ready
        for k, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]               # which galaxy is this?
            img = fut.result()
            if img is not None and img.shape == (NPIX, NPIX, 3):
                stamps[i] = img            # put the picture in its slot
                done[i] = True             # remember it worked

            # Every 500 pictures: save progress (so a crash does not lose everything)
            if k % 500 == 0:
                stamps.flush()
                np.save(done_file, done)
                print(f"{k}/{len(todo)}")

    stamps.flush()
    np.save(done_file, done)
    print(f"Downloaded OK: {done.sum()}/{N}  (failed: {(~done).sum()} -> just run again)")


if __name__ == "__main__":
    # If a VALID sample.csv exists reuse it (IMPORTANT: it must keep the same order as stamps.npy).
    # "Valid" = it has the 'label' and 'ra_' columns. Otherwise it is rebuilt.
    sample = None
    if os.path.exists("sample.csv"):
        sample = pd.read_csv("sample.csv")
        if not {"label", "ra_", "dec_"} <= set(sample.columns):
            print("sample.csv is old/broken -> rebuilding it")
            sample = None
    if sample is None:
        sample = build_sample()
    download(sample)
