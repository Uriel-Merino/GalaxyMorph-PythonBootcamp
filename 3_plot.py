"""
3_plot.py  --  PART 1: A PUBLICATION-READY PLOT
===============================================

What is this file?
    Makes one nice figure that could go in a scientific paper.

What does the figure show?
    A "color-magnitude diagram":
        - X axis: brightness of the galaxy in the r filter (smaller = brighter)
        - Y axis: color g - r  (bigger = redder)
    Spirals are drawn in BLUE, ellipticals in RED.
    You will see that ellipticals are red and spirals are bluer: color helps
    tell the two apart, which is exactly what our ML model will learn later!
    On the top and right there are little histograms (counts) of each variable.

How to run:
    python 3_plot.py     (needs clean_catalog.csv from 2_exploration.py)
"""

import pandas as pd
import matplotlib.pyplot as plt

# Global style settings so the plot looks "paper-like":
# ticks pointing inwards, ticks on all four sides, serif font.
plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 14,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "font.family": "serif", "mathtext.fontset": "cm",
})


def make_plot(csv="clean_catalog.csv", out="color_magnitude"):
    # Read the clean data and split it into the two groups
    df = pd.read_csv(csv)
    ell = df[df["label"] == "elliptical"]
    spi = df[df["label"] == "spiral"]

    # ---- Layout: a big central panel + a small panel on top + a small panel on the right
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.12, right=0.97, bottom=0.1, top=0.97,
                          wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])                 # main panel
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax)    # top histogram (shares the X axis)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax)    # right histogram (shares the Y axis)

    xlim, ylim = (12, 17.8), (0.0, 1.5)            # visible range of the plot

    # ---- Draw each group
    for sub, cmap, color, name in [(spi, "Blues", "tab:blue", "Spiral"),
                                   (ell, "Reds", "tab:red", "Elliptical")]:
        # hexbin = "honeycomb" density map. With >100k points, plain dots would
        # just make a messy blob; hexagons count how many galaxies fall in each cell.
        # bins="log" so both crowded and empty regions are visible.
        ax.hexbin(sub["mag_r"], sub["g_r"], gridsize=60, extent=(*xlim, *ylim),
                  cmap=cmap, bins="log", mincnt=1, alpha=0.8)

        # Invisible point just to get a legend entry with the number of galaxies
        ax.plot([], [], "s", color=color, label=f"{name} (N={len(sub):,})")

        # Little histograms on the top and on the right
        ax_x.hist(sub["mag_r"], bins=60, range=xlim, color=color,
                  histtype="step", lw=1.5, density=True)
        ax_y.hist(sub["g_r"], bins=60, range=ylim, color=color,
                  histtype="step", lw=1.5, density=True, orientation="horizontal")

    # ---- Labels and cosmetics
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$r$ (model mag, extinction corrected)")
    ax.set_ylabel(r"$g - r$")
    ax.legend(loc="upper left", frameon=False)

    # Hide the numbers on the small histograms (they are only for the "shape")
    for a in (ax_x, ax_y):
        a.tick_params(labelbottom=False, labelleft=False)
    ax_x.set_yticks([])
    ax_y.set_xticks([])

    # ---- Save the figure in two formats
    fig.savefig(f"{out}.pdf")          # PDF = vector (never pixelated, best for papers)
    fig.savefig(f"{out}.png", dpi=300) # PNG = normal image, 300 dpi = high quality
    print(f"Saved {out}.pdf and {out}.png")


if __name__ == "__main__":
    make_plot()
