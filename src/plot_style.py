"""Shared matplotlib style: Chinese labels, colour-blind-safe palette, vector output.

Every figure is written twice: ``figures/<name>.pdf`` (vector, for the submission)
and ``figures/<name>.png`` (300 dpi, embedded in the report).  Colours are the
Okabe-Ito palette and every series additionally carries its own marker and line
style, so the figures stay readable in greyscale and for colour-blind readers.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Okabe-Ito: safe for deuteranopia / protanopia
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#7F7F7F",
}
SERIES_STYLE = {
    "fgsm": dict(color=PALETTE["grey"], marker="v", ls=":", label="FGSM (基线)"),
    "pgd": dict(color=PALETTE["blue"], marker="o", ls="-", label="PGD-20 (自研)"),
    "mi_fgsm": dict(color=PALETTE["orange"], marker="s", ls="--", label="MI-FGSM (自研)"),
    "apgd": dict(color=PALETTE["vermillion"], marker="^", ls="-.",
                 label="APGD-DLR / AutoAttack-lite (自研)"),
    "cw_l2": dict(color=PALETTE["purple"], marker="D", ls="-", label="C&W-L2 (自研)"),
}

FIG_DIR = "figures"


def setup(font_candidates=("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun")):
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [f for f in font_candidates if f in available]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": chosen + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return chosen


def save(fig, name: str, outdir: str = FIG_DIR):
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, name + ".pdf")
    png = os.path.join(outdir, name + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print("saved", pdf, "+", png)
    return pdf, png
