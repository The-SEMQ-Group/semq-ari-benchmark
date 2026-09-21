# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Figure 3: batch size changes codes on GPU, and the same-batch control says so.

Reads experiments/batch-invariance/results/summary_with_control.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.0, "ytick.major.width": 0.0,
    "pdf.fonttype": 42,
})

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "..", "..", "experiments", "batch-invariance",
                   "results", "summary_with_control.json")
DATA = json.load(open(SRC))

SHORT = {
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM-L6",
    "BAAI/bge-large-en-v1.5": "bge-large",
    "sentence-transformers/all-mpnet-base-v2": "mpnet-base",
    "intfloat/multilingual-e5-large": "e5-large",
    "nomic-ai/nomic-embed-text-v1.5": "nomic-embed",
    "Snowflake/snowflake-arctic-embed-l": "arctic-embed-l",
    "mixedbread-ai/mxbai-embed-large-v1": "mxbai-large",
}
ORDER = list(SHORT)
BATCHES = [1, 8, 128, 512]
CELLS = [("A", "fp32, TF32 off"), ("C", "fp32, TF32 on")]


def grid(cell):
    out = np.full((len(ORDER), len(BATCHES) + 1), np.nan)
    for r, model in enumerate(ORDER):
        key = f"{model.replace('/', '_')}__cell{cell}"
        if key not in DATA["cells"]:
            continue
        c = DATA["cells"][key]
        out[r, 0] = c["control_HER"]
        her = {x["batch"]: x["HER"] for x in c["comparisons"]}
        for j, b in enumerate(BATCHES):
            out[r, j + 1] = her.get(b, np.nan)
    return out


CMAP = LinearSegmentedColormap.from_list("her", ["#3a3a3a", "#bdbdbd", "#eeeeee"])

fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.78))
fig.subplots_adjust(left=0.155, right=0.995, top=0.815, bottom=0.20, wspace=0.09)
INK = "#1a1a1a"

for ax, (cell, title) in zip(axes, CELLS):
    g = grid(cell)
    ax.imshow(g, cmap=CMAP, vmin=0.5, vmax=1.0, aspect="auto")
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            v = g[r, c]
            if np.isnan(v):
                continue
            txt = "1" if v == 1.0 else f"{v:.3f}".lstrip("0")
            ax.text(c, r, txt, ha="center", va="center", fontsize=5.6,
                    color="white" if v < 0.70 else INK)
    ax.set_xticks(range(g.shape[1]))
    ax.set_xticklabels(["ctrl\n(b32)"] + [f"b{b}" for b in BATCHES])
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([SHORT[m] for m in ORDER] if cell == "A" else [])
    ax.set_title(title, fontsize=7, pad=3, loc="left", color=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, g.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, g.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", length=0)
    ax.axvline(0.5, color=INK, linewidth=0.8)

fig.savefig(os.path.join(HERE, "figure3.pdf"), format="pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote figure3.pdf")
