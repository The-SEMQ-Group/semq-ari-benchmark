# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Figure 2: the hosted decoding panel, and the same model through two doors.

Panel A reads experiments/arid-dry-run/RESULTS.md's canonical rows, which are
regenerated from results/analysis.json and results/time_pair*.json.
Panel B reads the prefix-horizon control in the same RESULTS.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "pdf.fonttype": 42,
})

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
DRY = os.path.join(ROOT, "experiments", "arid-dry-run", "results")


# The canonical cell for each (subject, condition) is named by its transcript.
# analysis.json also holds prefix smokes, the no-seed confounder arm and the
# second time-pair batch, none of which is the published row.
TRANSCRIPT = {
    ("gemini", "same", None): "gemini_same.jsonl.gz",
    ("gemini", "proc", None): "gemini_proc.jsonl.gz",
    ("gemini", "conc", 64): "gemini_conc_b64.jsonl.gz",
    ("mistral", "same", None): "mistral_same.jsonl.gz",
    ("mistral", "proc", None): "mistral_proc.jsonl.gz",
    ("mistral", "conc", 16): "mistral_conc_b16.jsonl.gz",
    ("openai", "same", None): "openai_same.jsonl.gz",
    ("openai", "proc", None): "openai_proc.jsonl.gz",
    ("openai", "conc", 64): "openai_conc_b64.jsonl.gz",
    ("together", "same", None): "together_same.jsonl.gz",
    ("together", "proc", None): "together_proc.jsonl.gz",
    ("together", "conc", 64): "together_conc_b64.jsonl.gz",
    ("salesforce", "same", None): "salesforce_same.jsonl.gz",
    ("salesforce", "proc", None): "salesforce_proc.jsonl.gz",
}
TIME_PAIR = {"openai": "time_pair.json",
             "together": "time_pair_together.json",
             "salesforce": "time_pair_salesforce.json"}


def load():
    analysis = json.load(open(os.path.join(DRY, "analysis.json")))
    by_transcript = {c["transcript"]: c for c in analysis["conditions"]
                     if c["scope"] == "full"}
    rows = {}
    for key, name in TRANSCRIPT.items():
        eg = by_transcript[name]["exact_generation"]
        rows[key] = (eg["mean"], eg["ci"])
    for subj, fn in TIME_PAIR.items():
        eg = json.load(open(os.path.join(DRY, fn)))["exact_generation"]
        rows[(subj, "time", None)] = (eg["mean"], eg["ci"])
    return rows


ROWS = load()

# endpoint -> (display name, conc burst, conditions to draw)
PANEL = [
    ("gemini",     "gemini-3.6-flash",        64,  ["same", "proc", "conc"]),
    ("mistral",    "mistral-medium-2604",     16,  ["same", "proc", "conc"]),
    ("openai",     "gpt-4o-mini (vendor)",    64,  ["same", "proc", "conc", "time"]),
    ("together",   "Llama-3.3-70B (rehost)",  64,  ["same", "proc", "conc", "time"]),
    ("salesforce", "gpt-4o-mini (platform)",  None, ["same", "proc", "time"]),
]

MARK = {"same": ("o", "white"), "proc": ("s", "white"),
        "conc": ("^", "#1a1a1a"), "time": ("D", "#1a1a1a")}
INK, MID, PALE = "#1a1a1a", "#7a7a7a", "#c9c9c9"

fig = plt.figure(figsize=(5.5, 1.60))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.42, 1.0], wspace=0.34,
              left=0.215, right=0.985, top=0.865, bottom=0.225)

# --- panel A: five hosted decoding endpoints ------------------------------
axA = fig.add_subplot(gs[0, 0])
labels, ticks = [], []
for row, (subj, name, burst, conds) in enumerate(PANEL):
    y0 = -row
    ticks.append(y0)
    labels.append(name)
    offs = [0.26, 0.09, -0.09, -0.26][:len(conds)]
    for cond, off in zip(conds, offs):
        key = (subj, cond, burst if cond == "conc" else None)
        if key not in ROWS:
            continue
        v, (lo, hi) = ROWS[key]
        axA.plot([lo, hi], [y0 + off] * 2, color=INK, linewidth=0.7, zorder=2,
                 solid_capstyle="butt")
        mk, fc = MARK[cond]
        axA.plot(v, y0 + off, marker=mk, markersize=3.1, color=INK,
                 markerfacecolor=fc, markeredgewidth=0.65, zorder=3,
                 label=cond if row == 2 else None)
axA.set_yticks(ticks)
axA.set_yticklabels(labels)
axA.set_ylim(-len(PANEL) + 0.42, 0.42)
axA.set_xlim(-0.02, 1.06)
axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axA.set_xlabel("byte-identical repeat pairs", labelpad=1.5)
axA.grid(axis="x", color=PALE, linewidth=0.4, zorder=0)
axA.set_axisbelow(True)
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)
axA.legend(loc="upper left", ncol=2, fontsize=5.8, frameon=False,
           handletextpad=0.25, columnspacing=0.8, labelspacing=0.2,
           borderpad=0.0, borderaxespad=0.25)
axA.set_title("Hosted decoding (ARI-D), 100 frozen prompts, temperature 0",
              fontsize=7, pad=3, loc="left", color=INK)

# --- panel B: one model, two doors ----------------------------------------
axB = fig.add_subplot(gs[0, 1])
x = [0, 1, 2, 3, 4]
series = [("direct, seed", [0.924, 0.813, 0.644, 0.494, 0.470], "-", "o", INK),
          ("direct, no seed", [0.928, 0.807, 0.612, 0.459, 0.441], "--", "s", MID),
          ("platform door", [0.471, 0.245, 0.130, 0.107, 0.107], "-.", "^", INK)]
for lab, ys, ls, mk, col in series:
    axB.plot(x, ys, ls, marker=mk, markersize=3.0, linewidth=1.0, color=col,
             markerfacecolor="white", markeredgewidth=0.7, zorder=3, label=lab)
axB.set_xticks(x)
axB.set_xticklabels(["64", "128", "256", "512", "all"])
axB.set_xlabel("prefix compared (bytes)", labelpad=1.5)
axB.set_ylabel("byte-identical prefixes", labelpad=2)
axB.set_ylim(0, 1.02)
axB.set_yticks([0, 0.5, 1.0])
axB.legend(loc="upper right", fontsize=5.8, frameon=False, handlelength=2.0,
           labelspacing=0.22, borderpad=0.0, handletextpad=0.5)
axB.grid(color=PALE, linewidth=0.4, zorder=0)
axB.set_axisbelow(True)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
axB.set_title("One model, two doors", fontsize=7, pad=3, loc="left", color=INK)

fig.savefig(os.path.join(os.path.dirname(__file__), "figure2.pdf"), format="pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote figure2.pdf")
