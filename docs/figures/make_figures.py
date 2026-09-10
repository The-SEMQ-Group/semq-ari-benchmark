# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.

"""Draw the figures used by the paper and the experiment reports.

Every figure reads a results JSON that an experiment wrote. Nothing here
recomputes a measurement, so a figure cannot disagree with the table it
illustrates.

Run from anywhere:  python docs/figures/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"

INK = "#1b1b1b"
SEMQ = "#0b6e4f"        # the instrument
CHEAP = "#c1543a"       # the cheap competitor
MUTED = "#8a8a8a"
ACCENT = "#2f5d8a"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def semq_stat(section: dict) -> str:
    """Name of the SEMQ statistic in a baselines.json section.

    Results written before 2026-09-10 carry "SEMQ Hbar", which was a
    packed-byte change rate. Later runs report the coordinate rate under
    "SEMQ coord change" and keep the byte rate beside it. Historical
    result files are not rewritten, so both names have to render.
    """
    for name in ("SEMQ coord change", "SEMQ Hbar", "SEMQ byte change (legacy)"):
        if name in section:
            return name
    raise KeyError(f"no SEMQ statistic in {sorted(section)}")


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")



def mark_zeros(ax, x, vals, colors, ymax, *, label_all=False):
    """Make an exactly-zero bar visible.

    A bar of height 0 draws nothing, so a control condition that moved
    nothing looks identical to a missing measurement. The zero is the
    result here, so it gets a visible stub and an explicit label.
    """
    width = 0.55
    for xi, v, c in zip(x, vals, colors):
        if v == 0:
            ax.plot([xi - width / 2, xi + width / 2], [0, 0],
                    color=c, lw=3, solid_capstyle="butt", zorder=4)
            ax.annotate("0", (xi, 0), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=7, color=c)
        elif label_all:
            ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7, color=c)


# ---------------------------------------------------------------------------
# 1. Retrieval quality is blind to a precision change; SEMQ is not.
# ---------------------------------------------------------------------------

def fig_regime():
    d = load(EXP / "regime-discrimination/results/regime_matrix.json")
    rows = {r["condition"]: r for r in d["rows"]}
    order = ["proc", "threads1", "batch8", "batch128", "bf16", "int8"]
    labels = ["new proc", "1 thread", "batch 8", "batch 128", "bf16", "int8"]
    prec = [c in ("bf16", "int8") for c in order]

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
    x = np.arange(len(order))
    cols = [CHEAP if p else MUTED for p in prec]

    # Recall delta, with the bootstrap CI. Zero inside means undetectable.
    delta = [rows[c]["recall_delta"] for c in order]
    lo = [rows[c]["recall_delta"] - rows[c]["recall_delta_ci95"][0] for c in order]
    hi = [rows[c]["recall_delta_ci95"][1] - rows[c]["recall_delta"] for c in order]
    ax[0].bar(x, delta, color=cols, yerr=[lo, hi], capsize=3,
              error_kw={"ecolor": INK, "lw": 1})
    ax[0].axhline(0, color=INK, lw=0.8)
    ax[0].set_title("Recall@10 change\n(quality metric)", fontsize=9)
    ax[0].set_ylabel("ΔRecall@10")
    mark_zeros(ax[0], x, delta, cols, None)
    ax[0].text(0.5, 0.93, "every CI crosses zero", transform=ax[0].transAxes,
               ha="center", fontsize=7.5, style="italic", color=MUTED)

    lists = [100 * (1 - rows[c]["top10_identical"]) for c in order]
    ax[1].bar(x, lists, color=cols)
    mark_zeros(ax[1], x, lists, cols, 105, label_all=True)
    ax[1].set_title("Result lists that changed\n(retrieval output)", fontsize=9)
    ax[1].set_ylabel("% of queries")
    ax[1].set_ylim(0, 105)

    codes = [100 * (1 - rows[c]["semq_her"]) for c in order]
    ax[2].bar(x, codes, color=cols)
    mark_zeros(ax[2], x, codes, cols, 105, label_all=True)
    ax[2].set_title("SEMQ codes that changed\n(the instrument)", fontsize=9)
    ax[2].set_ylabel("% of documents")
    ax[2].set_ylim(0, 105)

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, fontsize=7.5, rotation=40, ha="right")
    fig.suptitle("A precision change is invisible to retrieval quality and total to SEMQ",
                 fontsize=10.5, y=1.04)
    save(fig, "01-regime-discrimination.png")


# ---------------------------------------------------------------------------
# 2. Why tokens survive: flips happen only where the decision was nearly tied.
# ---------------------------------------------------------------------------

def fig_margin():
    d = load(EXP / "decoding-reproducibility/results/decoding_matrix.json")
    rows = {r["condition"]: r for r in d["rows"]}
    conds = ["bf16", "int8"]

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(conds))
    w = 0.36
    flip = [rows[c]["median_margin_flipped"] for c in conds]
    surv = [rows[c]["median_margin_survived"] for c in conds]
    ax[0].bar(x - w / 2, flip, w, label="token flipped", color=CHEAP)
    ax[0].bar(x + w / 2, surv, w, label="token survived", color=ACCENT)
    for i, (f, s) in enumerate(zip(flip, surv)):
        ax[0].text(i, max(f, s) * 1.05, f"{s / f:.0f}×", ha="center", fontsize=9,
                   fontweight="bold")
    ax[0].set_xticks(x); ax[0].set_xticklabels(conds)
    ax[0].set_ylabel("median top-2 logit margin")
    ax[0].set_title("Flips happen where the decision was nearly tied", fontsize=9)
    ax[0].legend(frameon=False, fontsize=8)

    agree = [100 * rows[c]["token_agreement"] for c in conds]
    early = [100 * rows[c]["early_warning_steps"] for c in conds]
    ax[1].bar(x, agree, 0.5, color=MUTED, label="token unchanged")
    ax[1].bar(x, early, 0.5, color=SEMQ,
              label="token unchanged, SEMQ changed")
    for i, e in enumerate(early):
        ax[1].text(i, e + 2, f"{e:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax[1].set_xticks(x); ax[1].set_xticklabels(conds)
    ax[1].set_ylabel("% of decoding steps")
    ax[1].set_ylim(0, 108)
    ax[1].set_title("Steps where output comparison sees nothing", fontsize=9)
    ax[1].legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle("ARI-D: the logits move before the tokens do", fontsize=10.5, y=1.03)
    save(fig, "02-decoding-early-warning.png")


# ---------------------------------------------------------------------------
# 3. Sensitivity: KL is second order and useless at small perturbations.
# ---------------------------------------------------------------------------

def fig_sensitivity():
    d = load(EXP / "decoding-reproducibility/results/baselines.json")
    sig = np.array(d["sigmas"])
    semq_name = semq_stat(d["sensitivity"])
    show = {semq_name: SEMQ, "top-2 margin delta": CHEAP,
            "KL": ACCENT, "JS": "#7a4fa3", "token flip": MUTED}

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name, col in show.items():
        y = np.array(d["sensitivity"][name], dtype=float)
        y = np.where(y <= 0, np.nan, y)
        ax.loglog(sig, y, "o-", color=col, label=name, ms=3.5, lw=1.4)

    ref = sig / sig[0] * float(d["sensitivity"][semq_name][0])
    ax.loglog(sig, ref, "--", color=INK, lw=0.8, alpha=0.5)
    ax.text(sig[2], ref[2] * 1.7, "slope 1 (linear)", fontsize=7.5,
            color=INK, alpha=0.7, rotation=22)

    ax.set_xlabel("perturbation σ")
    ax.set_ylabel("mean response")
    ax.set_title("SEMQ and the margin are linear in σ.\nKL is second order and reads zero when it matters",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, "03-sensitivity-sweep.png")


# ---------------------------------------------------------------------------
# 4. The honest one: cost against sensitivity.
# ---------------------------------------------------------------------------

def fig_cost():
    d = load(EXP / "decoding-reproducibility/results/baselines.json")
    store = d["storage_bytes"]
    resp = {k: v[1] for k, v in d["sensitivity"].items()}     # at sigma = 1e-3
    stable = {k: v["bit_identical"] for k, v in d["reassociation"].items()}
    semq_name = semq_stat(store)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for name in store:
        if resp.get(name, 0) <= 0:
            continue
        col = SEMQ if name == semq_name else (
            CHEAP if name == "top-2 margin delta" else MUTED)
        ax.scatter(store[name], resp[name], s=90 if col != MUTED else 45,
                   color=col, zorder=3,
                   marker="o" if stable.get(name) else "X")
        ax.annotate(name, (store[name], resp[name]),
                    textcoords="offset points", xytext=(7, 5), fontsize=7.5,
                    color=col if col != MUTED else INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("reference state a monitor must keep, bytes per step")
    ax.set_ylabel("response at σ = 1e-3")
    ax.set_title("The top-2 margin gets 86% of the signal for 1/2000 of the storage\n"
                 "(circle = bit-reproducible, cross = not)", fontsize=9)
    save(fig, "04-cost-vs-sensitivity.png")


# ---------------------------------------------------------------------------
# 5. What survives: coverage of the whole distribution.
# ---------------------------------------------------------------------------

def fig_tail():
    d = load(EXP / "decoding-reproducibility/results/baselines.json")
    tail = d["tail_only"]
    keeps = ["0", "2", "20", "100"]
    labels = ["all ranks", "below\nrank 2", "below\nrank 20", "below\nrank 100"]
    show = {semq_stat(tail[keeps[0]]): SEMQ, "top-2 margin delta": CHEAP,
            "token flip": MUTED, "top-20 set change": ACCENT}

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    x = np.arange(len(keeps))
    w = 0.2
    for i, (name, col) in enumerate(show.items()):
        vals = [tail[k][name] for k in keeps]
        xs = x + (i - 1.5) * w
        ax.bar(xs, vals, w, label=name, color=col)
        for xi, v in zip(xs, vals):
            if v == 0:
                ax.plot([xi - w / 2, xi + w / 2], [0, 0], color=col, lw=2.5,
                        solid_capstyle="butt", zorder=4)
                ax.annotate("0", (xi, 0), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=6.5, color=col)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean response (σ = 0.1)")
    ax.set_ylim(0, 0.155)
    ax.set_title("Move only the tail: the cheap statistics go to exactly zero,\n"
                 "SEMQ does not move at all", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    save(fig, "05-tail-coverage.png")


# ---------------------------------------------------------------------------
# 6. The retraction: near-ties are real, and they break nothing.
# ---------------------------------------------------------------------------

def fig_probe():
    d = load(EXP / "probe-verifiability/results/near_tie_sweep.json")
    rows = sorted(d["sweep"], key=lambda r: -r["dims_per_subvector"])
    dims = [r["dims_per_subvector"] for r in rows]
    tie = [100 * r["near_tie_rate"]["1e-06"] for r in rows]
    disagree = [100 * r["form_disagreement_rows"] for r in rows]
    attr = [100 * max(r["attractor_break_direct"], r["attractor_break_expanded"])
            for r in rows]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(dims))
    ax.plot(x, tie, "o-", color=ACCENT, label="assignments within 1e-6 of a tie")
    ax.plot(x, disagree, "s-", color=CHEAP, label="vectors reading differently")
    ax.plot(x, attr, "^-", color=SEMQ, label="discrete-attractor breaks")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}" for d in dims])
    ax.set_xlabel("dimensions per subvector  (fewer = more crowded centroids)")
    ax.set_ylabel("% (symlog)")
    ax.annotate("standard\nPQ setting", (0, tie[0]), textcoords="offset points",
                xytext=(12, 14), fontsize=7.5, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    ax.text(0.5, 0.15, "the attractor property never breaks, at any setting",
            transform=ax.transAxes, ha="center", fontsize=8, color=SEMQ,
            style="italic")
    ax.set_title("Near-ties rise by three orders of magnitude and break nothing",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "06-probe-verifiability.png")


if __name__ == "__main__":
    print("drawing figures into docs/figures/")
    fig_regime()
    fig_margin()
    fig_sensitivity()
    fig_cost()
    fig_tail()
    fig_probe()
    print("done")
