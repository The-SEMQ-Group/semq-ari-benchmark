import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman","DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "pdf.fonttype": 42,
})

# --- verified data -------------------------------------------------------
conds  = ["controls", "TF32 off", "TF32 on", "bf16", "int8"]
her    = [1.0000, 0.9992, 0.5267, 0.0000, 0.0000]
d_r10  = [0.0000, 0.0000, 0.0000, 0.0033, -0.0061]
lo     = [0.0000, 0.0000, 0.0000, 0.0000, -0.0278]
hi     = [0.0000, 0.0000, 0.0000, 0.0100,  0.0156]

depth  = [0, 1, 5, 10, 15]
curves = [(r"$\sigma=0$",       [0,0.000,0.000,0.000,0.000], "-",  "o"),
          (r"$\sigma=10^{-4}$", [0,0.000,0.007,0.007,0.012], "--", "s"),
          (r"$\sigma=10^{-3}$", [0,0.002,0.040,0.071,0.115], "-.", "^"),
          (r"$\sigma=3\!\times\!10^{-3}$", [0,0.028,0.147,0.258,0.392], ":", "D")]

INK, MID, PALE = "#1a1a1a", "#7a7a7a", "#c9c9c9"

fig = plt.figure(figsize=(5.5, 1.80))
gs  = GridSpec(2, 2, figure=fig, width_ratios=[1.05, 1.0], height_ratios=[1.0, 0.92],
               hspace=0.18, wspace=0.30, left=0.085, right=0.985, top=0.885, bottom=0.28)

# --- panel A: internal agreement collapses --------------------------------
axA = fig.add_subplot(gs[0, 0])
x = range(len(conds))
fills  = [PALE, PALE, MID, INK, INK]
hatches= ["", "", "//", "xx", "xx"]
for i,(v,f,h) in enumerate(zip(her, fills, hatches)):
    axA.bar(i, v, width=0.62, color=f, edgecolor=INK, linewidth=0.6, hatch=h, zorder=3)
    axA.text(i, v+0.06 if v>0.05 else 0.06, f"{v:.3f}".rstrip("0").rstrip(".") if v>0 else "0",
             ha="center", va="bottom", fontsize=6.1, color=INK, zorder=4)
axA.set_ylim(0, 1.30); axA.set_yticks([0, 0.5, 1.0])
axA.set_ylabel("HER", labelpad=2)
axA.set_xticks(list(x)); axA.set_xticklabels([])
axA.grid(axis="y", color=PALE, linewidth=0.4, zorder=0); axA.set_axisbelow(True)
for sp in ("top","right"): axA.spines[sp].set_visible(False)
axA.set_title("Internal representation", fontsize=7, pad=3, loc="left", color=INK)

# --- panel B: retrieval quality does not move -----------------------------
axB = fig.add_subplot(gs[1, 0])
axB.axhline(0, color=MID, linewidth=0.6, zorder=1)
for i,(v,l,h) in enumerate(zip(d_r10, lo, hi)):
    axB.plot([i,i], [l,h], color=INK, linewidth=0.9, zorder=2, solid_capstyle="butt")
    axB.plot(i, v, marker="o", markersize=3.4, color=INK, zorder=3)
axB.set_ylim(-0.036, 0.024); axB.set_yticks([-0.03, 0, 0.02])
axB.set_ylabel(r"$\Delta$R@10", labelpad=2)
axB.set_xticks(list(x)); axB.set_xticklabels(conds, rotation=0)
axB.grid(axis="y", color=PALE, linewidth=0.4, zorder=0); axB.set_axisbelow(True)

for sp in ("top","right"): axB.spines[sp].set_visible(False)

# --- panel C: sequential compounding --------------------------------------
axC = fig.add_subplot(gs[:, 1])
for lab, ys, ls, mk in curves:
    axC.plot(depth, ys, ls, marker=mk, markersize=3.0, linewidth=1.0,
             color=INK, markerfacecolor="white", markeredgewidth=0.7, zorder=3, label=lab)
axC.set_xlim(-0.4, 16.2); axC.set_ylim(-0.015, 0.46)
leg = axC.legend(loc="upper left", fontsize=5.9, frameon=False, handlelength=2.0,
                 labelspacing=0.22, borderpad=0.0, handletextpad=0.5)
axC.set_xticks([0,5,10,15]); axC.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
axC.set_xlabel("retrieval hop", labelpad=1.5)
axC.set_ylabel("trajectory divergence", labelpad=2)
axC.grid(color=PALE, linewidth=0.4, zorder=0); axC.set_axisbelow(True)
for sp in ("top","right"): axC.spines[sp].set_visible(False)
axC.set_title("Sequential compounding", fontsize=7, pad=3, loc="left", color=INK)

fig.text(0.085, 0.012,
         r"$\Delta$R@10 with 95% CI:  bf16 $+0.0033\ [0,\,0.0100]$"
         " " r"int8 $-0.0061\ [-0.0278,\,0.0156]$",
         fontsize=5.6, color=INK, ha="left", va="bottom")

fig.savefig("figure1.pdf", format="pdf")
print("wrote figure1.pdf")
