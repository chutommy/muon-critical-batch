"""Tuned Muon learning rate $\\eta^\\ast$ vs batch size $B$."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style
from matplotlib.ticker import FormatStrFormatter

style.apply_style()
T = C.load_tables()
best = C.best_lr_per_Bmu(T.runs, T.val_loss)

fig, ax = plt.subplots(figsize=style.SIZE["landscape"])

anchor = best[best.batch_size == min(C.BATCH_SIZES)]["lr_muon"].mean()
Bg = np.array([min(C.BATCH_SIZES), max(C.BATCH_SIZES)], float)
ax.plot(Bg, anchor * Bg / Bg[0], color=style.GREY_DARK, **style.REF, label="$\\eta \\propto B$")

for mu in C.MUS:
    d = best[best.mu == mu].sort_values("batch_size")
    col = style.mu_color(mu)

    interior = d[~d.on_edge]
    edge = d[d.on_edge]

    ax.plot(d.batch_size, d.lr_muon, color=col, **style.PRIMARY)
    ax.plot(
        interior.batch_size,
        interior.lr_muon,
        "o",
        color=col,
        ms=5,
        mec="white",
        mew=0.6,
        ls="",
        zorder=3,
        label=f"$\\mu = {mu:g}$",
    )
    ax.plot(
        edge.batch_size,
        edge.lr_muon,
        "o",
        mfc="white",
        mec=col,
        mew=1.2,
        ms=5,
        ls="",
        zorder=3,
    )

style.batch_xaxis(ax, C.BATCH_SIZES)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.set_xlabel("batch size $B$ (tokens)")
ax.set_ylabel("tuned learning rate $\\eta^\\ast$")
ax.legend(title="momentum $\\mu$", loc="upper left")

fig.suptitle("Tuned learning rate $\\eta^\\ast$ vs batch size $B$")
fig.tight_layout()
style.save(fig, "lr_best_vs_B")
