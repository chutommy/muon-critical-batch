"""Validation loss vs learning rate $\\eta$."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style

style.apply_style()
T = C.load_tables()

la = C.loss_at_tokens(T.val_loss).merge(T.runs[["run_id", "batch_size", "mu", "lr_muon"]], on="run_id")

fig, axes = plt.subplots(
    2,
    3,
    figsize=style.SIZE["2x3"],
    sharex="col",
    sharey="row",
    gridspec_kw={"height_ratios": [1.0, 0.65]},
)

for row in axes:
    for ax, mu in zip(row, C.MUS):
        sub = la[la.mu == mu]
        for B in C.BATCH_SIZES:
            d = sub[sub.batch_size == B].sort_values("lr_muon")
            if len(d) < 2:
                continue
            col = style.b_color(B)
            ax.plot(
                d.lr_muon,
                d.val_loss_at_target,
                color=col,
                marker="o",
                ms=5,
                mec="white",
                mew=0.6,
                label=style.format_B(B),
                **style.PRIMARY,
            )
            m = d.loc[d.val_loss_at_target.idxmin()]
            ax.plot(
                [m.lr_muon],
                [m.val_loss_at_target],
                "o",
                mfc="none",
                mec=col,
                mew=1.3,
                ms=7,
                zorder=4,
            )
        ax.set_xscale("log")

for ax, mu in zip(axes[0], C.MUS):
    ax.set_title(f"$\\mu = {mu:g}$")
axes[0, 0].set_ylim(3.3, 4.7)

for ax in axes[1]:
    ax.set_xlabel("learning rate $\\eta$")
axes[1, 0].set_ylim(3.34, 3.49)

axes[0, 0].set_ylabel("validation loss")
axes[1, 0].set_ylabel("validation loss (zoomed in)")
h, l = axes[0, -1].get_legend_handles_labels()
style.outside_right_legend(fig, h, l, title="batch size $B$")

fig.suptitle("Validation loss vs learning rate $\\eta$")
fig.tight_layout()
style.save(fig, "loss_vs_lr")
