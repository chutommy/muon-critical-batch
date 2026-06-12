"""Validation and train loss vs steps, tokens, and wall-clock."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style
from matplotlib.ticker import FormatStrFormatter

style.apply_style()
T = C.load_tables()

best_all = C.best_lr_per_Bmu(T.runs, T.val_loss)

fig, axes = plt.subplots(len(C.MUS), 3, figsize=style.SIZE["3x3"], sharex="col", sharey=True)

for i, mu in enumerate(C.MUS):
    best = best_all[best_all.mu == mu]
    ids = {int(r.batch_size): r.run_id for _, r in best.iterrows()}
    row = axes[i]

    for B in C.BATCH_SIZES:
        rid = ids.get(B)
        if rid is None:
            continue
        col = style.b_color(B)

        v = T.val_loss[T.val_loss.run_id == rid].sort_values("step")
        v = v[(v.tokens > 0) & (v.val_loss > 0) & (v.step > 0) & (v.training_time_s > 0)]
        v = v[v.tokens <= C.TOKEN_CAP]
        if len(v) < 2:
            continue
        t0 = float(v["training_time_s"].iloc[0])
        v = v.assign(t_aligned=v["training_time_s"].to_numpy(float) - t0)

        g = T.train_step[T.train_step.run_id == rid].sort_values("step")
        g = g[(g.tokens > 0) & (g.train_loss > 0) & (g.step > 0)]
        g = g[g.tokens <= C.TOKEN_CAP]
        if len(g) >= 2:
            t_interp = np.interp(
                g["step"].to_numpy(float),
                v["step"].to_numpy(float),
                v["t_aligned"].to_numpy(float),
            )
            g = g.assign(t_interp=t_interp)
            for ax, xc in zip(row, ["step", "tokens", "t_interp"]):
                ax.plot(
                    g[xc].to_numpy(float),
                    g["train_loss"].to_numpy(float),
                    color=col,
                    **style.UNDERLAY,
                )

        for ax, xc in zip(row, ["step", "tokens", "t_aligned"]):
            ax.plot(
                v[xc].to_numpy(float),
                v["val_loss"].to_numpy(float),
                color=col,
                label=style.format_B(B),
                **style.PRIMARY,
            )

    axes[i, 0].set_ylabel(f"$\\mu = {mu:g}$\nloss")

for ax in axes.flat:
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%g"))
for j in (0, 1):
    axes[-1, j].set_xscale("log")
for j, xl in enumerate(["steps", "tokens", "wall-clock seconds"]):
    axes[-1, j].set_xlabel(xl)
axes[0, 0].set_ylim(top=4.6)
axes[0, 2].set_xlim(right=400)

handles, labels = axes[0, 0].get_legend_handles_labels()
style.outside_right_legend(fig, handles, labels, title="batch size $B$")

fig.suptitle("Validation and train loss vs steps, tokens, and wall-clock")
fig.tight_layout()
style.save(fig, "loss_curves_by_B")
