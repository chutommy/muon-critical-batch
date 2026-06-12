"""Gradient-covariance trace and squared gradient norm vs tokens."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style

style.apply_style()
T = C.load_tables()
SHOW_B = C.BATCH_SIZES

n = T.noise_scale.merge(T.runs[["run_id", "batch_size", "mu", "lr_muon"]], on="run_id")
best = C.best_lr_per_Bmu(T.runs, T.val_loss)
keep = set(best.run_id)
n = n[n.run_id.isin(keep)]
n = n[n["tokens"] <= 1_000_000_000]

fig, axes = plt.subplots(2, 3, figsize=style.SIZE["2x3"], sharex=True, sharey="row")
rows = [
    ("tr_sigma", "gradient-covariance trace $\\mathrm{tr}\\,\\Sigma$"),
    ("dot", "squared gradient norm $|G|^2$"),
]

for j, mu in enumerate(C.MUS):
    for i, (col_name, ylabel) in enumerate(rows):
        ax = axes[i, j]
        for B in SHOW_B:
            g = n[(n.mu == mu) & (n.batch_size == B)].sort_values("tokens")
            g = g[g[col_name] > 0]
            if len(g) < 3:
                continue
            ax.plot(
                g["tokens"],
                g[col_name],
                color=style.b_color(B),
                label=style.format_B(B),
                **style.PRIMARY,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        if i == 0:
            ax.set_title(f"$\\mu = {mu:g}$")
        if j == 0:
            ax.set_ylabel(ylabel)
        if i == 1:
            ax.set_xlabel("tokens")

h, l = axes[0, -1].get_legend_handles_labels()
style.outside_right_legend(fig, h, l, title="batch size $B$")
axes[1, 0].set_ylim(1e1, 1e7)

fig.suptitle("Gradient-covariance trace $\\mathrm{tr}\\,\\Sigma$ and squared gradient norm $|G|^2$ vs tokens")
fig.tight_layout()
style.save(fig, "noise_components")
