"""Noise scale $B_{\\mathrm{noise}}$ vs validation loss."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style

ROLL = 5
YLIM = (5e2, 2e6)


def geo_roll(s, w):
    return np.exp(np.log(s.clip(lower=1e-9)).rolling(w, center=True, min_periods=2).mean())


style.apply_style()
T = C.load_tables()
best = C.best_lr_per_Bmu(T.runs, T.val_loss)[["run_id", "batch_size", "mu"]]
n = C.bnoise_vs_loss(T.noise_scale, T.val_loss)
n = n.merge(best, on="run_id")

fig, axes = plt.subplots(1, 3, figsize=style.SIZE["1x3"], sharex=True, sharey=True)
for ax, mu in zip(axes, C.MUS):
    sub = n[(n.mu == mu) & np.isfinite(n.loss)]
    for b in C.BATCH_SIZES:
        g = sub[sub.batch_size == b].sort_values("loss", ascending=False)
        if len(g) < 3:
            continue
        ax.plot(
            g.loss,
            geo_roll(g.B_noise_tokens, ROLL),
            color=style.b_color(b),
            label=style.format_B(b),
            **style.PRIMARY,
        )
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.invert_xaxis()
    ax.set_xlim(9.3, 3.35)
    ax.set_title(f"$\\mu = {mu:g}$")
    ax.set_xlabel("validation loss")
axes[0].set_ylabel("noise scale $B_{\\mathrm{noise}}$ (tokens)")

h, l = axes[-1].get_legend_handles_labels()
style.outside_right_legend(fig, h, l, title="batch size $B$")
fig.suptitle("Noise scale $B_{\\mathrm{noise}}$ vs validation loss")
fig.tight_layout()
style.save(fig, "bnoise_trajectory")
