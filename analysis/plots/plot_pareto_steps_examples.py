"""Examples $E$ vs steps $S$ to reach loss $L$."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style

L_TARGETS = [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3]
FIT_CAP = 2_097_152

style.apply_style()
T = C.load_tables()
lcol = style.l_colors(L_TARGETS)
fig, axes = plt.subplots(1, 3, figsize=style.SIZE["1x3"], sharex=True, sharey=True)

for ax, mu in zip(axes, C.MUS):
    for L in L_TARGETS:
        d = C.steps_to_L_tuned(T.runs, T.val_loss, L)
        d = d[d.mu == mu].sort_values("batch_size")
        if len(d) < 2:
            continue
        c = lcol[L]
        ax.plot(
            d.step,
            d.E,
            "o",
            color=c,
            ms=5,
            mec="white",
            mew=0.6,
            label=f"$L = {L:.2f}$",
            **style.PRIMARY,
        )
        f = C.fit_bstar(T.runs, T.val_loss, L, mu, fit_cap=FIT_CAP)
        if np.isfinite(f.bstar) and f.elbow_interior:
            j = int(np.argmin(np.abs(d.batch_size.to_numpy() - f.bstar)))
            ax.plot(
                d.step.iloc[j],
                d.E.iloc[j],
                "*",
                color=c,
                ms=14,
                mec="black",
                mew=0.5,
                zorder=6,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"$\\mu = {mu:g}$")
    ax.set_xlabel("steps $S$ to reach loss $L$")
axes[0].set_ylabel("examples $E$ to reach loss $L$")
handles, labels = axes[-1].get_legend_handles_labels()
style.outside_right_legend(fig, handles, labels, title="loss target $L$")
fig.suptitle("Examples $E$ vs steps $S$ to reach loss $L$")
fig.tight_layout()
style.save(fig, "pareto_steps_examples")
