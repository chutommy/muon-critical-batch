"""Steps $S$ to reach loss $L$ vs batch size $B$."""

import common as C
import matplotlib.pyplot as plt
import style

L_TARGETS = [3.4, 3.6, 3.8, 4.0, 4.3, 4.5, 4.7]

style.apply_style()
T = C.load_tables()
lcol = style.l_colors(L_TARGETS)

fig, axes = plt.subplots(1, 3, figsize=style.SIZE["1x3"], sharey=True)

for ax, mu in zip(axes, C.MUS):
    for L in L_TARGETS:
        d = C.steps_to_L_tuned(T.runs, T.val_loss, L)
        d = d[d["mu"] == mu].sort_values("batch_size")
        if d.empty:
            continue
        ax.plot(
            d["batch_size"].to_numpy(float),
            d["step"].to_numpy(float),
            color=lcol[L],
            marker="o",
            ms=5,
            mec="white",
            mew=0.6,
            label=f"$L = {L:.2f}$",
            **style.PRIMARY,
        )

    style.batch_xaxis(ax, C.BATCH_SIZES)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("batch size $B$ (tokens)")
    if ax is axes[0]:
        ax.set_ylabel("steps $S$ to reach loss $L$")
    ax.set_title(f"$\\mu = {mu:g}$")

handles, labels = axes[-1].get_legend_handles_labels()
style.outside_right_legend(fig, handles, labels, title="loss target $L$")

fig.suptitle("Steps $S$ to reach loss $L$ vs batch size $B$")
fig.tight_layout()
style.save(fig, "S_vs_B_by_L")
