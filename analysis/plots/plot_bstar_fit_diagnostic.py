"""Parabola elbow vs McCandlish hyperbola fit diagnostic."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style

L_TARGETS = [3.55, 3.80, 4.05, 4.30]
FIT_CAP = 2_097_152

style.apply_style()
T = C.load_tables()

nrows = len(L_TARGETS)
fig, axes = plt.subplots(nrows, 3, figsize=style.SIZE["4x3"], sharex=True, sharey="row")

for i, L in enumerate(L_TARGETS):
    for j, mu in enumerate(C.MUS):
        ax = axes[i, j]
        col = style.mu_color(mu)
        f = C.fit_bstar(T.runs, T.val_loss, L, mu, fit_cap=FIT_CAP)

        d = C.steps_to_L_tuned(T.runs, T.val_loss, L)
        d = d[d["mu"] == mu].sort_values("batch_size")
        B_all = d["batch_size"].to_numpy(float)
        E_all = d["E"].to_numpy(float)
        in_fit = B_all <= FIT_CAP

        ax.plot(
            B_all[in_fit],
            E_all[in_fit],
            "o",
            color=col,
            ms=5,
            mec="white",
            mew=0.6,
            ls="",
            zorder=3,
        )
        ax.plot(
            B_all[~in_fit],
            E_all[~in_fit],
            "o",
            color=col,
            ms=5,
            mfc="white",
            mec=col,
            mew=1.2,
            ls="",
            zorder=3,
        )

        if f.n_points >= 3:
            xd = np.log2(f.B)
            yd = np.log10(f.E)
            coef = np.polyfit(xd, yd, 2)
            xs = np.linspace(xd.min(), xd.max(), 200)
            ax.plot(
                2**xs,
                10 ** np.polyval(coef, xs),
                color=col,
                **style.SECONDARY,
            )

        if np.isfinite(f.bstar):
            ax.axvspan(f.bstar_lo, f.bstar_hi, color=col, alpha=0.10, lw=0)
            ax.axvline(f.bstar, color=col, **style.VLINE)

        if f.hyp_ok and np.isfinite(f.bstar_hyp):
            B_grid = np.logspace(np.log10(B_all.min()), np.log10(B_all.max()), 200)
            ax.plot(
                B_grid,
                f.s_min * (B_grid + f.bstar_hyp),
                color=col,
                **style.EXTRAP,
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        if i == 0:
            ax.set_title(f"$\\mu = {mu:g}$")
        if j == 0:
            ax.set_ylabel(f"$L = {L:.2f}$")
        if i == nrows - 1:
            style.batch_xaxis(ax, C.BATCH_SIZES)

fig.supylabel("examples $E$ to reach $L$")
fig.supxlabel("batch size $B$ (tokens)")
fig.suptitle("Parabola and McCandlish hyperbola fits at fixed $L$ and $\\mu$")
fig.tight_layout()
style.save(fig, "bstar_fit_diagnostic")
