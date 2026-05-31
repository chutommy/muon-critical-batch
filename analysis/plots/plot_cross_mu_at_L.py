"""Steps $S$ and examples $E$ vs batch size $B$ at $L = 3.60$."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style

style.apply_style()
T = C.load_tables()
L = 3.6

fig, (axS, axE) = plt.subplots(1, 2, figsize=style.SIZE["1x2"])

for mu in C.MUS:
    fit = C.fit_bstar(T.runs, T.val_loss, L, mu)
    col = style.mu_color(mu)
    order = np.argsort(fit.B)
    B, S, E = fit.B[order], fit.S[order], fit.E[order]
    axS.plot(B, S, color=col, marker="o", ms=5, label=f"$\\mu = {mu:g}$", **style.PRIMARY)
    axE.plot(B, E, color=col, marker="o", ms=5, label=f"$\\mu = {mu:g}$", **style.PRIMARY)
    if len(B) >= 3:
        x, y = np.log2(B), np.log10(E)
        coef = np.polyfit(x, y, 2)
        xx = np.linspace(x.min(), x.max(), 200)
        axE.plot(2**xx, 10 ** np.polyval(coef, xx), color=col, **style.EXTRAP)
    if np.isfinite(fit.bstar):
        axE.plot([fit.bstar], [fit.e_min], "*", ms=14, color=col, mec="black", mew=0.5, zorder=5)
        axE.axvline(fit.bstar, color=col, **style.VLINE)

for ax in (axS, axE):
    style.batch_xaxis(ax, C.BATCH_SIZES)
    ax.set_yscale("log")
    ax.set_xlabel("batch size $B$ (tokens)")

axS.set_ylabel("steps $S$ to reach loss $L$")
axE.set_ylabel("examples $E$ to reach loss $L$")
axS.legend(title="momentum $\\mu$", loc="best")

fig.suptitle("Steps $S$ and examples $E$ to reach loss $L = 3.60$ vs batch size $B$")
fig.tight_layout()
style.save(fig, "cross_mu_at_L")
