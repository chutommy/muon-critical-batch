"""McCandlish hyperbola fit at $L = 3.60$."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style

style.apply_style()
L = 3.60
T = C.load_tables()

fig, axes = plt.subplots(
    2,
    3,
    figsize=style.SIZE["2row3col"],
    sharex=True,
    gridspec_kw=dict(height_ratios=[2.4, 1], hspace=0.08),
)

for j, mu in enumerate(C.MUS):
    f = C.fit_bstar(T.runs, T.val_loss, L, mu)
    col = style.mu_color(mu)
    B, S = f.B, f.S
    top, bot = axes[0, j], axes[1, j]

    leg_lo = f.bstar * 0.6
    Bgrid = np.logspace(np.log10(B.min() / 1.6), np.log10(B.max() * 1.6), 200)
    Shyp = f.s_min * (1 + f.bstar_hyp / Bgrid)
    fit_mask = Bgrid >= leg_lo

    top.axvspan(leg_lo, Bgrid.max(), color=col, alpha=0.10, lw=0)
    top.plot(Bgrid[fit_mask], Shyp[fit_mask], color=col, **style.SECONDARY, label="hyperbola fit")
    top.plot(Bgrid[~fit_mask], Shyp[~fit_mask], color=col, **style.EXTRAP, label="hyperbola, extrapolated")

    top.plot(B, S, "o", color=col, ms=5, mec="white", mew=0.6, ls="", zorder=5, label="observed")

    top.axvline(f.bstar, color=col, **style.VLINE)
    top.axvline(f.bstar_hyp, color=col, **style.VLINE_ALT)
    top.set_xscale("log")
    top.set_yscale("log")
    top.set_title(f"$\\mu = {mu:g}$")

    secax = top.secondary_xaxis("top")
    secax.set_xticks([f.bstar, f.bstar_hyp])
    secax.set_xticklabels([f"{f.bstar / 1e3:.0f}K", f"{f.bstar_hyp / 1e6:.1f}M"])
    secax.get_xticklabels()[0].set_color(style.GREY_DARK)
    secax.get_xticklabels()[1].set_color(col)

    Shyp_at = f.s_min * (1 + f.bstar_hyp / B)
    resid = (S - Shyp_at) / Shyp_at
    bot.axhline(0, color=style.GREY_MID, lw=0.8, alpha=0.7)
    bot.axvspan(leg_lo, Bgrid.max(), color=col, alpha=0.10, lw=0)
    bot.plot(B, resid, "o", color=col, ms=5, mec="white", mew=0.6, ls="", zorder=5)
    bot.plot(B, resid, color=col, **style.SECONDARY)
    bot.set_xscale("log")
    bot.set_ylim(-0.35, 0.9)

    if j == 0:
        top.set_ylabel("steps $S$ to reach loss $L$")
        bot.set_ylabel("$(S_{\\mathrm{obs}} - S_{\\mathrm{hyp}}) / S_{\\mathrm{hyp}}$")
    bot.set_xlabel("batch size $B$ (tokens)")

h, lab = axes[0, 0].get_legend_handles_labels()
elbow_line = plt.Line2D([], [], color=style.GREY_DARK, **style.VLINE, label="elbow $B^\\ast$")
hyp_line = plt.Line2D([], [], color=style.GREY_DARK, **style.VLINE_ALT, label="hyperbola $B^\\ast$")
style.outside_right_legend(fig, h + [elbow_line, hyp_line], lab + ["elbow $B^\\ast$", "hyperbola $B^\\ast$"], title="")

fig.suptitle("Right-leg McCandlish hyperbola fit at loss target $L = 3.60$")
fig.tight_layout()
style.save(fig, "hyperbola_fit_quality")
