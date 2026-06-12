"""Critical batch size $B^\\ast$ vs loss target $L$."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style

L_TARGETS = np.arange(3.45, 4.35 + 1e-9, 0.05)
FIT_CAP = 2_097_152
MIN_PTS_SOLID = 2

style.apply_style()
T = C.load_tables()
fig, ax = plt.subplots(figsize=style.SIZE["landscape"])

for mu in C.MUS:
    col = style.mu_color(mu)
    Ls, bs, lo, hi, solid = [], [], [], [], []
    for L in L_TARGETS:
        f = C.fit_bstar(T.runs, T.val_loss, L, mu, fit_cap=FIT_CAP)
        if not np.isfinite(f.bstar):
            continue
        Ls.append(L)
        bs.append(f.bstar)
        lo.append(f.bstar_lo)
        hi.append(f.bstar_hi)
        solid.append(f.elbow_interior and f.n_points >= MIN_PTS_SOLID)
    Ls = np.array(Ls)
    bs = np.array(bs)
    lo = np.array(lo)
    hi = np.array(hi)
    solid = np.array(solid)

    ax.fill_between(Ls, lo, hi, color=col, alpha=0.18, lw=0)
    style.plot_with_sparse(ax, Ls, bs, ~solid, color=col, label=f"$\\mu = {mu:g}$")

ax.set_xlim(4.40, 3.40)
ax.set_xticks(L_TARGETS[::2])
ax.set_xlabel("loss target $L$")
ax.set_ylabel("critical batch size $B^\\ast$ (tokens)")
ax.set_yscale("log")
ax.legend(title="momentum $\\mu$", loc="upper left")

fig.suptitle("Critical batch size $B^\\ast$ vs loss target $L$")
fig.tight_layout()
style.save(fig, "bstar_vs_L")
