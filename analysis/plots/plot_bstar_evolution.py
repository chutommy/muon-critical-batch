"""Critical batch size $B^\\ast$ and noise scale $B_{\\mathrm{noise}}$ vs loss target $L$."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style

style.apply_style()
L_TARGETS = np.arange(3.45, 4.35 + 1e-9, 0.05)
MID_B = [131072, 262144, 524288]
WINDOW = 0.06
T = C.load_tables()

bstar = {mu: {"L": [], "v": [], "lo": [], "hi": []} for mu in C.MUS}
for L in L_TARGETS:
    for mu in C.MUS:
        f = C.fit_bstar(T.runs, T.val_loss, L, mu)
        if np.isfinite(f.bstar):
            bstar[mu]["L"].append(L)
            bstar[mu]["v"].append(f.bstar)
            bstar[mu]["lo"].append(f.bstar_lo)
            bstar[mu]["hi"].append(f.bstar_hi)

bnoise = {mu: {"L": [], "v": []} for mu in C.MUS}
for L in L_TARGETS:
    bn = C.bnoise_near_L(T.noise_scale, T.val_loss, L, window=WINDOW)
    bn = bn.merge(T.runs[["run_id", "mu", "batch_size"]], on="run_id").dropna(subset=["bnoise"])
    bn = bn[bn["batch_size"].isin(MID_B)]
    for mu in C.MUS:
        v = C._gmean(bn.loc[bn["mu"] == mu, "bnoise"])
        if np.isfinite(v):
            bnoise[mu]["L"].append(L)
            bnoise[mu]["v"].append(v)

fig, ax = plt.subplots(figsize=style.SIZE["landscape"])
for mu in C.MUS:
    col = style.mu_color(mu)
    b = bstar[mu]
    ax.fill_between(b["L"], b["lo"], b["hi"], color=col, alpha=0.18, lw=0)
    ax.plot(
        b["L"],
        b["v"],
        "o",
        color=col,
        ms=5,
        mec="white",
        mew=0.6,
        zorder=4,
        label=f"$\\mu = {mu:g}$",
        **style.PRIMARY,
    )
    n = bnoise[mu]
    ax.plot(
        n["L"],
        n["v"],
        color=col,
        ms=5,
        mfc="white",
        mec=col,
        mew=1.2,
        marker="s",
        zorder=3,
        **style.EXTRAP,
    )

ax.set_yscale("log")
ax.set_yticks([5e4, 1e5, 3e5])
ax.set_yticklabels(["$5 \\times 10^{4}$", "$10^{5}$", "$3 \\times 10^{5}$"])
ax.set_yticks([], minor=True)
ax.set_xlabel("loss target $L$")
ax.set_ylabel("tokens")
ax.set_xlim(4.38, 3.4)
ax.set_xticks(L_TARGETS[::2])

leg1 = ax.legend(loc="upper left", title="momentum $\\mu$")
ax.add_artist(leg1)
solid = plt.Line2D([], [], color=style.GREY_DARK, marker="o", **style.PRIMARY, label="$B^\\ast$")
dash = plt.Line2D([], [], color=style.GREY_DARK, marker="s", mfc="white", mew=1.2, **style.EXTRAP, label="$B_{\\mathrm{noise}}$")
ax.legend(handles=[solid, dash], loc="lower right", handlelength=3.5)

fig.suptitle("Critical batch size $B^\\ast$ and noise scale $B_{\\mathrm{noise}}$ vs loss target $L$")
fig.tight_layout()
style.save(fig, "bstar_evolution")
