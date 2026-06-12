"""Critical batch size $B^\\ast$ vs noise scale $B_{\\mathrm{noise}}$."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style

L_TARGETS = [3.45, 3.50, 3.55, 3.60, 3.65, 3.70, 3.75, 3.80]
WINDOW = 0.03
FIT_CAP = 2_097_152

style.apply_style()
T = C.load_tables()
fig, ax = plt.subplots(figsize=style.SIZE["square"])

pts = {mu: [] for mu in C.MUS}
for L in L_TARGETS:
    bn = C.bnoise_near_L(T.noise_scale, T.val_loss, L, window=WINDOW)
    bn = bn.merge(T.runs[["run_id", "mu"]], on="run_id")
    for mu in C.MUS:
        f = C.fit_bstar(T.runs, T.val_loss, L, mu, fit_cap=FIT_CAP)
        sub = bn[bn.mu == mu]["bnoise"].dropna()
        if np.isfinite(f.bstar) and f.elbow_interior and len(sub):
            pts[mu].append((C._gmean(sub), f.bstar, L))

allv = [v for mu in C.MUS for (a, b, _) in pts[mu] for v in (a, b)]
lo, hi = min(allv) * 0.6, max(allv) * 1.6
ax.plot([lo, hi], [lo, hi], color=style.GREY_DARK, **style.REF)
ax.fill_between(
    [lo, hi],
    [lo / 2, hi / 2],
    [lo * 2, hi * 2],
    color=style.GREY_LIGHT,
    alpha=0.5,
    lw=0,
)
ax.text(hi, hi, "  $y = x$", color=style.GREY_DARK, va="center", ha="left")

for mu in C.MUS:
    if not pts[mu]:
        continue
    xb, yb, Ls = zip(*pts[mu])
    order = np.argsort(Ls)[::-1]
    xb = np.array(xb)[order]
    yb = np.array(yb)[order]
    col = style.mu_color(mu)
    ax.plot(xb, yb, color=col, **style.SECONDARY)
    ax.plot(
        xb,
        yb,
        "o",
        color=col,
        ms=5,
        mec="white",
        mew=0.6,
        ls="",
        label=f"$\\mu = {mu:g}$",
    )
    ax.plot([xb[0]], [yb[0]], "o", mfc="none", mec=col, mew=1.4, ms=10)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.set_xlabel("noise scale $B_{\\mathrm{noise}}$ (tokens)")
ax.set_ylabel("critical batch size $B^\\ast$ (tokens)")
ax.legend(title="momentum $\\mu$", loc="upper left")

fig.suptitle("Critical batch size $B^\\ast$ vs noise scale $B_{\\mathrm{noise}}$")
fig.tight_layout()
style.save(fig, "bnoise_vs_bstar")
