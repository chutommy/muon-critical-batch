"""Noise scale $B_{\\mathrm{noise}}$ at matched loss $L$ vs training batch size $B$."""

import common as C
import matplotlib.pyplot as plt
import numpy as np
import style

style.apply_style()
T = C.load_tables()
L_TARGETS = np.arange(3.45, 4.00 + 1e-9, 0.05)
WELL_SAMPLED = 524288

lcol = style.l_colors(L_TARGETS)
fig, ax = plt.subplots(figsize=style.SIZE["landscape"])

for L in L_TARGETS:
    tuned = set(C.steps_to_L_tuned(T.runs, T.val_loss, L).run_id)
    bn = C.bnoise_near_L(T.noise_scale, T.val_loss, L, window=0.04)
    bn = bn[bn.run_id.isin(tuned)].merge(T.runs[["run_id", "batch_size"]], on="run_id").dropna(subset=["bnoise"])
    if not len(bn):
        continue
    g = bn.groupby("batch_size")["bnoise"].apply(C._gmean).reindex(C.BATCH_SIZES).dropna()
    B = g.index.to_numpy(float)
    y = g.to_numpy(float)
    col = lcol[L]
    sparse_mask = B > WELL_SAMPLED
    style.plot_with_sparse(ax, B, y, sparse_mask, color=col, label=f"$L = {L:.2f}$")

ax.set_yscale("log")
style.batch_xaxis(ax, C.BATCH_SIZES)
ax.set_xlabel("batch size $B$ (tokens)")
ax.set_ylabel("noise scale $B_{\\mathrm{noise}}$ (tokens)")
ax.legend(title="loss target $L$", loc="upper right")

fig.suptitle("Noise scale $B_{\\mathrm{noise}}$ at matched loss $L$ vs training batch size $B$")
fig.tight_layout()
style.save(fig, "bnoise_stratification")
