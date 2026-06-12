"""Shared data loading and per-run reductions."""

import os
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

TABLES = os.path.join(os.path.dirname(__file__), "..", "tables")
TOKEN_CAP = 1_000_000_000
SEQLEN = 1024

MUS = [0.85, 0.95, 0.99]
BATCH_SIZES = [32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608]
LR_GRID_REL = [0.5, 0.75, 1.0, 1.5]


class Tables(NamedTuple):
    runs: pd.DataFrame
    val_loss: pd.DataFrame
    noise_scale: pd.DataFrame
    train_step: pd.DataFrame


def load_tables() -> Tables:
    def rd(f):
        return pd.read_parquet(os.path.join(TABLES, f))

    return Tables(
        rd("runs.parquet"),
        rd("val_loss.parquet"),
        rd("noise_scale.parquet"),
        rd("train_step.parquet"),
    )


from style import format_B  # noqa: F401

# ------------------------
#  Per-run single numbers
# ------------------------


def _sorted_curve(g):
    g = g.sort_values("tokens")
    return g["tokens"].to_numpy(float), g


def loss_at_tokens(val_loss: pd.DataFrame, target: int = TOKEN_CAP) -> pd.DataFrame:
    """Per-run validation loss interpolated at target tokens."""
    out = []
    for rid, g in val_loss.groupby("run_id"):
        t, g = _sorted_curve(g)
        out.append((rid, float(np.interp(target, t, g["val_loss"].to_numpy(float)))))
    return pd.DataFrame(out, columns=["run_id", "val_loss_at_target"])


def crossing(
    val_loss: pd.DataFrame,
    L: float,
    token_cap: int = TOKEN_CAP,
    interpolate: bool = True,
) -> pd.DataFrame:
    """First crossing of loss L within the cap, per run."""
    rows = []
    for rid, g in val_loss.groupby("run_id"):
        g = g[g["tokens"] <= token_cap].sort_values("tokens")
        hit_idx = np.where(g["val_loss"].to_numpy(float) <= L)[0]
        if len(hit_idx):
            i = hit_idx[0]
            if interpolate and i > 0:
                r0, r1 = g.iloc[i - 1], g.iloc[i]
                l0, l1 = float(r0["val_loss"]), float(r1["val_loss"])
                t = (l0 - L) / (l0 - l1)
                step = float(r0["step"]) + t * (float(r1["step"]) - float(r0["step"]))
                tok = float(r0["tokens"]) + t * (float(r1["tokens"]) - float(r0["tokens"]))
                time = float(r0["training_time_s"]) + t * (float(r1["training_time_s"]) - float(r0["training_time_s"]))
            else:
                r = g.iloc[i]
                step, tok, time = (
                    float(r["step"]),
                    float(r["tokens"]),
                    float(r["training_time_s"]),
                )
            rows.append((rid, step, tok, tok / SEQLEN, time))
        else:
            rows.append((rid, np.nan, np.nan, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["run_id", "step", "tokens", "examples", "training_time_s"])


def steps_to_L_tuned(
    runs: pd.DataFrame,
    val_loss: pd.DataFrame,
    L: float,
    token_cap: int = TOKEN_CAP,
    interpolate: bool = True,
) -> pd.DataFrame:
    """LR-tuned S(L): per (B, mu), the run with fewest steps to L."""
    cr = crossing(val_loss, L, token_cap, interpolate=interpolate).dropna(subset=["step"])
    cr = cr.merge(runs[["run_id", "batch_size", "mu", "lr_muon"]], on="run_id")
    idx = cr.groupby(["batch_size", "mu"])["step"].idxmin()
    best = cr.loc[idx].copy()
    best["E"] = best["step"] * best["batch_size"]
    return best.sort_values(["mu", "batch_size"]).reset_index(drop=True)


def best_lr_per_Bmu(runs: pd.DataFrame, val_loss: pd.DataFrame, token_cap: int = TOKEN_CAP) -> pd.DataFrame:
    """Per (B, mu): the run with lowest interpolated loss-at-cap."""
    la = loss_at_tokens(val_loss, token_cap).merge(runs[["run_id", "batch_size", "mu", "lr_muon"]], on="run_id")
    idx = la.groupby(["batch_size", "mu"])["val_loss_at_target"].idxmin()
    best = la.loc[idx].copy()
    lo = la.groupby(["batch_size", "mu"])["lr_muon"].transform("min")
    hi = la.groupby(["batch_size", "mu"])["lr_muon"].transform("max")
    edge = (np.isclose(la["lr_muon"], lo)) | (np.isclose(la["lr_muon"], hi))
    best = best.merge(
        la.assign(_edge=edge).loc[idx, ["run_id", "_edge"]].rename(columns={"_edge": "on_edge"}),
        on="run_id",
    )
    return best.sort_values(["mu", "batch_size"]).reset_index(drop=True)


# ------------------------------------
#  B_noise reduced near a loss target
# ------------------------------------


def _attach_loss_to_probes(noise_scale, val_loss):
    """Interpolate val loss onto every noise probe (by tokens, per run)."""
    vmap = {rid: g.sort_values("tokens") for rid, g in val_loss.groupby("run_id")}
    parts = []
    for rid, g in noise_scale.groupby("run_id"):
        g = g.copy()
        v = vmap.get(rid)
        if v is not None and len(v) >= 2:
            g["loss"] = np.interp(
                g["tokens"].to_numpy(float),
                v["tokens"].to_numpy(float),
                v["val_loss"].to_numpy(float),
            )
        else:
            g["loss"] = np.nan
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _gmean(x):
    x = np.asarray(x, float)
    x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def bnoise_near_L(noise_scale: pd.DataFrame, val_loss: pd.DataFrame, L: float, window: float = 0.10) -> pd.DataFrame:
    """Per-run geometric-mean B_noise over the loss window [L(1-w), L(1+w)]."""
    n = _attach_loss_to_probes(noise_scale, val_loss)
    w = n[(n["loss"] >= L * (1 - window)) & (n["loss"] <= L * (1 + window))]
    g = w.groupby("run_id")["B_noise_tokens"].apply(_gmean).rename("bnoise").reset_index()
    return g


def bnoise_vs_loss(noise_scale: pd.DataFrame, val_loss: pd.DataFrame) -> pd.DataFrame:
    """All noise probes tagged with validation loss."""
    return _attach_loss_to_probes(noise_scale, val_loss)


# ---------------
#  B* estimation
# ---------------


@dataclass
class BStarFit:
    L: float
    mu: float
    # elbow (primary)
    bstar: float
    bstar_lo: float
    bstar_hi: float
    e_min: float
    elbow_interior: bool
    n_points: int
    # hyperbola
    s_min: float
    bstar_hyp: float
    hyp_ok: bool
    # raw points used
    B: np.ndarray
    S: np.ndarray
    E: np.ndarray


def fit_bstar(runs, val_loss, L, mu, fit_cap=2_097_152, token_cap=TOKEN_CAP, interpolate=True):
    """Estimate critical batch size B*(L, mu)."""
    d = steps_to_L_tuned(runs, val_loss, L, token_cap, interpolate=interpolate)
    d = d[(d["mu"] == mu) & (d["batch_size"] <= fit_cap)].sort_values("batch_size")
    B = d["batch_size"].to_numpy(float)
    S = d["step"].to_numpy(float)
    E = d["E"].to_numpy(float)
    n = len(B)

    bstar = bstar_lo = bstar_hi = e_min = np.nan
    interior = False
    if n >= 3:
        x = np.log2(B)
        y = np.log10(E)
        coef, cov = np.polyfit(x, y, 2, cov=True)
        a, b, c = coef
        if a > 0:
            xv = -b / (2 * a)
            interior = (x.min() - 0.5) <= xv <= (x.max() + 0.5)
            bstar = float(2**xv)
            e_min = float(10 ** (c - b * b / (4 * a)))
            J = np.array([b / (2 * a * a), -1.0 / (2 * a), 0.0])
            var_xv = float(J @ cov @ J)
            sx = np.sqrt(max(var_xv, 0.0))
            bstar_lo = float(2 ** (xv - sx))
            bstar_hi = float(2 ** (xv + sx))

    # hyperbola on right leg
    s_min = bstar_hyp = np.nan
    hyp_ok = False
    if n >= 3 and np.isfinite(bstar):
        leg = B >= bstar * 0.6
        if leg.sum() >= 3:
            try:
                p, _ = curve_fit(
                    lambda B, Smin, Bs: Smin * (1 + Bs / B),
                    B[leg],
                    S[leg],
                    p0=[max(S[leg].min(), 1.0), bstar],
                    bounds=([1e-6, 1e3], [np.inf, 1e9]),
                    maxfev=40000,
                )
                s_min, bstar_hyp = float(p[0]), float(p[1])
                hyp_ok = np.isfinite(bstar_hyp)
            except Exception:
                pass

    bstarfit = {
        "L": L,
        "mu": mu,
        "bstar": bstar,
        "bstar_lo": bstar_lo,
        "bstar_hi": bstar_hi,
        "e_min": e_min,
        "elbow_interior": interior,
        "n_points": n,
        "s_min": s_min,
        "bstar_hyp": bstar_hyp,
        "hyp_ok": hyp_ok,
        "B": B,
        "S": S,
        "E": E,
    }

    return BStarFit(**bstarfit)
