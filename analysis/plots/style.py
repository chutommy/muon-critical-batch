"""Shared plotting style."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# --- Colour palettes (single source of truth) ---

# MU_COLORS = {0.85: "#0072B2", 0.95: "#E69F00", 0.99: "#009E73"}
# _B_CMAP = mpl.colormaps["viridis"]
# _L_CMAP = mpl.colormaps["cividis"]

cmap = mpl.colormaps["magma"]
MU_COLORS = {0.85: to_hex(cmap(0.15)), 0.95: to_hex(cmap(0.5)), 0.99: to_hex(cmap(0.8))}
_B_CMAP = cmap
_L_CMAP = cmap.reversed()

# --- Figure sizes ---

SIZE = {
    "square": (5.0, 5.0),
    "landscape": (6.4, 4.4),
    "1x2": (10.0, 4.4),
    "1x3": (12.0, 4.4),
    "2x3": (12.0, 6.6),
    "3x3": (12.0, 9.0),
    "4x3": (10.0, 8.0),
    "2row3col": (12.0, 7.0),
}

# --- Named line-style dicts ---
# Each is a kwargs dict spread into ax.plot via `**style.<NAME>`. Pick by role,
# not by appearance: changing the visual is a style.py edit, not a plot edit.

PRIMARY = dict(linestyle="-", linewidth=1.8, alpha=1.0)
SECONDARY = dict(linestyle="-", linewidth=1.2, alpha=1.0)
REF = dict(linestyle="--", linewidth=1.0, alpha=0.7)
EXTRAP = dict(linestyle=(0, (4, 3)), linewidth=0.8, alpha=1.0)
SPARSE = dict(linestyle=(0, (3, 2)), linewidth=1.0, alpha=0.45)
VLINE = dict(linestyle="-", linewidth=0.9, alpha=0.6)
VLINE_ALT = dict(linestyle=(0, (1, 4)), linewidth=1.4, alpha=1.0)
UNDERLAY = dict(linestyle="-", linewidth=0.6, alpha=0.25)

# --- Grey constants ---

GREY_DARK = "0.30"
GREY_MID = "0.55"
GREY_LIGHT = "0.85"

# --- Helpers ---


def format_B(b: int) -> str:
    """Canonical batch-size label (524288 -> '524K', 2_097_152 -> '2M')."""
    b = int(b)
    return f"{b // 1000}K" if b < 1_000_000 else f"{b // 1_000_000}M"


def batch_xaxis(ax, batch_sizes):
    """Log-2 x-axis with labelled ticks at every $B$."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([format_B(b) for b in batch_sizes], rotation=45, ha="right")


def outside_right_legend(fig, handles, labels, title):
    """Figure-level legend centred to the right of the axes."""
    fig.legend(handles, labels, title=title, loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0)


def plot_with_sparse(ax, x, y, sparse_mask, color, label=None, marker="o", ms=5):
    """Plot (x, y) with solid confident points and sparse low-confidence points."""
    import numpy as np

    x = np.asarray(x)
    y = np.asarray(y)
    sparse_mask = np.asarray(sparse_mask, dtype=bool)
    solid_mask = ~sparse_mask

    if solid_mask.any():
        ax.plot(x[solid_mask], y[solid_mask], color=color, marker=marker, ms=ms, mec="white", mew=0.6, label=label, **PRIMARY)

    if sparse_mask.any():
        first_sparse = np.where(sparse_mask)[0]
        last_solid_before = None
        if solid_mask.any():
            solid_indices = np.where(solid_mask)[0]
            before = solid_indices[solid_indices < first_sparse[0]]
            if len(before) > 0:
                last_solid_before = before[-1]

        bridge_x = []
        bridge_y = []
        if last_solid_before is not None:
            bridge_x = [x[last_solid_before], x[first_sparse[0]]]
            bridge_y = [y[last_solid_before], y[first_sparse[0]]]

        ax.plot(bridge_x, bridge_y, color=color, **SPARSE)
        ax.plot(x[sparse_mask], y[sparse_mask], color=color, marker=marker, ms=ms, mfc="white", mec=color, mew=1.2, **SPARSE)


# --- Colour accessors (signatures unchanged) ---

MUS = [0.85, 0.95, 0.99]
BATCH_SIZES = [32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608]

_B_COLORS = {b: _B_CMAP(0.10 + 0.80 * i / (len(BATCH_SIZES) - 1)) for i, b in enumerate(BATCH_SIZES)}


def mu_color(mu: float):
    """Colour for momentum mu (nearest of MUS)."""
    return MU_COLORS[min(MUS, key=lambda m: abs(m - mu))]


def b_color(b: int):
    """Colour for batch size b (nearest of BATCH_SIZES)."""
    return _B_COLORS[min(BATCH_SIZES, key=lambda x: abs(x - b))]


def l_colors(levels):
    """Dict of {L: colour} from the $L$ colormap."""
    levels = sorted(levels)
    n = len(levels)
    if n == 1:
        return {levels[0]: _L_CMAP(0.50)}
    return {L: _L_CMAP(0.15 + 0.70 * i / (n - 1)) for i, L in enumerate(levels)}


def b_norm():
    """(LogNorm, cmap) for batch-coloured continuous mappings."""
    return mpl.colors.LogNorm(vmin=BATCH_SIZES[0], vmax=BATCH_SIZES[-1]), _B_CMAP


# --- rcParams and save ---


def apply_style():
    """Set global matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#cccccc",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.6,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.titlesize": 12,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, name: str):
    """Save fig as PDF and PNG to analysis/figures/, then close."""
    os.makedirs(FIG_DIR, exist_ok=True)
    stem = name if name.startswith("plot_") else f"plot_{name}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
