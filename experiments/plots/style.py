# experiments/plots/style.py
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =========================
# Global style (white, modern, article-ready)
# =========================
plt.style.use("seaborn-v0_8-white")

DEFAULT_FIGSIZE = (10, 6)

# Your palette (kept)
METHOD_STYLES = {
    "jeffreys": {"label": "Jeffreys equal-tailed", "color": "#d62728"},
    "cp": {"label": "Exact Clopper–Pearson", "color": "#ff7f0e"},
    "normal": {"label": "Normal approximation", "color": "#2ca02c"},
    "jeffreys_ecb": {"label": "Jeffreys ECB (upper)", "color": "#1f77b4"},
    "cp_upper": {"label": "Exact CP upper", "color": "#8c564b"},
    "normal_upper": {"label": "Normal upper", "color": "#9467bd"},
}

def apply_rcparams():
    """One place to make figures look consistent everywhere."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,

        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.titleweight": "semibold",
        "axes.labelweight": "regular",

        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        "axes.linewidth": 1.0,
        "grid.linewidth": 0.8,
    })

apply_rcparams()


def new_figure(figsize=DEFAULT_FIGSIZE):
    """
    White background, subtle grid, clean spines.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # subtle grid (major only)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(False, which="minor")

    # remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # nicer ticks
    ax.tick_params(axis="both", which="major", length=5, width=1.0, direction="out")
    ax.tick_params(axis="both", which="minor", length=3, width=0.8, direction="out")

    return fig, ax


def finalize_ax(
    ax,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    add_legend: bool = True,
    legend_loc: str = "best",
    nominal_level: float | None = None,
    nominal_label: str = "Nominal level",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    y_as_percent: bool = False,
    x_nbins: int = 6,
    y_nbins: int = 6,
):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if nominal_level is not None:
        ax.axhline(
            y=nominal_level,
            color="black",
            linestyle=(0, (4, 4)),  # nicer dashed
            linewidth=1.4,
            alpha=0.9,
            label=nominal_label,
            zorder=1,
        )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=x_nbins))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=y_nbins))

    if y_as_percent:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    if add_legend:
        ax.legend(loc=legend_loc, frameon=False)

    return ax


def annotate_min(ax, x: float, y: float, text: str):
    """Modern, minimal callout."""
    ax.scatter([x], [y], s=30, zorder=4)
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(10, -18),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", alpha=0.95),
    )


def add_drift_marker(ax, T0: int, T: int):
    """
    Consistent drift onset marker (neutral, article-friendly).
    - Vertical line at T0+0.5
    - Light shaded region after onset
    """
    onset = T0 + 0.5
    end = T + 0.5
    ax.axvline(
        x=onset,
        color="black",
        alpha=0.8,
        linestyle=(0, (4, 4)),
        linewidth=1.4,
        zorder=2,
    )
    ax.axvspan(
        onset,
        end,
        color="black",
        alpha=0.04,
        zorder=0,
    )

def annotate_min_below_ylim(
    ax,
    p_at_min: float,
    min_y: float,
    ylim_low: float,
    text_color: str = "red",
):
    """
    If the minimum value is below the visible y-limit (ylim_low),
    draw an arrow to the bottom boundary and show the true min value.
    """
    # Arrow points to the bottom boundary (visible), text is below it (outside axes)
    ax.annotate(
        f"Minimum coverage: {min_y:.3f}",
        xy=(p_at_min, ylim_low),              # point on the visible boundary
        xycoords="data",
        xytext=(p_at_min, ylim_low - 0.06),   # text below the axis
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color=text_color, linewidth=1.2),
        fontsize=11,
        color=text_color,
        ha="center",
        va="top",
        clip_on=False,                        # IMPORTANT: allow outside drawing
    )

import numpy as np

def find_x_at_y_near_min(
    x: np.ndarray,
    y: np.ndarray,
    y0: float,
    idx_min: int,
) -> float | None:
    """
    Approximate an x such that y(x) = y0 (linear interpolation),
    choosing the crossing closest to idx_min (search left & right).
    Returns None if no crossing exists.
    """
    if len(x) < 2:
        return None

    # If the curve never reaches y0, no crossing
    if (y >= y0).all() or (y <= y0).all():
        return None

    def interp_x(i_left: int, i_right: int) -> float | None:
        y1, y2 = y[i_left], y[i_right]
        x1, x2 = x[i_left], x[i_right]
        if y2 == y1:
            return None
        w = (y0 - y1) / (y2 - y1)
        return float(x1 + w * (x2 - x1))

    best = None  # (distance_in_index, x_cross)

    # search left from idx_min for a sign change around y0
    for i in range(idx_min, 0, -1):
        if (y[i] - y0) == 0:
            return float(x[i])
        if (y[i] - y0) * (y[i - 1] - y0) < 0:
            xc = interp_x(i - 1, i)
            if xc is not None:
                best = (idx_min - i, xc)
            break

    # search right from idx_min
    for i in range(idx_min, len(x) - 1):
        if (y[i] - y0) == 0:
            return float(x[i])
        if (y[i] - y0) * (y[i + 1] - y0) < 0:
            xc = interp_x(i, i + 1)
            if xc is not None:
                cand = (i - idx_min, xc)
                if best is None or cand[0] < best[0]:
                    best = cand
            break

    return None if best is None else best[1]


def annotate_min_below_ylim_at_crossing(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    idx_min: int,
    ylim_low: float,
    text_color: str = "red",
):
    """
    If min(y) < ylim_low, annotate the true min and draw an arrow to the point
    where the curve crosses y=ylim_low closest to the minimum (approx f^{-1}(ylim_low)).
    Falls back to x at min if no crossing.
    """
    min_x = float(x[idx_min])
    min_y = float(y[idx_min])

    x_cross = find_x_at_y_near_min(x=x, y=y, y0=ylim_low, idx_min=idx_min)
    if x_cross is None:
        x_cross = min_x

    ax.annotate(
        f"Minimum coverage: {min_y:.3f} (p={min_x:.3f})",
        xy=(x_cross, ylim_low),
        xycoords="data",
        xytext=(x_cross, ylim_low - 0.06),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color=text_color, linewidth=1.1),
        fontsize=11,
        color=text_color,
        ha="center",
        va="top",
        clip_on=False,
    )





def save_figure(fig, out_path: Path, also_pdf: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}" + (" (+ pdf)" if also_pdf else ""))
