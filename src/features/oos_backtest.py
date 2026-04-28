# src/features/oos_backtest.py
# -*- coding: utf-8 -*-
"""
OOS Backtest (paper-ready) — feature module
==========================================

This module contains the full out-of-sample (OOS) PD backtesting logic used in the
paper-ready pipeline:

- Build a (Quarter x Grade) aggregation table from a loan-level scored dataset.
- Compute Jeffreys / Clopper–Pearson / Normal Approx. intervals.
- Produce per-quarter binary heatmaps (reject/ok/NA).
- Build traffic-light matrices (p-value or decision-based).
- Produce severity plots (expected shortfall maps, evidence–severity plane).
- Plot PD evolution per grade (with Jeffreys CI).
- Plot Jeffreys posterior densities over time for one grade.
- Export a LaTeX snapshot table (booktabs) with robust rounding.

The module is intentionally side-effect free except for file outputs in the
`run_oos_backtest` entry point.

Typical usage (called from a CLI wrapper):
    from features.oos_backtest import BacktestConfig, run_oos_backtest
    cfg = BacktestConfig(oos_path=..., bucket_stats_path=..., out_dir=...)
    run_oos_backtest(cfg)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Ellipse, RegularPolygon, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

from scipy.stats import beta as beta_dist

# Project intervals (your existing implementation)
from experiments.stats.intervals import (
    jeffreys_alpha2,
    exact_cp,
    approx_normal,
)


# =============================================================================
# Styling (paper look)
# =============================================================================
#
# Color-blind safe palette built on Okabe–Ito.
# Triple encoding everywhere: hue + shape + letter, plus a soft cell tint
# so the global pattern reads at a glance even in grayscale.
#
PAPER = {
    # Surface
    "bg":       "#FFFFFF",   # soft off-white background
    "panel":    "#FFFFFF",   # bright panel for cells
    "grid":     "#FFFFFF",   # subtle grid
    "frame":    "#CBD0D6",   # outer frame
    "axis":     "#111827",   # near-black text
    "muted":    "#6B7280",   # muted grey labels
    "muted_2":  "#9CA3AF",   # secondary muted (year separators)

    # Categorical signals — pastel colorblind-friendly palette
    "blue":     "#9ECAE1",   # pastel blue        — B
    "amber":    "#F2C879",   # pastel amber       — A
    "red":      "#E9A17B",   # pastel vermillion  — R
    "na":       "#D6D6D6",   # light grey         — NA

    # Soft fills reused across figures for visual consistency.
    # Since the main traffic-light colors are already pastel, the same
    # tones can be used directly as cell fills in the binary heatmaps.
    "blue_tint":  "#9ECAE1",
    "amber_tint": "#F2C879",
    "red_tint":   "#E9A17B",
    "na_tint":    "#D6D6D6",

    # Misc tile color for NaN in binary heatmap
    "nan_tile": "#F1F5F9",

    # Marker styling
    "stroke":        "#FFFFFF",
    "stroke_width":  1.0,
    "marker_fontsize": 7.2,

    # Compact traffic-light table typography
    "cell_fontsize": 6.4,
    "quarter_fontsize": 6.4,
    "year_fontsize": 7.6,
    "grade_fontsize": 7.2,
    "legend_fontsize": 6.4,
}


def apply_paper_rcparams() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 420,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.facecolor": PAPER["bg"],
            "figure.facecolor": PAPER["bg"],
            "savefig.facecolor": PAPER["bg"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def paper_axes(ax) -> None:
    ax.set_facecolor(PAPER["bg"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, colors=PAPER["muted"])


# =============================================================================
# Traffic-light accessibility helpers
# =============================================================================

TL_LABELS = {
    "blue":  "Blue / ✓",
    "amber": "Amber / ●",
    "red":   "Red / ×",
    "na":    "NA",
}

TL_SYMBOLS = {
    "blue":  "✓",
    "amber": "●",
    "red":   "×",
    "na":    "—",
}

TL_MARKERS = {
    "blue":  "o",     # legend/scatter proxy for ellipse
    "amber": "D",     # diamond
    "red":   "^",     # triangle (pointing up)
    "na":    "s",     # square
}

TL_HATCHES = {
    "blue":  "",
    "amber": "///",
    "red":   "xxx",
    "na":    "...",
}

TL_TINTS = {
    "blue":  PAPER["blue_tint"],
    "amber": PAPER["amber_tint"],
    "red":   PAPER["red_tint"],
    "na":    PAPER["na_tint"],
}


def _tl_level(x) -> str:
    """Normalize traffic-light levels to one of: blue, amber, red, na."""
    if x is None:
        return "na"
    try:
        if pd.isna(x):
            return "na"
    except Exception:
        pass

    s = str(x).strip().lower()
    if s in {"blue", "amber", "red", "na"}:
        return s
    return "na"


def _tl_color(level: str) -> str:
    level = _tl_level(level)
    return PAPER[level] if level in {"blue", "amber", "red", "na"} else PAPER["na"]


def _tl_tint(level: str) -> str:
    return TL_TINTS.get(_tl_level(level), PAPER["na_tint"])


def _tl_text_color(level: str) -> str:
    level = _tl_level(level)
    if level == "amber":
        return PAPER["axis"]
    if level == "na":
        return PAPER["muted"]
    return PAPER["bg"]


def draw_tl_marker(
    ax,
    x: float,
    y: float,
    level: str,
    *,
    size: float = 0.72,
    annotate: bool = True,
    zorder: int = 3,
) -> None:
    """
    Draw a color-blind-friendly traffic-light marker.

    Shape encoding (dual-redundant with hue + letter):
      - Blue  : balanced horizontal ellipse + B
      - Amber : diamond + A
      - Red   : upward-pointing triangle + R
      - NA    : rounded square + NA
    """
    level = _tl_level(level)
    color = _tl_color(level)

    # Letter offset (some shapes need optical correction)
    label_dy = 0.0

    if level == "blue":
        # Balanced ellipse — softly oval, not flattened pancake-style.
        patch = Ellipse(
            (x, y),
            width=size * 0.82,
            height=size * 0.62,
            facecolor=color,
            edgecolor=PAPER["stroke"],
            linewidth=PAPER["stroke_width"],
            zorder=zorder,
        )

    elif level == "amber":
        # Diamond — RegularPolygon with 4 vertices, orientation 0 → diamond shape.
        patch = RegularPolygon(
            (x, y),
            numVertices=4,
            radius=size * 0.40,
            orientation=0.0,
            facecolor=color,
            edgecolor=PAPER["stroke"],
            linewidth=PAPER["stroke_width"],
            zorder=zorder,
        )

    elif level == "red":
        # FIX: orientation=0 means "vertex at top" → triangle pointing UP.
        # The previous π/2 rotation produced a sideways "pennant" look.
        patch = RegularPolygon(
            (x, y),
            numVertices=3,
            radius=size * 0.46,
            orientation=0.0,
            facecolor=color,
            edgecolor=PAPER["stroke"],
            linewidth=PAPER["stroke_width"],
            zorder=zorder,
        )
        # Optical centering: lower the "R" so it sits in the wide base.
        label_dy = -0.06 * size

    else:  # na
        w = size * 0.72
        h = size * 0.50
        patch = FancyBboxPatch(
            (x - w / 2.0, y - h / 2.0),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor=PAPER["stroke"],
            linewidth=PAPER["stroke_width"],
            zorder=zorder,
        )

    ax.add_patch(patch)

    if annotate:
        ax.text(
            x,
            y + label_dy,
            TL_SYMBOLS[level],
            ha="center",
            va="center",
            fontsize=PAPER["marker_fontsize"],
            fontweight=800,
            color=_tl_text_color(level),
            zorder=zorder + 1,
            clip_on=True,
        )


def traffic_light_legend_handles(include_na: bool = True):
    levels = ["blue", "amber", "red"]
    if include_na:
        levels.append("na")

    return [
        Line2D(
            [0],
            [0],
            marker=TL_MARKERS[level],
            linestyle="None",
            markersize=10,
            markerfacecolor=_tl_color(level),
            markeredgecolor=PAPER["stroke"],
            markeredgewidth=1.0,
            label=TL_LABELS[level],
        )
        for level in levels
    ]


def traffic_light_patch_handles(include_na: bool = True):
    levels = ["blue", "amber", "red"]
    if include_na:
        levels.append("na")

    return [
        Patch(
            facecolor=_tl_color(level),
            edgecolor=PAPER["axis"],
            linewidth=0.6,
            hatch=TL_HATCHES[level],
            label=TL_LABELS[level],
        )
        for level in levels
    ]


def legend_outside(ax, handles, labels=None, *, side="right", pad=0.02, fontsize=9):
    """Place legend outside the axes so it never overlaps the plot."""
    if labels is None:
        labels = [h.get_label() for h in handles]

    if side == "right":
        leg = ax.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            bbox_to_anchor=(1.0 + pad, 1.0),
            frameon=True,
            framealpha=1.0,
            facecolor=PAPER["bg"],
            edgecolor=PAPER["grid"],
            fontsize=fontsize,
            borderpad=0.7,
            labelspacing=0.6,
            handlelength=1.8,
        )
    elif side == "bottom":
        leg = ax.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10 - pad),
            ncol=max(1, min(5, len(handles))),
            frameon=True,
            framealpha=1.0,
            facecolor=PAPER["bg"],
            edgecolor=PAPER["grid"],
            fontsize=fontsize,
            borderpad=0.7,
            labelspacing=0.6,
            handlelength=1.8,
        )
    else:
        raise ValueError("side must be 'right' or 'bottom'")

    for txt in leg.get_texts():
        txt.set_color(PAPER["axis"])
    return leg


def save_fig(fig, png: Path, pdf: Path | None = None) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.03)
    if pdf is not None:
        pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class BacktestConfig:
    # I/O
    oos_path: Path
    bucket_stats_path: Path
    bucket_section: str = "train"
    out_dir: Path = Path("outputs/oos_backtest")
    save_pdf: bool = True

    # Columns
    vintage_col: str = "vintage"
    grade_col: str = "grade"
    default_col: str = "default_12m"
    pd_loan_col: str = "pd"

    # Target PD to test per grade (in the aggregated table)
    # computed as pd_ttc from bucket_stats + also has pd_hat (mean loan PD)
    pdk_target: str = "pd_ttc"  # "pd_ttc" or "pd_hat"

    # Stats / intervals
    conf_level: float = 0.95

    # PD evolution plots
    pd_evolution_max_grades: Optional[int] = None
    pd_plot_logy: bool = True
    pd_y_units: str = "bps"  # "bps" or "percent"

    # Traffic light
    tl_pval_amber: float = 0.10
    tl_prob_red: float = 0.95
    tl_prob_amber: float = 0.90
    tl_es_red: float = 0.0005     # 5 bps
    tl_es_amber: float = 0.00025  # 2.5 bps
    tl_main_mode: str = "decision"  # "pval" or "decision"
    focus_year: Optional[int] = None

    # Beta posterior plot
    beta_grade: int = 5
    beta_x_points: int = 1500
    beta_quantile_span: float = 0.999
    beta_fill_alpha: float = 0.10
    beta_line_alpha: float = 0.25
    beta_year_linewidth: float = 2.4
    beta_quarter_linewidth: float = 1.0

    # Paper LaTeX snapshot table
    paper_table_snapshot: str = "2023Q4"
    paper_table_units: str = "bps"      # "bps" or "percent"
    paper_table_alpha: float = 0.05
    paper_table_include_counts: bool = True
    paper_table_caption: str = "Jeffreys posterior diagnostics and ECB p-values (snapshot: 2023Q4)"
    paper_table_label: str = "tab:detailed_backtest_2023Q4"
    paper_table_stem: str = "paper_table_jeffreys_snapshot_2023Q4"


# =============================================================================
# Data + stats helpers
# =============================================================================

def _safe_int_grade(x) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(x)


def parse_vintage_to_period(s: pd.Series, freq: str = "Q") -> pd.PeriodIndex:
    s_str = s.astype(str).str.strip()
    m = s_str.str.upper().str.replace(" ", "", regex=False).str.match(r"^\d{4}-?Q[1-4]$")
    if m.all():
        cleaned = s_str.str.upper().str.replace(" ", "", regex=False).str.replace("-", "", regex=False)
        return pd.PeriodIndex(cleaned, freq="Q")
    dt = pd.to_datetime(s_str, errors="coerce")
    return dt.dt.to_period(freq)


def infer_year_from_quarter(q: str) -> int | None:
    q = str(q)
    if len(q) >= 4 and q[:4].isdigit():
        return int(q[:4])
    return None


def load_bucket_pd_map(bucket_stats_path: Path, section: str) -> dict[int, float]:
    data = json.loads(bucket_stats_path.read_text(encoding="utf-8"))
    if section not in data:
        raise KeyError(f"'{section}' not in bucket_stats.json. Keys: {list(data.keys())}")
    return {int(e["bucket"]): float(e["pd"]) for e in data[section]}


def build_vintage_grade_table(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    need_cols = [cfg.vintage_col, cfg.grade_col, cfg.default_col, cfg.pd_loan_col]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in OOS file: {missing}")

    x = df[need_cols].copy()
    x[cfg.grade_col] = x[cfg.grade_col].map(_safe_int_grade)

    agg = x.groupby([cfg.vintage_col, cfg.grade_col], as_index=False).agg(
        n=(cfg.default_col, "count"),
        d=(cfg.default_col, "sum"),
        pd_hat=(cfg.pd_loan_col, "mean"),
    )
    agg["pd_obs"] = agg["d"] / agg["n"]

    pd_ttc_map = load_bucket_pd_map(cfg.bucket_stats_path, cfg.bucket_section)
    agg["pd_ttc"] = agg[cfg.grade_col].map(pd_ttc_map).astype(float)

    try:
        per = parse_vintage_to_period(agg[cfg.vintage_col], freq="Q")
        agg["_per"] = per
        agg = agg.sort_values(["_per", cfg.grade_col]).drop(columns="_per")
    except Exception:
        agg = agg.sort_values([cfg.vintage_col, cfg.grade_col])

    return agg.reset_index(drop=True)


def fmt_interval(lb: float, ub: float) -> str:
    if not np.isfinite(lb) or not np.isfinite(ub):
        return ""
    return f"[{lb:.6g}, {ub:.6g}]"


def _reject(pd_k: float, lb: float, ub: float) -> bool:
    if not (np.isfinite(pd_k) and np.isfinite(lb) and np.isfinite(ub)):
        return False
    return (pd_k < lb) or (pd_k > ub)


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(min(1.0, max(0.0, x)))


def jeffreys_posterior_params(n: int, d: int) -> tuple[float, float]:
    return float(d + 0.5), float((n - d) + 0.5)


def expected_shortfall_beta(a: float, b: float, c: float) -> float:
    """
    ES = E[(p-c)+ | p ~ Beta(a,b)]
       = (a/(a+b)) * P(Beta(a+1,b)>c) - c * P(Beta(a,b)>c)
    """
    c = _clip01(c)
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
        return np.nan
    tail_ab = beta_dist.sf(c, a, b)
    tail_a1b = beta_dist.sf(c, a + 1.0, b)
    term1 = (a / (a + b)) * tail_a1b
    term2 = c * tail_ab
    es = term1 - term2
    return float(max(0.0, es))


def add_all_stats(df_vg: pd.DataFrame, cfg: BacktestConfig, pd_col: str) -> pd.DataFrame:
    out = df_vg.copy()
    if pd_col not in out.columns:
        raise ValueError(f"pd_col='{pd_col}' not in df columns")

    conf_level = float(cfg.conf_level)

    for name in ["Jeffreys", "Clopper-Pearson", "Normal Approx."]:
        out[f"{name}_lb"] = np.nan
        out[f"{name}_ub"] = np.nan
        out[f"{name}_CI"] = ""
        out[f"{name}_reject"] = np.nan

    out["Jeffreys_pval_H0"] = np.nan
    out["Jeffreys_prob_under"] = np.nan
    out["Jeffreys_ES"] = np.nan
    out["Jeffreys_q95_excess"] = np.nan

    n_arr = out["n"].to_numpy(int)
    d_arr = out["d"].to_numpy(int)
    pd_arr = out[pd_col].to_numpy(float)

    for i, (n, d, pd_k) in enumerate(zip(n_arr, d_arr, pd_arr)):
        if n <= 0 or not np.isfinite(pd_k):
            continue

        pd_k = _clip01(float(pd_k))
        a, b = jeffreys_posterior_params(int(n), int(d))

        pval_H0 = beta_dist.cdf(pd_k, a, b)  # P(p <= PD_k | y)
        out.at[i, "Jeffreys_pval_H0"] = float(pval_H0) if np.isfinite(pval_H0) else np.nan
        out.at[i, "Jeffreys_prob_under"] = float(1.0 - pval_H0) if np.isfinite(pval_H0) else np.nan

        out.at[i, "Jeffreys_ES"] = expected_shortfall_beta(a, b, pd_k)
        q95 = beta_dist.ppf(conf_level, a, b)
        out.at[i, "Jeffreys_q95_excess"] = float(q95 - pd_k) if np.isfinite(q95) else np.nan

        lb, ub = jeffreys_alpha2(int(n), int(d), conf_level)
        lb = float(lb)
        ub = float(ub)
        out.at[i, "Jeffreys_lb"] = lb
        out.at[i, "Jeffreys_ub"] = ub
        out.at[i, "Jeffreys_CI"] = fmt_interval(lb, ub)
        out.at[i, "Jeffreys_reject"] = _reject(pd_k, lb, ub)

        lb, ub = exact_cp(int(n), int(d), conf_level)
        lb = float(lb)
        ub = float(ub)
        out.at[i, "Clopper-Pearson_lb"] = lb
        out.at[i, "Clopper-Pearson_ub"] = ub
        out.at[i, "Clopper-Pearson_CI"] = fmt_interval(lb, ub)
        out.at[i, "Clopper-Pearson_reject"] = _reject(pd_k, lb, ub)

        lb, ub = approx_normal(int(n), int(d), conf_level)
        lb = max(0.0, float(lb))
        ub = min(1.0, float(ub))
        out.at[i, "Normal Approx._lb"] = lb
        out.at[i, "Normal Approx._ub"] = ub
        out.at[i, "Normal Approx._CI"] = fmt_interval(lb, ub)
        out.at[i, "Normal Approx._reject"] = _reject(pd_k, lb, ub)

    return out


def _safe_fname(x: str) -> str:
    s = str(x).strip().replace(" ", "_")
    for ch in ["/", "\\", ":", ";", "|", ",", "."]:
        s = s.replace(ch, "-")
    return s


# =============================================================================
# Paper table helpers (LaTeX)
# =============================================================================

def _fmt_num(x: float, decimals: int) -> str:
    if not np.isfinite(x):
        return ""
    return f"{float(x):.{decimals}f}"


def fmt_prob(x: float, units: str = "bps") -> str:
    if not np.isfinite(x):
        return ""
    x = float(x)
    u = (units or "").lower().strip()

    if u == "bps":
        v = x * 1e4
        if v < 1:
            return _fmt_num(v, 2)
        if v < 10:
            return _fmt_num(v, 1)
        return _fmt_num(v, 0)

    v = x * 100.0
    if v < 0.1:
        return _fmt_num(v, 3)
    if v < 1.0:
        return _fmt_num(v, 2)
    return _fmt_num(v, 1)


def fmt_pval_ecb(p: float) -> str:
    if not np.isfinite(p):
        return ""
    p = float(min(1.0, max(0.0, p)))
    if p < 0.0005:
        return "<0.1"
    if p > 0.9995:
        return ">99.9"
    return f"{p*100.0:.1f}"


def ecb_status_from_pval(pval_h0: float, alpha: float = 0.05) -> str:
    if not np.isfinite(pval_h0):
        return "NA"
    return "Fail" if float(pval_h0) <= float(alpha) else "Pass"


def latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    repl = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def render_snapshot_table_latex(
    out_long: pd.DataFrame,
    quarter: str,
    *,
    units: str = "bps",
    alpha: float = 0.05,
    include_counts: bool = True,
    caption: str = "",
    label: str = "",
) -> str:
    df = out_long.copy()
    df["Quarter"] = df["Quarter"].astype(str)
    df = df[df["Quarter"] == str(quarter)].copy()
    if df.empty:
        raise ValueError(f"No rows for quarter={quarter} in out_long.")

    df["Grade"] = df["Grade"].astype(int)
    df = df.sort_values("Grade")

    df["_pd"] = df["PD_k"].astype(float)
    df["_lb"] = df["Jeffreys_lb"].astype(float)
    df["_ub"] = df["Jeffreys_ub"].astype(float)
    df["_pval"] = df["Jeffreys_pval_H0"].astype(float)
    df["_status"] = df["_pval"].apply(lambda x: ecb_status_from_pval(x, alpha=alpha))

    def _status_cell(st: str) -> str:
        if st == "Pass":
            return r"\textcolor{teal}{\textbf{Pass}}"
        if st == "Fail":
            return r"\textcolor{red}{\textbf{Fail}}"
        return "NA"

    u = (units or "").lower().strip()
    if u == "bps":
        pd_hdr = r"Target PD (bps)"
        lb_hdr = r"LB (2.5\%, bps)"
        ub_hdr = r"UB (97.5\%, bps)"
    else:
        pd_hdr = r"Target PD (\%)"
        lb_hdr = r"LB (2.5\%, \%)"
        ub_hdr = r"UB (97.5\%, \%)"

    if include_counts:
        colspec = "c r r r r r r r"
        header = (
            r"\textbf{Grade} & \textbf{N} & \textbf{D} & "
            rf"\textbf{{{pd_hdr}}} & \textbf{{{lb_hdr}}} & \textbf{{{ub_hdr}}} & "
            r"\textbf{ECB p-value} $\mathbf{P(p \le PD_{tgt})}$ & \textbf{Status} \\"
        )
    else:
        colspec = "c r r r r r"
        header = (
            rf"\textbf{{Grade}} & \textbf{{{pd_hdr}}} & \textbf{{{lb_hdr}}} & \textbf{{{ub_hdr}}} & "
            r"\textbf{ECB p-value} $\mathbf{P(p \le PD_{tgt})}$ & \textbf{Status} \\"
        )

    rows = []
    for _, r in df.iterrows():
        g = int(r["Grade"])
        pd_cell = fmt_prob(r["_pd"], units=units)
        lb_cell = fmt_prob(r["_lb"], units=units)
        ub_cell = fmt_prob(r["_ub"], units=units)
        pval_cell = fmt_pval_ecb(r["_pval"])
        st_cell = _status_cell(r["_status"])

        if include_counts:
            n_cell = f"{int(r['n_k'])}" if np.isfinite(r["n_k"]) else ""
            d_cell = f"{int(r['d_k'])}" if np.isfinite(r["d_k"]) else ""
            rows.append(
                f"{g} & {n_cell} & {d_cell} & {pd_cell} & {lb_cell} & {ub_cell} & {pval_cell}\\% & {st_cell} \\\\"
            )
        else:
            rows.append(
                f"{g} & {pd_cell} & {lb_cell} & {ub_cell} & {pval_cell}\\% & {st_cell} \\\\"
            )

    cap = latex_escape(caption) if caption else ""
    lab = latex_escape(label) if label else ""

    note_units = "bps" if u == "bps" else r"\%"
    note = (
        r"\item \textit{Note:} The Jeffreys posterior is $\mathrm{Beta}(D+1/2,\,N-D+1/2)$. "
        r"LB/UB are bilateral Jeffreys credible bounds. "
        r"ECB p-value is $\mathbb{P}(p \le PD_{tgt}\mid y)$ and the one-sided status flags underestimation "
        rf"when p-value $\le {alpha:.2f}$. Values are reported in {note_units}."
    )

    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    if cap:
        tex.append(rf"\caption{{{cap}}}")
    if lab:
        tex.append(rf"\label{{{lab}}}")
    tex.append(r"\small")
    tex.append(r"\renewcommand{\arraystretch}{1.15}")
    tex.append(rf"\begin{{tabular}}{{{colspec}}}")
    tex.append(r"\toprule")
    tex.append(header)
    tex.append(r"\midrule")
    tex.extend(rows)
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\begin{tablenotes}")
    tex.append(r"\footnotesize")
    tex.append(note)
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{table}")
    tex.append("")
    return "\n".join(tex)


# =============================================================================
# Plotting
# =============================================================================

def _year_break_indices(quarters: list) -> list[int]:
    """Index positions where the year changes (used for subtle vertical separators)."""
    years = [str(q)[:4] for q in quarters]
    return [i for i in range(1, len(years)) if years[i] != years[i - 1]]


def plot_binary_heatmap_for_quarter(
    df_q: pd.DataFrame,
    out_png: Path,
    out_pdf: Path | None = None,
    annotate: bool = True,
) -> None:
    method_cols = ["Jeffreys_reject", "Clopper-Pearson_reject", "Normal Approx._reject"]
    xlabels = ["Jeffreys", "Clopper–Pearson", "Normal"]

    Z = df_q[method_cols].copy()
    mat = np.full(Z.shape, -1, dtype=int)
    for j, col in enumerate(method_cols):
        s = Z[col]
        is_na = s.isna().to_numpy()
        val = s.fillna(False).astype(int).to_numpy()
        mat[:, j] = val
        mat[is_na, j] = -1

    n_grades = mat.shape[0]
    cmap = ListedColormap([PAPER["nan_tile"], PAPER["blue_tint"], PAPER["red_tint"]])
    norm = BoundaryNorm(boundaries=[-1.5, -0.5, 0.5, 1.5], ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.42 * n_grades)))
    paper_axes(ax)

    ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest", origin="lower")

    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=0, color=PAPER["muted"])
    ax.set_yticks(range(n_grades))
    ax.set_yticklabels(df_q.index.astype(str), color=PAPER["muted"])

    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_grades, 1), minor=True)
    ax.grid(which="minor", color=PAPER["grid"], linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for r in range(n_grades):
            for c in range(len(xlabels)):
                v = mat[r, c]
                if v == -1:
                    txt, color = "—", PAPER["muted"]
                elif v == 0:
                    txt, color = "✓", PAPER["axis"]
                else:
                    txt, color = "✕", PAPER["axis"]
                ax.text(
                    c, r,
                    txt,
                    ha="center", va="center",
                    fontsize=12, fontweight=800,
                    color=color,
                )

    handles = [
        Patch(facecolor=PAPER["blue_tint"], edgecolor=PAPER["blue"], linewidth=0.8, label="OK"),
        Patch(facecolor=PAPER["red_tint"],  edgecolor=PAPER["red"],  linewidth=0.8, label="Reject"),
        Patch(facecolor=PAPER["nan_tile"],  edgecolor=PAPER["muted"], linewidth=0.8, label="NA"),
    ]
    legend_outside(ax, handles, side="right", pad=0.02)

    fig.tight_layout()
    save_fig(fig, out_png, out_pdf)


def plot_traffic_light_matrix(mat: pd.DataFrame, out_png: Path, out_pdf: Path | None = None) -> None:
    """
    Compact publication-quality traffic-light matrix.

    Design:
      - slightly rectangular horizontal cells;
      - thin white separators;
      - colorblind-safe palette: blue, amber, vermillion, grey;
      - bold letters inside cells;
      - subtle year labels above quarters;
      - compact journal-ready layout.
    """
    grades = mat.index.tolist()
    quarters = [str(q) for q in mat.columns.tolist()]
    nrows, ncols = len(grades), len(quarters)

    if nrows == 0 or ncols == 0:
        return

    # Geometry — wider than tall
    cell_w = 1.12
    cell_h = 0.62
    sep = 0.030
    year_gap = 0.12
    year_breaks = _year_break_indices(quarters)

    x_left = np.zeros(ncols)
    offset = 0.0
    for c in range(ncols):
        if c in year_breaks:
            offset += year_gap
        x_left[c] = c * cell_w + offset

    total_w = x_left[-1] + cell_w
    total_h = nrows * cell_h

    fig_w = max(6.6, 0.55 * ncols + 1.2)
    fig_h = max(2.4, 0.22 * nrows + 0.75)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(PAPER["bg"])
    ax.set_facecolor(PAPER["bg"])
    ax.axis("off")

    letter_color = {
        "blue": PAPER["axis"],
        "amber": PAPER["axis"],
        "red": PAPER["axis"],
        "na": PAPER["muted"],
    }

    # Cells
    for r in range(nrows):
        for c in range(ncols):
            lvl = _tl_level(mat.iloc[r, c])

            x0 = x_left[c] + sep / 2.0
            y0 = r * cell_h + sep / 2.0
            w = cell_w - sep
            h = cell_h - sep

            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    w,
                    h,
                    facecolor=_tl_color(lvl),
                    edgecolor="#FFFFFF",
                    linewidth=0.45,
                    zorder=2,
                )
            )

            ax.text(
                x_left[c] + cell_w / 2.0,
                r * cell_h + cell_h / 2.0,
                TL_SYMBOLS[lvl],
                ha="center",
                va="center",
                fontsize=PAPER["cell_fontsize"] + 0.8 if lvl != "na" else PAPER["cell_fontsize"],
                fontweight=700,
                color=letter_color[lvl],
                family="DejaVu Sans",
                zorder=3,
            )

    # Grade labels
    for r, grade in enumerate(grades):
        ax.text(
            -0.15,
            r * cell_h + cell_h / 2.0,
            str(grade),
            ha="right",
            va="center",
            fontsize=PAPER["grade_fontsize"],
            color=PAPER["axis"],
            family="DejaVu Sans",
            fontstyle="italic",
        )

    # Quarter labels
    for c, q in enumerate(quarters):
        q_clean = q.replace("-", "")
        q_short = q_clean[-2:] if len(q_clean) >= 2 else q_clean

        ax.text(
            x_left[c] + cell_w / 2.0,
            -0.10,
            q_short,
            ha="center",
            va="top",
            fontsize=PAPER["quarter_fontsize"],
            color=PAPER["muted"],
            family="DejaVu Sans",
            fontweight=500,
        )

    # Year labels and rules
    year_groups: dict[str, list[int]] = {}
    for c, q in enumerate(quarters):
        year = q[:4]
        year_groups.setdefault(year, []).append(c)

    year_y = total_h + 0.22
    rule_y = total_h + 0.075

    for year, cols in year_groups.items():
        x0 = x_left[cols[0]]
        x1 = x_left[cols[-1]] + cell_w
        xm = 0.5 * (x0 + x1)

        ax.text(
            xm,
            year_y,
            year,
            ha="center",
            va="center",
            fontsize=PAPER["year_fontsize"],
            fontweight=600,
            color=PAPER["muted"],
            family="DejaVu Sans",
        )

        ax.plot(
            [x0 + 0.05, x1 - 0.05],
            [rule_y, rule_y],
            color=PAPER["frame"],
            linewidth=0.55,
            solid_capstyle="butt",
            zorder=1,
        )

    # Subtle vertical separators between years
    for c in year_breaks:
        xb = x_left[c] - 0.5 * year_gap
        ax.plot(
            [xb, xb],
            [0.02, total_h - 0.02],
            color="#FFFFFF",
            linewidth=1.0,
            zorder=4,
        )

    # Axis label
    ax.text(
        -0.15,
        total_h + 0.075,
        "Grade",
        ha="right",
        va="center",
        fontsize=PAPER["quarter_fontsize"],
        color=PAPER["muted"],
        family="DejaVu Sans",
    )

    # Compact legend
    legend_levels = ["blue", "amber", "red", "na"]
    legend_text = {
        "blue":  "Blue / ✓",
        "amber": "Amber / ●",
        "red":   "Red / ×",
        "na":    "NA",
    }
    

    sw = 0.35
    sh = 0.30
    label_pad = 0.035
    item_gap = 0.78
    legend_y = -0.78

    label_widths = {
    "blue": 1.08,
    "amber": 1.28,
    "red": 1.08,
    "na": 0.28,
    }
    legend_total_w = sum(sw + label_pad + label_widths[lvl] for lvl in legend_levels) + item_gap * (len(legend_levels) - 1)
    legend_x0 = (total_w - legend_total_w) / 2.0

    cursor = legend_x0
    for lvl in legend_levels:
        ax.add_patch(
            Rectangle(
                (cursor, legend_y - sh / 2.0),
                sw,
                sh,
                facecolor=_tl_color(lvl),
                edgecolor="#FFFFFF",
                linewidth=0.45,
                zorder=2,
            )
        )

        ax.text(
            cursor + sw / 2.0,
            legend_y,
            TL_SYMBOLS[lvl],
            ha="center",
            va="center",
            fontsize=5.8 if lvl != "na" else 5.0,
            fontweight=700,
            color=letter_color[lvl],
            family="DejaVu Sans",
            zorder=3,
        )

        ax.text(
            cursor + sw + label_pad,
            legend_y,
            legend_text[lvl],
            ha="left",
            va="center",
            fontsize=PAPER["legend_fontsize"],
            color=PAPER["axis"],
            family="DejaVu Sans",
        )

        cursor += sw + label_pad + label_widths[lvl] + item_gap

    # Final limits
    legend_left = legend_x0 - 0.10
    legend_right = legend_x0 + legend_total_w + 0.10
    x_lo = min(-0.65, legend_left)
    x_hi = max(total_w + 0.10, legend_right)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.95, total_h + 0.42)
    ax.set_aspect("equal")

    save_fig(fig, out_png, out_pdf)


def plot_traffic_light_summary_by_quarter(mat: pd.DataFrame, out_png: Path, out_pdf: Path | None = None) -> pd.DataFrame:
    quarters = mat.columns.tolist()

    def shares(col: pd.Series) -> dict[str, float]:
        tot = col.notna().sum()
        if tot == 0:
            return {"blue": 0.0, "amber": 0.0, "red": 0.0}
        return {
            "blue":  float((col == "blue").sum())  / tot,
            "amber": float((col == "amber").sum()) / tot,
            "red":   float((col == "red").sum())   / tot,
        }

    s = pd.DataFrame([shares(mat[q]) for q in quarters], index=quarters)

    fig_w = max(8.0, 0.6 * len(quarters)) + 1.4
    fig, ax = plt.subplots(figsize=(fig_w, 3.4))
    paper_axes(ax)

    bottom = np.zeros(len(quarters))
    bar_kw = {"edgecolor": PAPER["bg"], "linewidth": 0.8}

    ax.bar(
        quarters,
        s["blue"].to_numpy(),
        bottom=bottom,
        color=PAPER["blue"],
        hatch=TL_HATCHES["blue"],
        label=TL_LABELS["blue"],
        **bar_kw,
    )
    bottom += s["blue"].to_numpy()

    ax.bar(
        quarters,
        s["amber"].to_numpy(),
        bottom=bottom,
        color=PAPER["amber"],
        hatch=TL_HATCHES["amber"],
        label=TL_LABELS["amber"],
        **bar_kw,
    )
    bottom += s["amber"].to_numpy()

    ax.bar(
        quarters,
        s["red"].to_numpy(),
        bottom=bottom,
        color=PAPER["red"],
        hatch=TL_HATCHES["red"],
        label=TL_LABELS["red"],
        **bar_kw,
    )

    # Year separators on the bar chart too
    for xb in _year_break_indices(quarters):
        ax.axvline(xb - 0.5, color=PAPER["frame"], linewidth=1.0, alpha=0.55, zorder=0)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1, 6)])
    ax.grid(axis="y", color=PAPER["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=0)

    handles = traffic_light_patch_handles(include_na=False)
    legend_outside(ax, handles, side="right", pad=0.02)

    fig.tight_layout()
    save_fig(fig, out_png, out_pdf)
    return s


def plot_severity_map_year(out: pd.DataFrame, year: int, out_png: Path, out_pdf: Path | None = None) -> None:
    df = out.copy()
    df["Year"] = df["Quarter"].astype(str).str.slice(0, 4)
    df = df[df["Year"] == str(year)].copy()
    if df.empty:
        return

    per = parse_vintage_to_period(df["Quarter"].astype(str), freq="Q")
    q_order = (
        df[["Quarter"]]
        .assign(_per=per)
        .drop_duplicates()
        .sort_values("_per")["Quarter"]
        .astype(str)
        .tolist()
    )
    df["Quarter"] = pd.Categorical(df["Quarter"].astype(str), categories=q_order, ordered=True)
    df["Grade"] = df["Grade"].astype(int)

    mat = df.pivot(index="Grade", columns="Quarter", values="Jeffreys_ES").sort_index()
    ES_bps = mat.to_numpy(float) * 1e4

    finite = np.isfinite(ES_bps)
    vmax = float(np.nanquantile(ES_bps[finite], 0.98)) if finite.any() else 1.0

    fig_w = max(7.0, 1.0 + 0.95 * len(q_order)) + 1.4
    fig_h = max(3.6, 0.50 * len(mat.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    paper_axes(ax)

    # 'cividis' is perceptually uniform AND specifically engineered for
    # color-vision deficiencies — ideal for severity.
    im = ax.imshow(
        ES_bps,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
        origin="lower",
        cmap="cividis",
    )

    ax.set_xticks(range(len(q_order)))
    ax.set_xticklabels(q_order, rotation=0, color=PAPER["muted"])
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index.astype(str), color=PAPER["muted"])

    ax.set_xticks(np.arange(-0.5, len(q_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
    ax.grid(which="minor", color=PAPER["grid"], linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("ES (bps)", color=PAPER["muted"])
    cbar.ax.tick_params(colors=PAPER["muted"])
    cbar.outline.set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_png, out_pdf)


def plot_evidence_severity_plane_year(out: pd.DataFrame, year: int, cfg: BacktestConfig, out_png: Path, out_pdf: Path | None = None) -> None:
    df = out.copy()
    df["Year"] = df["Quarter"].astype(str).str.slice(0, 4)
    df = df[df["Year"] == str(year)].copy()
    if df.empty:
        return

    df["ES_bps"] = df["Jeffreys_ES"].astype(float) * 1e4
    df["prob_under"] = df["Jeffreys_prob_under"].astype(float)
    df["n_k"] = df["n_k"].astype(float)

    size = np.log1p(df["n_k"].to_numpy(float))
    denom = (np.nanmax(size) - np.nanmin(size) + 1e-12)
    size = 40 + 110 * (size - np.nanmin(size)) / denom

    def _tl_decision(prob_under: float, es: float) -> str:
        if not (np.isfinite(prob_under) and np.isfinite(es)):
            return "na"
        if (prob_under >= cfg.tl_prob_red) and (es >= cfg.tl_es_red):
            return "red"
        if (prob_under >= cfg.tl_prob_amber and es >= cfg.tl_es_amber) or (prob_under >= cfg.tl_prob_red):
            return "amber"
        return "blue"

    df["tl"] = df.apply(lambda r: _tl_decision(r["prob_under"], r["Jeffreys_ES"]), axis=1)
    df["_size"] = size

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    paper_axes(ax)
    ax.grid(True, color=PAPER["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Threshold guides — drawn before scatter so they sit underneath
    for y in (cfg.tl_prob_red, cfg.tl_prob_amber):
        ax.axhline(y, linewidth=1.0, color=PAPER["muted_2"], linestyle=":", zorder=1)
    for x in (cfg.tl_es_red * 1e4, cfg.tl_es_amber * 1e4):
        ax.axvline(x, linewidth=1.0, color=PAPER["muted_2"], linestyle=":", zorder=1)

    for level in ["blue", "amber", "red", "na"]:
        sub = df[df["tl"] == level]
        if sub.empty:
            continue
        ax.scatter(
            sub["ES_bps"],
            sub["prob_under"],
            s=sub["_size"],
            c=_tl_color(level),
            marker=TL_MARKERS[level],
            alpha=0.88,
            edgecolors=PAPER["stroke"],
            linewidths=1.0,
            label=TL_LABELS[level],
            zorder=3,
        )

    ax.set_xlabel("Expected shortfall (bps)", color=PAPER["axis"])
    ax.set_ylabel("P(p > PD | y)", color=PAPER["axis"])
    ax.set_ylim(0, 1)

    handles = traffic_light_legend_handles(include_na=True)
    legend_outside(ax, handles, side="right", pad=0.02)

    fig.tight_layout()
    save_fig(fig, out_png, out_pdf)


def _format_y_units(units: str):
    u = (units or "").lower().strip()
    if u == "percent":
        return lambda y: y * 100.0
    return lambda y: y * 1e4  # bps


def plot_pd_evolution_per_grade(
    out_long: pd.DataFrame,
    grade: int,
    cfg: BacktestConfig,
    out_png: Path,
    out_pdf: Path | None = None,
) -> None:
    df = out_long.copy()
    df["Quarter_str"] = df["Quarter"].astype(str)
    df["Grade_int"] = df["Grade"].astype(int)
    df = df[df["Grade_int"] == int(grade)].copy()
    if df.empty:
        return

    per = parse_vintage_to_period(df["Quarter_str"], freq="Q")
    df["_per"] = per
    df = df.sort_values("_per").drop(columns=["_per"])

    scale = _format_y_units(cfg.pd_y_units)

    x = np.arange(len(df))
    y_obs = df["p_hat_k"].to_numpy(float)
    y_lb = df["Jeffreys_lb"].to_numpy(float)
    y_ub = df["Jeffreys_ub"].to_numpy(float)
    y_tgt = df["PD_k"].to_numpy(float)

    y_obs_s = scale(y_obs)
    y_lb_s = scale(y_lb)
    y_ub_s = scale(y_ub)
    y_tgt_s = scale(y_tgt)

    fig, ax = plt.subplots(figsize=(10.4, 3.6))
    paper_axes(ax)
    ax.grid(axis="y", color=PAPER["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Two-tone CI band: light fill + soft edge for a refined look
    ci_face = "#D7DEE6"
    ci_edge = "#94A3B8"
    ax.fill_between(
        x,
        y_lb_s,
        y_ub_s,
        facecolor=ci_face,
        alpha=0.55,
        linewidth=0.0,
        label=f"Jeffreys {int(cfg.conf_level*100)}% CI",
        zorder=1,
    )
    ax.plot(x, y_lb_s, color=ci_edge, linewidth=0.8, alpha=0.7, zorder=1)
    ax.plot(x, y_ub_s, color=ci_edge, linewidth=0.8, alpha=0.7, zorder=1)

    ax.plot(
        x,
        y_obs_s,
        color=PAPER["axis"],
        linewidth=2.0,
        marker="o",
        markersize=4.0,
        markerfacecolor=PAPER["axis"],
        markeredgecolor=PAPER["bg"],
        markeredgewidth=0.8,
        label="Observed PD",
        zorder=3,
    )
    ax.plot(
        x,
        y_tgt_s,
        color=PAPER["blue"],
        linewidth=1.8,
        linestyle="--",
        label=f"Target PD ({cfg.pdk_target})",
        zorder=2,
    )

    q_labels = df["Quarter_str"].tolist()
    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, rotation=0, color=PAPER["muted"])

    if cfg.pd_plot_logy:
        pos = np.r_[
            y_obs_s[np.isfinite(y_obs_s) & (y_obs_s > 0)],
            y_tgt_s[np.isfinite(y_tgt_s) & (y_tgt_s > 0)],
        ]
        if pos.size > 0:
            ax.set_yscale("log")

    handles = [
        Patch(facecolor=ci_face, edgecolor=ci_edge, label=f"Jeffreys {int(cfg.conf_level*100)}% CI"),
        Line2D([0], [0], color=PAPER["axis"], lw=2.0, marker="o", markersize=4, label="Observed PD"),
        Line2D([0], [0], color=PAPER["blue"], lw=1.8, ls="--", label=f"Target PD ({cfg.pdk_target})"),
    ]
    legend_outside(ax, handles, side="right", pad=0.02)

    fig.tight_layout()
    save_fig(fig, out_png, out_pdf)


def plot_beta_posteriors_grade_over_time(
    out_long: pd.DataFrame,
    grade: int,
    cfg: BacktestConfig,
    out_png: Path,
    out_pdf: Path | None = None,
) -> None:
    df = out_long.copy()
    df["Quarter_str"] = df["Quarter"].astype(str)
    df["Grade_int"] = df["Grade"].astype(int)
    df = df[df["Grade_int"] == int(grade)].copy()
    if df.empty:
        return

    per = parse_vintage_to_period(df["Quarter_str"], freq="Q")
    df["_per"] = per
    df = df.sort_values("_per")

    df["Year"] = df["Quarter_str"].str.slice(0, 4)
    df = df[df["Year"].str.fullmatch(r"\d{4}", na=False)].copy()
    if df.empty:
        return
    df["Year"] = df["Year"].astype(int)

    pd0 = float(np.nanmedian(df["PD_k"].to_numpy(float)))
    pd0 = _clip01(pd0)

    n = df["n_k"].to_numpy(int)
    d = df["d_k"].to_numpy(int)
    a = d + 0.5
    b = (n - d) + 0.5

    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if not ok.any():
        return
    df_ok = df.loc[ok].copy()
    a_ok = a[ok]
    b_ok = b[ok]

    q_hi = min(0.9999, max(0.90, float(cfg.beta_quantile_span)))
    q_lo = 1.0 - q_hi
    lo = beta_dist.ppf(q_lo, a_ok, b_ok)
    hi = beta_dist.ppf(q_hi, a_ok, b_ok)
    x_min = float(max(0.0, np.nanmin(lo)))
    x_max = float(min(1.0, np.nanmax(hi)))

    if np.isfinite(pd0):
        x_min = float(min(x_min, max(0.0, pd0 * 0.5)))
        x_max = float(max(x_max, min(1.0, pd0 * 1.5)))

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x_min, x_max = 0.0, 1.0

    span = x_max - x_min
    x_min = max(0.0, x_min - 0.05 * span)
    x_max = min(1.0, x_max + 0.05 * span)

    x = np.linspace(x_min, x_max, int(cfg.beta_x_points))

    years = sorted(df_ok["Year"].unique().tolist())
    n_years = len(years)

    # Color-blind safe categorical palette (Okabe–Ito subset, reordered to
    # avoid clashing with the traffic-light foreground palette).
    OKABE_ITO_YEARS = [
        PAPER["blue"],  # same pastel blue as the traffic-light figures
        PAPER["red"],   # same pastel vermillion as the traffic-light figures
        "#009E73",      # bluish green
        "#CC79A7",      # reddish purple
        "#56B4E9",      # sky blue
        PAPER["amber"], # same pastel amber as the traffic-light figures
        "#F0E442",      # yellow
        "#000000",      # black
    ]
    if n_years <= len(OKABE_ITO_YEARS):
        colors = {yr: OKABE_ITO_YEARS[i] for i, yr in enumerate(years)}
    else:
        # Fallback for more than 8 years — modern API, no deprecation
        cmap_obj = mpl.colormaps["tab20"]
        colors = {yr: cmap_obj(i % cmap_obj.N) for i, yr in enumerate(years)}

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    paper_axes(ax)
    ax.grid(axis="y", color=PAPER["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    handles_year = []
    for yr in years:
        sub = df_ok[df_ok["Year"] == yr].copy()
        if sub.empty:
            continue
        col = colors[yr]

        dens_stack = []
        for _, r in sub.iterrows():
            ai = float(r["d_k"] + 0.5)
            bi = float(r["n_k"] - r["d_k"] + 0.5)
            y = beta_dist.pdf(x, ai, bi)
            dens_stack.append(y)

            ax.plot(x, y, color=col, alpha=cfg.beta_line_alpha,
                    linewidth=cfg.beta_quarter_linewidth, zorder=2)
            ax.fill_between(x, 0.0, y, color=col,
                            alpha=cfg.beta_fill_alpha, linewidth=0.0, zorder=1)

        if len(dens_stack) > 0:
            y_mean = np.nanmean(np.vstack(dens_stack), axis=0)
            ax.plot(x, y_mean, color=col, alpha=0.95,
                    linewidth=cfg.beta_year_linewidth, zorder=3)

        handles_year.append(Line2D([0], [0], color=col,
                                   lw=cfg.beta_year_linewidth, label=str(yr)))

    if np.isfinite(pd0):
        ax.axvline(pd0, color=PAPER["axis"], linewidth=1.8, linestyle="--", zorder=4)

    ax.set_xlabel("p", color=PAPER["axis"])
    ax.set_ylabel("density", color=PAPER["axis"])

    handles = handles_year.copy()
    if np.isfinite(pd0):
        handles.append(Line2D([0], [0], color=PAPER["axis"], lw=1.8, ls="--",
                              label=f"Target PD ({cfg.pdk_target})"))

    legend_outside(ax, handles, side="right", pad=0.02, fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_png, out_pdf)


# =============================================================================
# Traffic light logic
# =============================================================================

def tl_from_pval(pval_H0: float, alpha: float, amber: float) -> str:
    if not np.isfinite(pval_H0):
        return "na"
    if pval_H0 < alpha:
        return "red"
    if pval_H0 < amber:
        return "amber"
    return "blue"


def tl_from_decision(prob_under: float, es: float, cfg: BacktestConfig) -> str:
    if not (np.isfinite(prob_under) and np.isfinite(es)):
        return "na"
    if (prob_under >= cfg.tl_prob_red) and (es >= cfg.tl_es_red):
        return "red"
    if (prob_under >= cfg.tl_prob_amber and es >= cfg.tl_es_amber) or (prob_under >= cfg.tl_prob_red):
        return "amber"
    return "blue"


def prep_tl_matrix(out_long: pd.DataFrame, cfg: BacktestConfig, mode: str) -> pd.DataFrame:
    tmp = out_long.copy()
    tmp["Quarter"] = tmp["Quarter"].astype(str)
    tmp["Grade"] = tmp["Grade"].astype(int)

    per = parse_vintage_to_period(tmp["Quarter"], freq="Q")
    order = (
        tmp[["Quarter"]]
        .assign(_per=per)
        .drop_duplicates()
        .sort_values("_per")["Quarter"]
        .astype(str)
        .tolist()
    )
    tmp["Quarter"] = pd.Categorical(tmp["Quarter"], categories=order, ordered=True)

    alpha_two_sided = 1.0 - float(cfg.conf_level)

    if mode == "pval":
        tmp["tl"] = tmp["Jeffreys_pval_H0"].apply(lambda p: tl_from_pval(p, alpha_two_sided, cfg.tl_pval_amber))
    elif mode == "decision":
        tmp["tl"] = tmp.apply(lambda r: tl_from_decision(r["Jeffreys_prob_under"], r["Jeffreys_ES"], cfg), axis=1)
    else:
        raise ValueError("mode must be 'pval' or 'decision'")

    return tmp.pivot(index="Grade", columns="Quarter", values="tl").sort_index()


def choose_focus_year(out: pd.DataFrame, cfg: BacktestConfig) -> int | None:
    df = out.copy()
    df["Year"] = df["Quarter"].astype(str).str.slice(0, 4)
    df = df[df["Year"].str.fullmatch(r"\d{4}", na=False)].copy()
    if df.empty:
        return None
    df["Year"] = df["Year"].astype(int)

    mat = prep_tl_matrix(df, cfg, mode="decision")
    q_share = {}
    for q in mat.columns:
        col = mat[q]
        tot = col.notna().sum()
        q_share[q] = float((col == "red").sum()) / tot if tot > 0 else 0.0
    s = pd.Series(q_share, name="share_red").reset_index().rename(columns={"index": "Quarter"})
    s["Year"] = s["Quarter"].apply(infer_year_from_quarter)
    s = s.dropna(subset=["Year"])
    if s.empty:
        return None
    best_year = int(s.groupby("Year")["share_red"].mean().sort_values(ascending=False).index[0])
    return best_year


# =============================================================================
# Main entry point
# =============================================================================

def run_oos_backtest(cfg: BacktestConfig) -> None:
    apply_paper_rcparams()

    if not cfg.oos_path.exists():
        raise FileNotFoundError(f"OOS file not found: {cfg.oos_path}")
    if not cfg.bucket_stats_path.exists():
        raise FileNotFoundError(f"bucket_stats.json not found: {cfg.bucket_stats_path}")

    out_dir = cfg.out_dir
    tables_dir = out_dir / "tables"
    heatmap_dir = out_dir / "heatmaps_by_quarter"
    tl_dir = out_dir / "traffic_light"
    sev_dir = out_dir / "severity_figures"
    pd_dir = out_dir / "pd_evolution"
    beta_dir = out_dir / "beta_posteriors"

    for d in [tables_dir, heatmap_dir, tl_dir, sev_dir, pd_dir, beta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df_oos = pd.read_parquet(cfg.oos_path)
    df_vg = build_vintage_grade_table(df_oos, cfg)

    if cfg.pdk_target not in df_vg.columns or df_vg[cfg.pdk_target].isna().all():
        raise ValueError(f"pdk_target='{cfg.pdk_target}' not available. Try 'pd_ttc' or 'pd_hat'.")

    df_vg = add_all_stats(df_vg, cfg, pd_col=cfg.pdk_target)

    out = df_vg[
        [
            cfg.vintage_col,
            cfg.grade_col,
            "n",
            "d",
            "pd_obs",
            cfg.pdk_target,
            "Jeffreys_lb",
            "Jeffreys_ub",
            "Jeffreys_CI",
            "Clopper-Pearson_CI",
            "Normal Approx._CI",
            "Jeffreys_reject",
            "Clopper-Pearson_reject",
            "Normal Approx._reject",
            "Jeffreys_pval_H0",
            "Jeffreys_prob_under",
            "Jeffreys_ES",
            "Jeffreys_q95_excess",
        ]
    ].copy()

    out = out.rename(
        columns={
            cfg.vintage_col: "Quarter",
            cfg.grade_col: "Grade",
            "n": "n_k",
            "d": "d_k",
            "pd_obs": "p_hat_k",
            cfg.pdk_target: "PD_k",
        }
    )

    # Save full table
    out_csv = tables_dir / "oos_backtest_full_with_severity.csv"
    out_pq = tables_dir / "oos_backtest_full_with_severity.parquet"
    out.to_csv(out_csv, index=False)
    out.to_parquet(out_pq, index=False)
    print("[OK] wrote:", out_csv)
    print("[OK] wrote:", out_pq)

    # Per-quarter tables + heatmaps
    quarters = out["Quarter"].dropna().unique().tolist()
    for q in quarters:
        df_q = out.loc[out["Quarter"] == q].copy()
        df_q = df_q.sort_values("Grade").set_index("Grade")

        q_tag = _safe_fname(q)
        (tables_dir / f"oos_backtest_{q_tag}.csv").write_text(df_q.to_csv(index=True), encoding="utf-8")
        df_q.to_parquet(tables_dir / f"oos_backtest_{q_tag}.parquet", index=True)

        png_path = heatmap_dir / f"oos_binary_{q_tag}.png"
        pdf_path = (heatmap_dir / f"oos_binary_{q_tag}.pdf") if cfg.save_pdf else None
        plot_binary_heatmap_for_quarter(df_q, out_png=png_path, out_pdf=pdf_path, annotate=True)

    print("[OK] per-quarter tables:", tables_dir)
    print("[OK] per-quarter heatmaps:", heatmap_dir)

    # Traffic lights
    tl_pval = prep_tl_matrix(out, cfg, mode="pval")
    tl_dec = prep_tl_matrix(out, cfg, mode="decision")

    tl_pval.to_csv(tl_dir / "traffic_light_matrix_pval.csv")
    tl_pval.to_parquet(tl_dir / "traffic_light_matrix_pval.parquet")
    tl_dec.to_csv(tl_dir / "traffic_light_matrix_decision.csv")
    tl_dec.to_parquet(tl_dir / "traffic_light_matrix_decision.parquet")

    if cfg.tl_main_mode == "pval":
        mat = tl_pval
        stem = "traffic_light_main_pval"
    elif cfg.tl_main_mode == "decision":
        mat = tl_dec
        stem = "traffic_light_main_decision"
    else:
        raise ValueError("tl_main_mode must be 'pval' or 'decision'")

    plot_traffic_light_matrix(
        mat,
        out_png=tl_dir / f"{stem}.png",
        out_pdf=(tl_dir / f"{stem}.pdf") if cfg.save_pdf else None,
    )

    sum_pval = plot_traffic_light_summary_by_quarter(
        tl_pval,
        out_png=tl_dir / "traffic_light_summary_pval.png",
        out_pdf=(tl_dir / "traffic_light_summary_pval.pdf") if cfg.save_pdf else None,
    )
    sum_dec = plot_traffic_light_summary_by_quarter(
        tl_dec,
        out_png=tl_dir / "traffic_light_summary_decision.png",
        out_pdf=(tl_dir / "traffic_light_summary_decision.pdf") if cfg.save_pdf else None,
    )
    sum_pval.to_csv(tl_dir / "traffic_light_summary_pval.csv")
    sum_dec.to_csv(tl_dir / "traffic_light_summary_decision.csv")
    print("[OK] traffic light outputs:", tl_dir)

    # Severity figures (focus year)
    focus = cfg.focus_year if cfg.focus_year is not None else choose_focus_year(out, cfg)
    if focus is not None:
        plot_severity_map_year(
            out,
            focus,
            out_png=sev_dir / f"severity_map_{focus}.png",
            out_pdf=(sev_dir / f"severity_map_{focus}.pdf") if cfg.save_pdf else None,
        )
        plot_evidence_severity_plane_year(
            out,
            focus,
            cfg,
            out_png=sev_dir / f"evidence_severity_plane_{focus}.png",
            out_pdf=(sev_dir / f"evidence_severity_plane_{focus}.pdf") if cfg.save_pdf else None,
        )
        print(f"[OK] severity figures for year={focus}:", sev_dir)
    else:
        print("[WARN] could not infer a focus year; skipped severity plots.")

    # PD evolution by grade
    grades = sorted(out["Grade"].dropna().astype(int).unique().tolist())
    if cfg.pd_evolution_max_grades is not None:
        grades = grades[: int(cfg.pd_evolution_max_grades)]

    for g in grades:
        plot_pd_evolution_per_grade(
            out_long=out,
            grade=int(g),
            cfg=cfg,
            out_png=pd_dir / f"pd_evolution_grade_{int(g)}.png",
            out_pdf=(pd_dir / f"pd_evolution_grade_{int(g)}.pdf") if cfg.save_pdf else None,
        )
    print("[OK] PD evolution figures:", pd_dir)

    # Beta posterior densities (one grade)
    plot_beta_posteriors_grade_over_time(
        out_long=out,
        grade=int(cfg.beta_grade),
        cfg=cfg,
        out_png=beta_dir / f"beta_posterior_grade_{int(cfg.beta_grade)}.png",
        out_pdf=(beta_dir / f"beta_posterior_grade_{int(cfg.beta_grade)}.pdf") if cfg.save_pdf else None,
    )
    print("[OK] beta posterior plot:", beta_dir)

    # Paper LaTeX snapshot table
    try:
        tex = render_snapshot_table_latex(
            out_long=out,
            quarter=cfg.paper_table_snapshot,
            units=cfg.paper_table_units,
            alpha=cfg.paper_table_alpha,
            include_counts=cfg.paper_table_include_counts,
            caption=cfg.paper_table_caption,
            label=cfg.paper_table_label,
        )
        tex_path = tables_dir / f"{cfg.paper_table_stem}.tex"
        tex_path.write_text(tex, encoding="utf-8")
        print("[OK] wrote LaTeX table:", tex_path)
    except Exception as e:
        print("[WARN] could not write LaTeX snapshot table:", repr(e))

    alpha_two_sided = 1.0 - float(cfg.conf_level)
    print("[INFO] thresholds:")
    print(f"  pval TL: alpha(two-sided)={alpha_two_sided:.2f}, amber<{cfg.tl_pval_amber:.2f}")
    print(
        f"  decision TL: prob_red={cfg.tl_prob_red:.2f}, prob_amber={cfg.tl_prob_amber:.2f}, "
        f"ES_red={cfg.tl_es_red*1e4:.1f}bps, ES_amber={cfg.tl_es_amber*1e4:.1f}bps"
    )
    print(f"  tl_main_mode={cfg.tl_main_mode}")
    print(f"[DONE] {datetime.now().isoformat(timespec='seconds')}")