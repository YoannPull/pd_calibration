#!/usr/bin/env python3
# src/generate_vintage_grade_report.py
# -*- coding: utf-8 -*-

"""
generate_vintage_grade_report.py
================================

Vintage / Grade evolution report from a scored dataset.

This script analyzes how *grades* evolve across vintages (or any other segment)
using a scored file. It produces a self-contained HTML report (figures embedded
as base64 PNGs) with volume tables, stacked distributions, and optional scoring
metrics/calibration diagnostics.

Grade convention (aligned with the training pipeline)
-----------------------------------------------------
- grade = 1 : least risky class
- grade = N : most risky class
Therefore, PDs are expected to be increasing with the grade index.

Input
-----
A parquet/csv scored file containing at least:
- a vintage/segment column (e.g., "vintage")
- a grade column (e.g., "grade")

Optionally:
- a model PD column (e.g., "pd")
- a binary target column (e.g., "default_24m")

TTC reference (master scale)
----------------------------
Optionally, the report can use a grade-level TTC reference PD derived from
bucket_stats.json produced at training time:
- stats["train"] section, with a "pd" value for each "bucket"/grade.

Output
------
An HTML report including:
- global scoring metrics (if both PD and target are available)
- volume tables by (vintage, grade): counts and percentages
- stacked bar chart: grade distribution by vintage
- optional: mean PD by (vintage, grade)
- optional: observed default rate by vintage
- calibration tables by (vintage, grade) comparing:
    * pd_hat (mean model PD) vs pd_ttc (master-scale PD) and/or pd_obs (observed DR)
- a monotonicity comment across grades for pd_ttc / pd_hat / pd_obs
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # Batch rendering backend (no GUI required)
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401 (kept for consistency / possible extensions)

from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


# ---------------------------------------------------------------------------
# Dark theme (kept consistent with the other report)
# ---------------------------------------------------------------------------

DARK_BG = "#1E1E1E"
DARK_PANEL = "#2A2A2A"
BLUE = "#4EA8FF"
YELLOW = "#FFDD57"
GREY = "#ABB2BF"

plt.rcParams.update(
    {
        "axes.facecolor": DARK_PANEL,
        "figure.facecolor": DARK_BG,
        "savefig.facecolor": DARK_BG,
        "text.color": GREY,
        "axes.labelcolor": GREY,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "axes.edgecolor": GREY,
        "grid.color": "#555555",
    }
)

# =============================================================================
# Helpers
# =============================================================================


def load_any(path: str) -> pd.DataFrame:
    """
    Load a dataset from disk.

    Supported formats:
    - Parquet (.parquet, .pq)
    - CSV (fallback)
    """
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def fig_to_base64(fig) -> str:
    """
    Convert a matplotlib figure into an embedded base64 PNG string.

    The output can be inserted directly in HTML:
        <img src="data:image/png;base64,...">
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def _sort_grades_like_numbers(cols):
    """
    Sort grade labels numerically if possible (1, 2, 3, ..., 10), otherwise
    fallback to lexicographic sorting.

    This preserves the intended ordering: grade 1 (least risky) -> grade N (most risky).
    """
    try:
        return sorted(cols, key=lambda x: float(x))
    except Exception:
        return sorted(cols)


# =============================================================================
# Load TTC/master-scale PD from bucket_stats.json
# =============================================================================


def load_pd_ttc_from_master_scale(bucket_stats_path: Path):
    """
    Read bucket_stats.json (generated at training) and return a dict:
        {grade/bucket -> PD_TTC}
    based on the "train" section.

    Returns None if the file is missing or has an unexpected structure.
    """
    if bucket_stats_path is None:
        print("[WARN] No bucket_stats.json path provided; pd_ttc will be empty.", file=sys.stderr)
        return None

    if not bucket_stats_path.exists():
        print(
            f"[WARN] bucket_stats.json not found at {bucket_stats_path}. "
            "pd_ttc will be empty in the report.",
            file=sys.stderr,
        )
        return None

    stats = json.loads(bucket_stats_path.read_text())
    train_stats = stats.get("train", [])
    if not train_stats:
        print("[WARN] Missing or empty 'train' section in bucket_stats.json.", file=sys.stderr)
        return None

    df_train = pd.DataFrame(train_stats)
    if "bucket" not in df_train.columns or "pd" not in df_train.columns:
        print("[WARN] Missing 'bucket'/'pd' keys in bucket_stats.json.", file=sys.stderr)
        return None

    return df_train.set_index("bucket")["pd"].to_dict()


# =============================================================================
# Calibration error (ECE)
# =============================================================================


def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    """
    Compute Expected Calibration Error (ECE) using quantile bins.

    ECE = sum_b |obs_b - pred_b| * (n_b / n)
    where bins are formed using qcut (roughly equal-frequency).
    """
    df = pd.DataFrame({"y": y_true, "pred": y_prob})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")

    grp = (
        df.groupby("bin")
        .agg(count=("y", "size"), obs=("y", "mean"), pred=("pred", "mean"))
        .dropna()
    )

    total = grp["count"].sum()
    ece = float(np.sum(np.abs(grp["obs"] - grp["pred"]) * (grp["count"] / total)))
    return ece


# =============================================================================
# 1) Volume tables by (vintage, grade)
# =============================================================================


def build_volume_tables(df, vintage_col, grade_col):
    """
    Build two pivot tables:
    - pivot_count: counts by (vintage x grade)
    - pivot_pct: row-wise percentages (each vintage sums to 100%)
    """
    tab_count = (
        df.groupby([vintage_col, grade_col])
        .size()
        .rename("count")
        .reset_index()
    )

    pivot_count = (
        tab_count.pivot(index=vintage_col, columns=grade_col, values="count")
        .fillna(0)
        .astype(int)
    )
    pivot_pct = pivot_count.div(pivot_count.sum(axis=1), axis=0) * 100

    # Enforce grade ordering: 1 -> N (numeric if possible)
    try:
        sorted_cols = _sort_grades_like_numbers(pivot_count.columns)
        pivot_count = pivot_count.reindex(columns=sorted_cols)
        pivot_pct = pivot_pct.reindex(columns=sorted_cols)
    except Exception:
        pass

    return pivot_count, pivot_pct


# =============================================================================
# 2) Stacked bar: grade distribution by vintage
# =============================================================================


def plot_grade_distribution(pivot_pct: pd.DataFrame, vintage_col: str, grade_name: str) -> str:
    """
    Stacked bar chart of grade shares (in %) by vintage.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    pivot_pct = pivot_pct.sort_index()
    bottoms = np.zeros(len(pivot_pct))
    x = np.arange(len(pivot_pct.index))

    # Ensure grade columns are ordered as 1 -> N
    cols = _sort_grades_like_numbers(pivot_pct.columns)

    for g in cols:
        vals = pivot_pct[g].values
        ax.bar(x, vals, bottom=bottoms, label=f"{grade_name} {g}")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(pivot_pct.index, rotation=45, ha="right")
    ax.set_ylabel("Volume (%)")
    ax.set_xlabel(vintage_col)
    ax.set_title("Grade distribution by vintage")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(ncol=4, fontsize=8)

    return fig_to_base64(fig)


# =============================================================================
# 3) Mean PD by (vintage, grade) (optional)
# =============================================================================


def plot_pd_by_vintage_and_grade(df, vintage_col, grade_col, pd_col) -> str:
    """
    Plot the mean model PD per (vintage, grade).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    grp = (
        df.groupby([vintage_col, grade_col])[pd_col]
        .mean()
        .reset_index()
    )

    for g in _sort_grades_like_numbers(grp[grade_col].unique()):
        sub = grp[grp[grade_col] == g].copy().sort_values(vintage_col)
        ax.plot(sub[vintage_col], sub[pd_col], marker="o", label=f"Grade {g}")

    ax.set_xlabel(vintage_col)
    ax.set_ylabel(f"Mean PD ({pd_col})")
    ax.set_title("Mean PD by vintage and grade")
    ax.grid(True, alpha=0.2)
    ax.legend(ncol=3, fontsize=8)
    plt.xticks(rotation=45, ha="right")

    return fig_to_base64(fig)


# =============================================================================
# 4) Observed default rate by vintage (optional)
# =============================================================================


def plot_dr_by_vintage(df, vintage_col, target_col) -> str:
    """
    Plot the observed default rate by vintage.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    grp = (
        df.groupby(vintage_col)[target_col]
        .mean()
        .reset_index()
        .sort_values(vintage_col)
    )

    ax.plot(grp[vintage_col], grp[target_col], marker="o", color=BLUE)
    ax.set_xlabel(vintage_col)
    ax.set_ylabel(f"Default rate ({target_col})")
    ax.set_title("Observed default rate by vintage")
    ax.grid(True, alpha=0.2)
    plt.xticks(rotation=45, ha="right")

    return fig_to_base64(fig)


# =============================================================================
# 5) Global scoring metrics (optional)
# =============================================================================


def compute_global_metrics(df, target_col, pd_col) -> dict:
    """
    Compute global discrimination and calibration metrics on the full dataset.

    Requires both a binary target and a predicted PD column.
    """
    y_true = df[target_col].astype(int).values
    y_prob = df[pd_col].astype(float).values

    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    ll = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)

    return {
        "AUC": auc,
        "Gini": gini,
        "LogLoss": ll,
        "Brier": brier,
        "ECE": ece,
        "N": int(len(y_true)),
        "Default Rate": float(y_true.mean()),
    }


# =============================================================================
# 6) Calibration table vs TTC master scale (optional)
# =============================================================================


def build_calibration_table(
    df,
    vintage_col,
    grade_col,
    pd_col=None,
    target_col=None,
    pd_ttc_map=None,
):
    """
    Build a calibration table by (vintage, grade), comparing:
    - n: volume
    - n_defaults: observed default count (if target available)
    - pd_hat: mean model PD (if pd_col available)
    - pd_obs: observed default rate (if target available)
    - pd_ttc: TTC/master-scale PD for the grade (if pd_ttc_map available)

    PD_TTC_master(grade) comes from bucket_stats.json ("train" section).
    """
    calib = (
        df.groupby([vintage_col, grade_col])
        .size()
        .rename("n")
        .reset_index()
    )

    if pd_col is not None and pd_col in df.columns:
        pd_hat = (
            df.groupby([vintage_col, grade_col])[pd_col]
            .mean()
            .reset_index()
            .rename(columns={pd_col: "pd_hat"})
        )
        calib = calib.merge(pd_hat, on=[vintage_col, grade_col], how="left")

    if target_col is not None and target_col in df.columns:
        agg = (
            df.groupby([vintage_col, grade_col])[target_col]
            .agg(pd_obs="mean", n_defaults="sum")
            .reset_index()
        )
        calib = calib.merge(agg, on=[vintage_col, grade_col], how="left")

    if pd_ttc_map is not None:
        master = pd.DataFrame(
            {
                grade_col: list(pd_ttc_map.keys()),
                "pd_ttc": list(pd_ttc_map.values()),
            }
        )
        calib = calib.merge(master, on=grade_col, how="left")
    else:
        calib["pd_ttc"] = np.nan

    return calib


def build_calibration_tables_by_vintage_html(calib_df: pd.DataFrame, vintage_col: str, grade_col: str) -> str:
    """
    Build an HTML block with one calibration table per vintage.

    For readability:
    - the vintage column is removed inside each table
    - grades are sorted 1 -> N
    - n and n_defaults are placed next to each other when available
    """
    if calib_df.empty:
        return "<p>No calibration information available.</p>"

    parts = []
    for v in sorted(calib_df[vintage_col].unique()):
        sub = calib_df[calib_df[vintage_col] == v].copy()

        try:
            sub[grade_col] = sub[grade_col].astype(float)
            sub = sub.sort_values(by=grade_col)
        except Exception:
            sub = sub.sort_values(by=grade_col)

        if vintage_col in sub.columns:
            sub = sub.drop(columns=[vintage_col])

        preferred_order = [grade_col, "n", "n_defaults", "pd_hat", "pd_obs", "pd_ttc"]
        cols = [c for c in preferred_order if c in sub.columns]
        others = [c for c in sub.columns if c not in cols]
        sub = sub[cols + others]

        parts.append(f"<h4>Vintage = {v}</h4>")
        parts.append(sub.round(6).to_html(index=False))

    return "\n".join(parts)


# =============================================================================
# 7) Monotonicity comment across grades
# =============================================================================


def build_monotonicity_comment(calib_df: pd.DataFrame, grade_col: str) -> str:
    """
    Check monotonicity across grades (aggregated over all vintages) for:
    - pd_ttc (TTC reference)
    - pd_hat (model PD)
    - pd_obs (observed default rate)

    Convention: grade 1 (least risky) -> grade N (most risky), so we check that
    PD increases as grade increases.
    """
    if calib_df.empty or grade_col not in calib_df.columns:
        return "<p>Monotonicity: not evaluated (insufficient data).</p>"

    agg = calib_df.groupby(grade_col)[["pd_ttc", "pd_hat", "pd_obs"]].mean()

    try:
        idx_sorted = sorted(agg.index, key=lambda x: float(x))
        agg = agg.loc[idx_sorted]
    except Exception:
        agg = agg.sort_index()

    msgs = []
    for col, label in [
        ("pd_ttc", "TTC PD (pd_ttc)"),
        ("pd_hat", "Model PD (pd_hat)"),
        ("pd_obs", "Observed PD (pd_obs)"),
    ]:
        if col not in agg.columns or agg[col].isna().all():
            continue
        vals = agg[col].values
        if len(vals) <= 1:
            continue

        diffs = np.diff(vals)
        if np.all(diffs >= -1e-8):
            msgs.append(f"{label}: monotonicity respected.")
        else:
            msgs.append(f"{label}: <strong>monotonicity violated</strong>.")

    if not msgs:
        return "<p>Monotonicity: not evaluated (no usable PD columns).</p>"

    return (
        "<p><strong>Monotonicity comment (by grade, aggregated over vintages):</strong><br>"
        + "<br>".join(msgs)
        + "</p>"
    )


# =============================================================================
# HTML
# =============================================================================


def build_html(
    out_path: Path,
    vintage_col: str,
    grade_col: str,
    pivot_count: pd.DataFrame,
    pivot_pct: pd.DataFrame,
    img_dist: str,
    img_pd_by_grade: str | None,
    img_dr: str | None,
    metrics: dict | None,
    calib_tables_html: str,
    mono_comment_html: str,
    sample_name: str,
    vintage_min: str,
    vintage_max: str,
):
    """
    Write the final HTML report.

    The report is self-contained: all images are embedded as base64 strings.
    """
    # Global metrics table (if available)
    if metrics is not None:
        metrics_rows = "\n".join(
            f"<tr><td>{k}</td><td>{(v if k=='N' else round(v, 6))}</td></tr>"
            for k, v in metrics.items()
        )
        metrics_html = f"""
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {metrics_rows}
        </table>
        """
    else:
        metrics_html = "<p>No metrics available (missing PD and/or target).</p>"

    legend_html = """
    <p><strong>Legend (calibration tables):</strong></p>
    <ul>
        <li><code>grade</code>: risk class (master scale), with 1 = least risky and N = most risky.</li>
        <li><code>n</code>: number of exposures in the (vintage, grade) cell.</li>
        <li><code>n_defaults</code>: observed defaults in that cell (if target available).</li>
        <li><code>pd_hat</code>: mean model PD in that cell (mean of the PD column).</li>
        <li><code>pd_obs</code>: observed default rate in that cell (if target available).</li>
        <li><code>pd_ttc</code>: TTC/master-scale PD for that grade (from bucket_stats.json, train section).</li>
    </ul>
    """

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Vintage / Grade Report – {sample_name} ({vintage_min} → {vintage_max})</title>
<style>
    body {{
        background-color: #1E1E1E;
        font-family: Arial, sans-serif;
        color: #D0D0D0;
        margin: 0;
        padding: 0;
    }}
    .container {{
        max-width: 1200px;
        margin: auto;
        padding: 30px;
        background-color: #1E1E1E;
    }}
    h1, h2, h3, h4 {{
        color: #E8E8E8;
    }}
    .section {{
        margin-top: 30px;
        padding: 20px;
        background-color: #2A2A2A;
        border-radius: 8px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        color: #E8E8E8;
        font-size: 13px;
    }}
    th {{
        background-color: #333333;
        padding: 6px;
        text-align: center;
    }}
    td {{
        background-color: #2E2E2E;
        padding: 6px;
        text-align: right;
    }}
    img {{
        width: 100%;
        border-radius: 6px;
        margin-top: 15px;
    }}
</style>
</head>
<body>
<div class="container">

    <h1>Vintage / Grade Report – {sample_name}</h1>
    <p><em>Covered vintages:</em> <code>{vintage_min} → {vintage_max}</code></p>

    <div class="section">
        <h2>0. Global scoring metrics</h2>
        {metrics_html}
    </div>

    <div class="section">
        <h2>1. Volumes by {vintage_col} and {grade_col}</h2>
        <h3>Counts (n)</h3>
        {pivot_count.to_html(classes="", border=0)}

        <h3>Shares (%)</h3>
        {pivot_pct.round(2).to_html(classes="", border=0)}
    </div>

    <div class="section">
        <h2>2. Grade distribution by vintage</h2>
        <img src="{img_dist}">
    </div>
"""

    if img_pd_by_grade is not None:
        html += f"""
    <div class="section">
        <h2>3. Mean PD by vintage and grade</h2>
        <img src="{img_pd_by_grade}">
    </div>
"""

    if img_dr is not None:
        html += f"""
    <div class="section">
        <h2>4. Observed default rate by vintage</h2>
        <img src="{img_dr}">
    </div>
"""

    html += f"""
    <div class="section">
        <h2>5. Calibration by (vintage, grade) vs TTC/master-scale PD</h2>
        <p>PD_TTC_master(grade) is the TTC PD for each grade from the master scale (bucket_stats.json).</p>
        {legend_html}
        {mono_comment_html}
        {calib_tables_html}
    </div>

</div>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


# =============================================================================
# Main
# =============================================================================


def parse_args():
    """Define and parse CLI arguments."""
    p = argparse.ArgumentParser(description="Vintage/Grade evolution + calibration report from a scored file.")
    p.add_argument("--data", required=True, help="Scored input file (parquet/csv).")
    p.add_argument("--out", required=True, help="Output HTML report path.")
    p.add_argument("--vintage-col", default="vintage", help="Vintage/segment column name")
    p.add_argument("--grade-col", default="grade", help="Grade column name")
    p.add_argument("--pd-col", default="pd", help="Model PD column name (optional).")
    p.add_argument("--target", default=None, help="Binary target column name (optional).")
    p.add_argument("--sample-name", default="OOS", help="Sample label used in the report title.")
    p.add_argument("--bucket-stats", default=None, help="Path to bucket_stats.json (TTC master scale).")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_any(args.data)

    if args.vintage_col not in df.columns:
        raise ValueError(f"Missing vintage column '{args.vintage_col}'.")
    if args.grade_col not in df.columns:
        raise ValueError(f"Missing grade column '{args.grade_col}'.")

    # Force vintage to string for display and stable ordering in tables/plots
    df[args.vintage_col] = df[args.vintage_col].astype(str)

    vintages_sorted = sorted(df[args.vintage_col].unique())
    vintage_min = vintages_sorted[0]
    vintage_max = vintages_sorted[-1]

    # 1) Tables: volumes and shares
    pivot_count, pivot_pct = build_volume_tables(df, args.vintage_col, args.grade_col)

    # 2) Stacked distribution plot
    img_dist = plot_grade_distribution(pivot_pct, args.vintage_col, args.grade_col)

    # 3) Mean PD by (vintage, grade) if available
    img_pd_by_grade = None
    if args.pd_col is not None and args.pd_col in df.columns:
        img_pd_by_grade = plot_pd_by_vintage_and_grade(df, args.vintage_col, args.grade_col, args.pd_col)

    # 4) Observed default rate by vintage if a target is available
    img_dr = None
    if args.target is not None and args.target in df.columns:
        img_dr = plot_dr_by_vintage(df, args.vintage_col, args.target)

    # 5) Global metrics if both PD and target are available
    metrics = None
    if (
        args.target is not None
        and args.target in df.columns
        and args.pd_col is not None
        and args.pd_col in df.columns
    ):
        metrics = compute_global_metrics(df, args.target, args.pd_col)

    # 6) Load TTC/master-scale PDs if provided
    if args.bucket_stats is not None:
        pd_ttc_map = load_pd_ttc_from_master_scale(Path(args.bucket_stats))
    else:
        print("[WARN] --bucket-stats not provided; pd_ttc will be empty in tables.", file=sys.stderr)
        pd_ttc_map = None

    # 7) Calibration table vs TTC/master scale
    calib_df = build_calibration_table(
        df,
        vintage_col=args.vintage_col,
        grade_col=args.grade_col,
        pd_col=args.pd_col if (args.pd_col and args.pd_col in df.columns) else None,
        target_col=args.target if (args.target and args.target in df.columns) else None,
        pd_ttc_map=pd_ttc_map,
    )
    calib_tables_html = build_calibration_tables_by_vintage_html(calib_df, args.vintage_col, args.grade_col)

    # Monotonicity comment across grades
    mono_comment_html = build_monotonicity_comment(calib_df, args.grade_col)

    # 8) Write HTML
    out_path = Path(args.out)
    build_html(
        out_path=out_path,
        vintage_col=args.vintage_col,
        grade_col=args.grade_col,
        pivot_count=pivot_count,
        pivot_pct=pivot_pct,
        img_dist=img_dist,
        img_pd_by_grade=img_pd_by_grade,
        img_dr=img_dr,
        metrics=metrics,
        calib_tables_html=calib_tables_html,
        mono_comment_html=mono_comment_html,
        sample_name=args.sample_name,
        vintage_min=vintage_min,
        vintage_max=vintage_max,
    )

    print(f"✔ Vintage/Grade + calibration report written to: {out_path}")


if __name__ == "__main__":
    main()
