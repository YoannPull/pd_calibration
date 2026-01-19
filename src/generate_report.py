#!/usr/bin/env python3
# src/generate_report.py
# -*- coding: utf-8 -*-

"""
generate_report.py — Dark report (Train vs Validation) with grade-level breakdowns
-------------------------------------------------------------------------------

This script generates a complete banking-style validation report comparing the
TRAIN and VALIDATION samples. The report is exported as a single self-contained
HTML file (all figures are embedded as base64 PNGs).

Main additions in this version
------------------------------
- Global default rate summary (Train / Validation): N, #defaults, default rate.
- Grade-level tables:
    * observed default rate (pd_obs)
    * share of defaults by grade (%_defauts)
    * share of exposures by grade (%_individus)
- TTC tables by grade using a master-scale PD (from bucket_stats.json), when available.

Grade convention (aligned with the training pipeline)
-----------------------------------------------------
- grade = 1 : least risky bucket
- grade = N : most risky bucket
Therefore, the expected pattern is that PDs increase with the grade index.

Inputs
------
- train/validation datasets (parquet expected in this script version)
- a saved model package (joblib) containing LR coefficients and feature names
- optionally: bucket_stats.json (to fetch the TTC/master-scale PD by grade)

Output
------
- A single HTML report (dark theme), including:
    * Global metrics table
    * ROC curve
    * Calibration curve
    * Score distribution
    * Master scale plot + grade tables + TTC tables
    * Coefficient plot + coefficient table
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for batch/CLI rendering
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

# =============================================================================
# Dark theme configuration
# =============================================================================

DARK_BG = "#1E1E1E"
DARK_PANEL = "#2A2A2A"
BLUE = "#4EA8FF"
CYAN = "#7DE2D1"
MAGENTA = "#D16EE0"
YELLOW = "#FFDD57"
RED = "#E06C75"
GREEN = "#98C379"
GREY = "#ABB2BF"

# Apply global matplotlib styling (used by all figures)
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


def fig_to_base64(fig) -> str:
    """
    Convert a matplotlib figure into an embedded base64 PNG string.

    The output can be used directly in HTML as:
        <img src="data:image/png;base64,....">
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def _sort_grades_like_numbers(values):
    """
    Sort grades numerically if possible (1, 2, 3, ..., 10); otherwise fallback
    to lexicographic sorting.

    This keeps the intended convention: grade 1 (least risky) -> grade N (most risky).
    """
    try:
        return sorted(values, key=lambda x: float(x))
    except Exception:
        return sorted(values)


def add_global_default_info(metrics: dict, y: np.ndarray) -> dict:
    """
    Add global sample information to a metric dictionary.

    Adds:
    - n: number of observations
    - n_defaults: number of defaults
    - default_rate: observed default frequency
    """
    y = np.asarray(y).astype(int)
    n = int(len(y))
    n_def = int(y.sum())
    dr = float(n_def / n) if n > 0 else float("nan")

    metrics = dict(metrics)
    metrics["n"] = n
    metrics["n_defaults"] = n_def
    metrics["default_rate"] = dr
    return metrics


def add_grade_default_shares(gdf: pd.DataFrame, total_n: int, total_bad: int) -> pd.DataFrame:
    """
    Add exposure and default shares to a grade-level aggregation.

    Requires:
      - count: grade exposure count
      - bad: grade default count

    Adds:
      - pct_individus: count / total_n
      - pct_defauts: bad / total_bad
    """
    out = gdf.copy()
    out["pct_individus"] = (out["count"] / total_n) if total_n > 0 else np.nan
    out["pct_defauts"] = (out["bad"] / total_bad) if total_bad > 0 else np.nan
    return out


def html_table_with_percentages(df: pd.DataFrame, pct_cols=None, round_cols=6) -> str:
    """
    Render a dataframe as HTML, converting selected columns from proportions to percentages.

    Parameters
    ----------
    df : pd.DataFrame
        Table to render.
    pct_cols : list[str], optional
        Columns that should be multiplied by 100 for display.
    round_cols : int
        Number of decimals for rounding.
    """
    if pct_cols is None:
        pct_cols = []
    d = df.copy()
    for c in pct_cols:
        if c in d.columns:
            d[c] = 100.0 * d[c]
    return d.round(round_cols).to_html(index=False)


# =============================================================================
# Calibration error (ECE)
# =============================================================================


def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    """
    Compute the Expected Calibration Error (ECE) using quantile bins.

    We bin predictions into (approx.) equal-frequency groups using qcut and then
    compute:
        ECE = sum_b |obs_b - pred_b| * (n_b / n)
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
# ROC curve
# =============================================================================


def plot_roc(train_y, train_pd, val_y, val_pd):
    """Overlay ROC curves for Train and Validation and return the embedded image."""
    fig, ax = plt.subplots(figsize=(7, 5))

    fpr_t, tpr_t, _ = roc_curve(train_y, train_pd)
    auc_t = roc_auc_score(train_y, train_pd)

    fpr_v, tpr_v, _ = roc_curve(val_y, val_pd)
    auc_v = roc_auc_score(val_y, val_pd)

    ax.plot(fpr_t, tpr_t, color=BLUE, lw=2, label=f"Train AUC = {auc_t:.4f}")
    ax.plot(fpr_v, tpr_v, color=YELLOW, lw=2, linestyle="--", label=f"Val AUC = {auc_v:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    ax.set_title("ROC Curve (Train vs Validation)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.2)

    return fig_to_base64(fig), auc_t, auc_v


# =============================================================================
# Calibration curve
# =============================================================================


def plot_calibration(train_y, train_pd, val_y, val_pd):
    """Overlay calibration curves for Train and Validation and return the embedded image."""
    fig, ax = plt.subplots(figsize=(7, 5))

    frac_t, pred_t = calibration_curve(train_y, train_pd, n_bins=10)
    frac_v, pred_v = calibration_curve(val_y, val_pd, n_bins=10)

    ax.plot(pred_t, frac_t, "o-", color=BLUE, label="Train")
    ax.plot(pred_v, frac_v, "o--", color=YELLOW, label="Validation")
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    ax.set_title("Calibration Curve")
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Observed PD")
    ax.legend()
    ax.grid(True, alpha=0.2)

    return fig_to_base64(fig)


# =============================================================================
# Score distribution
# =============================================================================


def plot_score_distribution(train_df, val_df, score_col, target_col):
    """
    Plot KDE score distributions (Train vs Validation).

    The target_col is currently not used, but kept as a parameter so the function
    signature remains stable if you later want to split by default/non-default.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.kdeplot(train_df[score_col], label="Train", color=BLUE, lw=2, ax=ax)
    sns.kdeplot(val_df[score_col], label="Validation", color=YELLOW, lw=2, ax=ax)

    ax.set_title("Score Distribution (Train vs Validation)")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.2)

    return fig_to_base64(fig)


# =============================================================================
# Master scale + TTC tables
# =============================================================================


def plot_master_scale(train_df, val_df, grade_col, target_col, pd_col):
    """
    Plot master scale information and build grade-level aggregations.

    For each grade we compute:
    - count: exposure volume
    - bad: default count
    - pred: mean predicted PD
    - obs: observed default rate

    The plot combines:
    - bar charts for volumes (train/val)
    - line plots for observed and predicted PDs (train/val)
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Grade-level aggregation on TRAIN
    gtr = (
        train_df.groupby(grade_col)
        .agg(count=(grade_col, "size"), bad=(target_col, "sum"), pred=(pd_col, "mean"))
        .reset_index()
    )
    gtr["obs"] = gtr["bad"] / gtr["count"]

    # Grade-level aggregation on VALIDATION
    gva = (
        val_df.groupby(grade_col)
        .agg(count=(grade_col, "size"), bad=(target_col, "sum"), pred=(pd_col, "mean"))
        .reset_index()
    )
    gva["obs"] = gva["bad"] / gva["count"]

    # Add exposure share and default share
    total_n_tr = int(len(train_df))
    total_bad_tr = int(train_df[target_col].sum())
    total_n_va = int(len(val_df))
    total_bad_va = int(val_df[target_col].sum())

    gtr = add_grade_default_shares(gtr, total_n=total_n_tr, total_bad=total_bad_tr)
    gva = add_grade_default_shares(gva, total_n=total_n_va, total_bad=total_bad_va)

    # Sort grades (prefer numeric ordering)
    try:
        gtr[grade_col] = gtr[grade_col].astype(float)
        gva[grade_col] = gva[grade_col].astype(float)
    except Exception:
        pass
    gtr = gtr.sort_values(grade_col)
    gva = gva.sort_values(grade_col)

    # --- Volumes as bars (left axis) ---
    ax1.bar(
        gtr[grade_col] - 0.15,
        gtr["count"],
        width=0.3,
        color=BLUE,
        alpha=0.5,
        label="Train Volume",
    )
    ax1.bar(
        gva[grade_col] + 0.15,
        gva["count"],
        width=0.3,
        color=YELLOW,
        alpha=0.5,
        label="Val Volume",
    )
    ax1.set_ylabel("Volume")
    ax1.set_xlabel("Grade")

    # --- PD curves (right axis) ---
    ax2 = ax1.twinx()
    ax2.plot(gtr[grade_col], gtr["obs"], "o-", color=BLUE, label="Obs PD Train")
    ax2.plot(gtr[grade_col], gtr["pred"], "x--", color=CYAN, label="Pred PD Train")
    ax2.plot(gva[grade_col], gva["obs"], "o-", color=YELLOW, label="Obs PD Val")
    ax2.plot(gva[grade_col], gva["pred"], "x--", color=MAGENTA, label="Pred PD Val")

    ax2.set_ylabel("PD")
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Master Scale Analysis (Train + Validation)")
    fig.legend(loc="upper center", ncol=4)

    return fig_to_base64(fig), gtr, gva


# =============================================================================
# Coefficients plot
# =============================================================================


def plot_coefficients(df_coef):
    """
    Plot logistic regression coefficients (sorted by absolute magnitude).

    The coefficients are interpreted as impacts on log-odds.
    """
    df_sorted = df_coef.reindex(df_coef["Coefficient"].abs().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df_coef) * 0.35)))
    sns.barplot(data=df_sorted, x="Coefficient", y="Feature", palette="coolwarm", ax=ax)

    ax.axvline(0, color=GREY, lw=1)
    ax.set_title("Model Coefficients (Log-Odds Impact)")
    ax.grid(True, alpha=0.2)

    return fig_to_base64(fig)


# =============================================================================
# TTC tables by grade
# =============================================================================


def build_ttc_table(gdf, pd_ttc_map=None):
    """
    Build a TTC table by grade.

    Parameters
    ----------
    gdf : pd.DataFrame
        Must contain:
          - grade, count, bad, obs, pred
        May optionally contain:
          - pct_individus, pct_defauts
    pd_ttc_map : dict, optional
        Mapping {grade -> PD_TTC_master_scale}. If provided, pd_ttc uses this
        master-scale PD; otherwise it falls back to the sample mean prediction
        per grade (column 'pred').

    Returns
    -------
    str
        HTML table.
    """
    df = gdf.copy()

    # TTC PD: either master-scale PD (preferred) or mean predicted PD by grade.
    if pd_ttc_map is not None:
        df["pd_ttc"] = df["grade"].map(pd_ttc_map)
    else:
        df["pd_ttc"] = df["pred"]

    # Deviations vs TTC
    df["delta_abs"] = df["obs"] - df["pd_ttc"]
    df["delta_rel"] = 100 * df["delta_abs"] / df["pd_ttc"].replace(0, np.nan)

    cols = ["grade", "count", "bad", "obs", "pd_ttc", "delta_abs", "delta_rel"]
    rename = {"count": "n_individus", "bad": "n_defauts", "obs": "pd_obs"}

    # Optional share columns (if present)
    if "pct_individus" in df.columns:
        df["pct_individus"] = 100 * df["pct_individus"]
        cols.insert(3, "pct_individus")
        rename["pct_individus"] = "%_individus"

    if "pct_defauts" in df.columns:
        df["pct_defauts"] = 100 * df["pct_defauts"]
        cols.insert(cols.index("bad") + 1, "pct_defauts")
        rename["pct_defauts"] = "%_defauts"

    df_final = df[cols].rename(columns=rename)

    # Sort by grade
    try:
        df_final["grade"] = df_final["grade"].astype(float)
    except Exception:
        pass
    df_final = df_final.sort_values("grade")

    # Presentation rounding
    if "%_individus" in df_final.columns:
        df_final["%_individus"] = df_final["%_individus"].round(2)
    if "%_defauts" in df_final.columns:
        df_final["%_defauts"] = df_final["%_defauts"].round(2)
    df_final["delta_rel"] = df_final["delta_rel"].round(2)

    return df_final.to_html(index=False)


# =============================================================================
# Load TTC/master-scale PDs from bucket_stats.json
# =============================================================================


def load_pd_ttc_from_master_scale(bucket_stats_path: Path):
    """
    Load bucket_stats.json (generated during training) and return a mapping:
        {grade/bucket -> PD_TTC}
    based on the "train" section.

    If the file is missing or the expected structure is not found, returns None
    and the report will fall back to using predicted mean PDs by grade.
    """
    if not bucket_stats_path.exists():
        print(
            f"[WARN] bucket_stats.json not found at {bucket_stats_path}. "
            "TTC tables will use mean predicted PD per grade.",
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
# CLI arguments
# =============================================================================


def parse_args():
    """Define and parse CLI arguments for report generation."""
    p = argparse.ArgumentParser(description="Generate a unified model report (dark theme)")
    p.add_argument("--train", required=True, help="Path to binned/scored train dataset (parquet)")
    p.add_argument("--validation", required=True, help="Path to binned/scored validation dataset (parquet)")
    p.add_argument("--out", required=True, help="Output HTML report path")
    p.add_argument("--target", default="default_24m", help="Target column name")
    p.add_argument("--score", default="score", help="Score column name (for distributions)")
    p.add_argument("--pd", default="pd", help="Predicted PD column name")
    p.add_argument("--grade", default="grade", help="Grade column name")
    p.add_argument("--model", required=True, help="Path to model_best.joblib (for coefficients)")
    p.add_argument(
        "--bucket-stats",
        default=None,
        help="Path to bucket_stats.json (TTC PD master scale). Default: same folder as model.",
    )
    return p.parse_args()


# =============================================================================
# HTML generation
# =============================================================================


def build_html(
    out_path,
    train_metrics,
    val_metrics,
    roc_img,
    cal_img,
    dist_img,
    ms_img,
    gtr_table_html,
    gva_table_html,
    ttc_train_html,
    ttc_val_html,
    coef_img,
    coef_table_html,
    intercept_val,
):
    """
    Build and write the final HTML report.

    The report is self-contained: images are embedded as base64 strings.
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Credit Risk Model Report — Dark Mode</title>

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

    h1, h2, h3 {{
        color: #E8E8E8;
    }}

    .section {{
        margin-top: 40px;
        padding: 20px;
        background-color: #2A2A2A;
        border-radius: 8px;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
        color: #E8E8E8;
    }}

    th {{
        background-color: #333333;
        padding: 8px;
    }}

    td {{
        background-color: #2E2E2E;
        padding: 8px;
        text-align: center;
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

    <h1>Credit Risk Model — Full Validation Report</h1>
    <p><strong>Generated:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")}</p>

    <!-- GLOBAL METRICS -->
    <div class="section">
        <h2>1. Global Metrics (Train vs Validation)</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Train</th>
                <th>Validation</th>
            </tr>
            <tr><td>N</td><td>{train_metrics['n']}</td><td>{val_metrics['n']}</td></tr>
            <tr><td># Defaults</td><td>{train_metrics['n_defaults']}</td><td>{val_metrics['n_defaults']}</td></tr>
            <tr><td>Default Rate</td><td>{train_metrics['default_rate']:.6f}</td><td>{val_metrics['default_rate']:.6f}</td></tr>
            <tr><td>AUC</td><td>{train_metrics['auc']:.4f}</td><td>{val_metrics['auc']:.4f}</td></tr>
            <tr><td>Gini</td><td>{train_metrics['gini']:.4f}</td><td>{val_metrics['gini']:.4f}</td></tr>
            <tr><td>LogLoss</td><td>{train_metrics['logloss']:.4f}</td><td>{val_metrics['logloss']:.4f}</td></tr>
            <tr><td>Brier Score</td><td>{train_metrics['brier']:.6f}</td><td>{val_metrics['brier']:.6f}</td></tr>
            <tr><td>ECE</td><td>{train_metrics['ece']:.6f}</td><td>{val_metrics['ece']:.6f}</td></tr>
        </table>
    </div>

    <!-- ROC -->
    <div class="section">
        <h2>2. ROC Curve</h2>
        <img src="{roc_img}">
    </div>

    <!-- Calibration -->
    <div class="section">
        <h2>3. Calibration Curve</h2>
        <img src="{cal_img}">
    </div>

    <!-- Score Dist -->
    <div class="section">
        <h2>4. Score Distribution</h2>
        <img src="{dist_img}">
    </div>

    <!-- Master Scale -->
    <div class="section">
        <h2>5. Master Scale Analysis (Train + Validation)</h2>
        <img src="{ms_img}">

        <h3>Train Grades (incl. % individuals / % defaults)</h3>
        {gtr_table_html}

        <h3>Validation Grades (incl. % individuals / % defaults)</h3>
        {gva_table_html}

        <h3>Train TTC Table (PD TTC = Master Scale)</h3>
        {ttc_train_html}

        <h3>Validation TTC Table (PD TTC = Master Scale)</h3>
        {ttc_val_html}
    </div>

    <!-- Coefficients -->
    <div class="section">
        <h2>6. Model Coefficients</h2>
        <p><strong>Intercept:</strong> {intercept_val:.6f}</p>
        <img src="{coef_img}">
        <h3>Coefficients Table</h3>
        {coef_table_html}
    </div>

</div>
</body>
</html>
    """

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    print(f"Loading TRAIN: {args.train}")
    print(f"Loading VALIDATION: {args.validation}")

    # Note: this version reads parquet directly; adapt load if you want csv support.
    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.validation)

    # Drop rows with missing target to avoid metric issues
    train_df = train_df.dropna(subset=[args.target])
    val_df = val_df.dropna(subset=[args.target])

    # Extract y and predicted PDs
    y_tr = train_df[args.target].astype(int).values
    y_va = val_df[args.target].astype(int).values
    pd_tr = train_df[args.pd].astype(float).values
    pd_va = val_df[args.pd].astype(float).values

    # -------------------------
    # Global metrics
    # -------------------------
    train_metrics = dict(
        auc=roc_auc_score(y_tr, pd_tr),
        gini=2 * roc_auc_score(y_tr, pd_tr) - 1,
        logloss=log_loss(y_tr, pd_tr),
        brier=brier_score_loss(y_tr, pd_tr),
        ece=expected_calibration_error(y_tr, pd_tr),
    )
    train_metrics = add_global_default_info(train_metrics, y_tr)

    val_metrics = dict(
        auc=roc_auc_score(y_va, pd_va),
        gini=2 * roc_auc_score(y_va, pd_va) - 1,
        logloss=log_loss(y_va, pd_va),
        brier=brier_score_loss(y_va, pd_va),
        ece=expected_calibration_error(y_va, pd_va),
    )
    val_metrics = add_global_default_info(val_metrics, y_va)

    # -------------------------
    # Plots + grade aggregations
    # -------------------------
    roc_img, _, _ = plot_roc(y_tr, pd_tr, y_va, pd_va)
    cal_img = plot_calibration(y_tr, pd_tr, y_va, pd_va)
    dist_img = plot_score_distribution(train_df, val_df, args.score, args.target)
    ms_img, gtr, gva = plot_master_scale(train_df, val_df, args.grade, args.target, args.pd)

    # -------------------------
    # Model coefficients
    # -------------------------
    model_pkg = joblib.load(args.model)
    best_lr = model_pkg["best_lr"]
    kept = model_pkg["kept_features"]

    coef_df = pd.DataFrame({"Feature": kept, "Coefficient": best_lr.coef_[0]})
    intercept_val = float(best_lr.intercept_[0])

    coef_img = plot_coefficients(coef_df)
    coef_table_html = coef_df.to_html(index=False)

    # -------------------------
    # TTC/master-scale PD mapping (optional)
    # -------------------------
    if args.bucket_stats is not None:
        bucket_stats_path = Path(args.bucket_stats)
    else:
        bucket_stats_path = Path(args.model).with_name("bucket_stats.json")

    pd_ttc_map = load_pd_ttc_from_master_scale(bucket_stats_path)

    # TTC tables expect a column named "grade"
    gtr2 = gtr.copy()
    gtr2["grade"] = gtr2[args.grade]
    gva2 = gva.copy()
    gva2["grade"] = gva2[args.grade]

    ttc_train_html = build_ttc_table(gtr2, pd_ttc_map=pd_ttc_map)
    ttc_val_html = build_ttc_table(gva2, pd_ttc_map=pd_ttc_map)

    # -------------------------
    # Display tables for grade aggregates (with % shares)
    # -------------------------
    gtr_disp = gtr.copy().rename(
        columns={
            args.grade: "grade",
            "count": "n_individus",
            "bad": "n_defauts",
            "pred": "pd_pred_mean",
            "obs": "pd_obs",
            "pct_individus": "%_individus",
            "pct_defauts": "%_defauts",
        }
    )
    gva_disp = gva.copy().rename(
        columns={
            args.grade: "grade",
            "count": "n_individus",
            "bad": "n_defauts",
            "pred": "pd_pred_mean",
            "obs": "pd_obs",
            "pct_individus": "%_individus",
            "pct_defauts": "%_defauts",
        }
    )

    # Convert shares to percentages for display
    if "%_individus" in gtr_disp.columns:
        gtr_disp["%_individus"] = 100 * gtr_disp["%_individus"]
    if "%_defauts" in gtr_disp.columns:
        gtr_disp["%_defauts"] = 100 * gtr_disp["%_defauts"]

    if "%_individus" in gva_disp.columns:
        gva_disp["%_individus"] = 100 * gva_disp["%_individus"]
    if "%_defauts" in gva_disp.columns:
        gva_disp["%_defauts"] = 100 * gva_disp["%_defauts"]

    # Sort by grade (numeric if possible)
    try:
        gtr_disp["grade"] = gtr_disp["grade"].astype(float)
        gva_disp["grade"] = gva_disp["grade"].astype(float)
    except Exception:
        pass
    gtr_disp = gtr_disp.sort_values("grade")
    gva_disp = gva_disp.sort_values("grade")

    gtr_html = gtr_disp.round(6).to_html(index=False)
    gva_html = gva_disp.round(6).to_html(index=False)

    # -------------------------
    # Write HTML report
    # -------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    build_html(
        out_path=out_path,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        roc_img=roc_img,
        cal_img=cal_img,
        dist_img=dist_img,
        ms_img=ms_img,
        gtr_table_html=gtr_html,
        gva_table_html=gva_html,
        ttc_train_html=ttc_train_html,
        ttc_val_html=ttc_val_html,
        coef_img=coef_img,
        coef_table_html=coef_table_html,
        intercept_val=intercept_val,
    )

    print(f"\n✔ Report generated: {out_path}")


if __name__ == "__main__":
    main()
