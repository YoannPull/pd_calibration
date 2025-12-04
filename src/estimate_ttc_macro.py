#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
estimate_ttc_macro.py

Estimate a macro-based TTC PD per grade (pd_ttc_macro) from:
- scored train & validation datasets (with grades and defaults),
- macro time series from FRED (CSV) in data/raw/macro.

Methodology (grade × time panel):

1. Build a panel (grade, period) with:
   - exposures_{g,t} (number of loans),
   - defaults_{g,t},
   - default rate DR_{g,t} = defaults / exposures.

2. Build a quarterly macro panel with:
   - INF (YoY inflation from CPIAUCSL),
     GDP_GROW (YoY GDP growth from GDPC1),
     HPI_GROW (YoY housing price growth from CSUSHPINSA),
     UNRATE (unemployment rate),
     MORTGAGE30US (30Y mortgage rate),
     NFCI (financial conditions index).

3. Estimate a logistic regression at aggregated level:
   logit(p_{g,t}) = α_g + γ' M_t,
   where M_t are the transformed macro variables.

4. Define a "neutral" macro scenario as the historical mean of M_t
   (implemented here via centered macros => neutral = 0).

5. For each grade g, compute:
   PD_TTC_macro(g) = logistic(α_g + γ' M_neutral).

Outputs:
- JSON file with:
  - pd_ttc_macro per grade
  - neutral_macro (non centered)
  - macro_model_metrics
  - macro_model_coefficients
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Load scored data (train / val)
# ---------------------------------------------------------------------------

def load_scored(path: str | Path, target: str, time_col: str, grade_col: str = "grade") -> pd.DataFrame:
    """Load a scored dataset (train or validation) and keep only what's needed."""
    df = pd.read_parquet(path)
    missing = [c for c in [target, time_col, grade_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df[[target, time_col, grade_col]].copy()
    df = df.dropna(subset=[target, time_col, grade_col])
    return df


def to_quarter_period(series: pd.Series) -> pd.PeriodIndex:
    """
    Convert a 'vintage' / time series to quarterly PeriodIndex.

    Strategy:
    1. Try to parse as datetime and convert to quarterly periods.
    2. If that fails, assume strings like 'YYYYQn' or similar and
       build a PeriodIndex with freq='Q'.
    """
    try:
        dt = pd.to_datetime(series)
        return dt.dt.to_period("Q")
    except Exception:
        # Fallback: assume quarter labels like "2010Q1"
        return pd.PeriodIndex(series.astype(str), freq="Q")


# ---------------------------------------------------------------------------
# Macro FRED: loading and preparation
# ---------------------------------------------------------------------------

def load_fred_series(csv_path: Path, col_alias: str) -> pd.DataFrame:
    """
    Load a FRED CSV with columns like:
      - observation_date,<SERIES_CODE>
    or
      - DATE,<SERIES_CODE>

    and rename the series column to `col_alias`.
    """
    df = pd.read_csv(csv_path)

    # Date column
    if "DATE" in df.columns:
        date_col = "DATE"
    elif "observation_date" in df.columns:
        date_col = "observation_date"
    else:
        raise ValueError(
            f"Unexpected date column in {csv_path}. "
            f"Expected 'DATE' or 'observation_date', got: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])

    # Value column (everything except the date)
    value_cols = [c for c in df.columns if c != date_col]
    if len(value_cols) != 1:
        raise ValueError(
            f"Unexpected number of value columns in {csv_path}. "
            f"Expected 1, got {len(value_cols)}: {value_cols}"
        )

    series_name = value_cols[0]
    df = df[[date_col, series_name]].rename(columns={date_col: "DATE", series_name: col_alias})
    return df


def to_quarterly(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Convert a (DATE, value_col) series to quarterly frequency by mean,
    and return (period, value_col).
    """
    df_q = (
        df.set_index("DATE")[value_col]
        .resample("Q")
        .mean()
        .to_frame()
    )
    df_q["period"] = df_q.index.to_period("Q")
    df_q = df_q.reset_index(drop=True)
    return df_q[["period", value_col]]


def prepare_macro_panel(macro_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """
    Prepare quarterly macro panel with transformed variables.

    Input files in macro_dir (FRED CSV format):
      - CPIAUCSL.csv       -> CPI (price level)
      - GDPC1.csv          -> GDP (real GDP)
      - CSUSHPINSA.csv     -> HPI (house price index)
      - MORTGAGE30US.csv   -> mortgage rate 30Y
      - NFCI.csv           -> NFCI
      - UNRATE.csv         -> unemployment rate

    Returns:
      macro_panel: DataFrame with columns:
        - period (Period[Q])
        - transformed & centered macro variables (suffix '_c')
      macro_cols_centered: list of macro column names to be used in the model
      neutral_macro: dict of neutral macro values (means of transformed vars, non centered)
    """
    macro_dir = Path(macro_dir)

    # --- 1. Load raw series ---
    cpi = load_fred_series(macro_dir / "CPIAUCSL.csv", "CPI")
    gdp = load_fred_series(macro_dir / "GDPC1.csv", "GDP")
    hpi = load_fred_series(macro_dir / "CSUSHPINSA.csv", "HPI")
    mort = load_fred_series(macro_dir / "MORTGAGE30US.csv", "MORTGAGE30US")
    nfci = load_fred_series(macro_dir / "NFCI.csv", "NFCI")
    unrate = load_fred_series(macro_dir / "UNRATE.csv", "UNRATE")

    # Quarterly aggregation
    cpi_q = to_quarterly(cpi, "CPI")
    gdp_q = to_quarterly(gdp, "GDP")
    hpi_q = to_quarterly(hpi, "HPI")
    mort_q = to_quarterly(mort, "MORTGAGE30US")
    nfci_q = to_quarterly(nfci, "NFCI")
    unrate_q = to_quarterly(unrate, "UNRATE")

    # Merge all series on period (quarterly levels)
    macro = cpi_q.merge(gdp_q, on="period", how="outer")
    macro = macro.merge(hpi_q, on="period", how="outer")
    macro = macro.merge(mort_q, on="period", how="outer")
    macro = macro.merge(nfci_q, on="period", how="outer")
    macro = macro.merge(unrate_q, on="period", how="outer")

    macro = macro.sort_values("period").reset_index(drop=True)

    # --- SAVE STEP 1: quarterly levels (CPI, GDP, HPI, ...) ---
    processed_macro_dir = Path("data/processed/macro")
    processed_macro_dir.mkdir(parents=True, exist_ok=True)
    macro_levels = macro.copy()
    macro_levels.to_parquet(processed_macro_dir / "macro_quarterly_levels.parquet", index=False)

    # --- 2. Transformations (YoY growth / inflation) ---
    macro["INF"] = 100.0 * (np.log(macro["CPI"]) - np.log(macro["CPI"].shift(4)))
    macro["GDP_GROW"] = 100.0 * (np.log(macro["GDP"]) - np.log(macro["GDP"].shift(4)))
    macro["HPI_GROW"] = 100.0 * (np.log(macro["HPI"]) - np.log(macro["HPI"].shift(4)))

    macro_features = ["INF", "GDP_GROW", "HPI_GROW", "UNRATE", "MORTGAGE30US", "NFCI"]
    macro = macro[["period"] + macro_features].copy()
    macro = macro.dropna(subset=macro_features)

    # --- SAVE STEP 2: transformed quarterly macros (non centered) ---
    macro_transformed = macro.copy()
    macro_transformed.to_parquet(processed_macro_dir / "macro_quarterly_transformed.parquet", index=False)

    # --- 3. Centering (neutral macro = mean) ---
    neutral_macro: Dict[str, float] = {}
    for col in macro_features:
        macro[col] = macro[col].astype(float)
        mean_val = float(macro[col].mean())
        neutral_macro[col] = mean_val
        macro[col + "_c"] = macro[col] - mean_val

    macro_cols_centered = [col + "_c" for col in macro_features]
    macro_panel = macro[["period"] + macro_cols_centered].copy()

    # --- SAVE STEP 3: centered macro panel used in the TTC model ---
    macro_panel.to_parquet(processed_macro_dir / "macro_quarterly_centered.parquet", index=False)

    return macro_panel, macro_cols_centered, neutral_macro


# ---------------------------------------------------------------------------
# Panel (grade, period)
# ---------------------------------------------------------------------------

def build_grade_time_panel(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    target: str,
    time_col: str,
    grade_col: str = "grade",
) -> pd.DataFrame:
    """
    Build a (grade, period) panel with exposures, defaults and default rate.
    """
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    df_all = df_all.dropna(subset=[target, time_col, grade_col])

    df_all["period"] = to_quarter_period(df_all[time_col])
    df_all["grade"] = df_all[grade_col].astype(int)
    df_all["default_flag"] = df_all[target].astype(int)

    panel = (
        df_all.groupby(["grade", "period"], as_index=False)
        .agg(
            exposures=("default_flag", "size"),
            defaults=("default_flag", "sum"),
        )
    )
    panel["dr"] = panel["defaults"] / panel["exposures"]
    return panel


# ---------------------------------------------------------------------------
# Macro TTC model (aggregated logit, X construit à la main)
# ---------------------------------------------------------------------------

def fit_macro_ttc_model(
    panel: pd.DataFrame,
    macro_panel: pd.DataFrame,
    macro_cols_centered: List[str],
) -> Tuple[LogisticRegression, pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Merge grade-time panel with macro panel and fit an aggregated logistic regression:

    logit(p_{g,t}) = α_g + γ' M_t

    Implementation:
      - Build X_base[g,t] = [1_{grade=g} (for each grade), macro_t (centered) ...]
      - For each (g,t), create two observations:
          y=1 with weight = defaults_{g,t}
          y=0 with weight = exposures_{g,t} - defaults_{g,t}
      - Fit sklearn.LogisticRegression with sample_weight.

    Returns:
      - lr: fitted LogisticRegression
      - df_merge: DataFrame panel+macro (one row per (g,t))
      - X_base: design matrix (one row per (g,t))
      - grades: sorted unique grades (np.array)
      - macro_cols_centered: list of macro columns used (for naming)
    """
    df = panel.merge(macro_panel, on="period", how="inner")
    df = df[df["exposures"] > 0].copy()

    if df.empty:
        raise ValueError("No overlapping periods between grade-time panel and macro panel.")

    df = df.sort_values(["period", "grade"]).reset_index(drop=True)

    grades = np.sort(df["grade"].unique())
    n_grades = len(grades)
    grade_to_idx = {g: i for i, g in enumerate(grades)}

    n_macros = len(macro_cols_centered)
    n_features = n_grades + n_macros

    # --- Build X_base: one row per (g,t) ---
    X_rows: List[np.ndarray] = []
    for _, row in df.iterrows():
        x = np.zeros(n_features, dtype=float)
        # Grade dummies
        g = int(row["grade"])
        x[grade_to_idx[g]] = 1.0
        # Macros
        for j, col in enumerate(macro_cols_centered):
            x[n_grades + j] = float(row[col])
        X_rows.append(x)

    X_base = np.vstack(X_rows)

    exposures = df["exposures"].values.astype(float)
    defaults = df["defaults"].values.astype(float)

    # --- Aggregated representation for logit ---
    X_rep: List[np.ndarray] = []
    y_rep: List[int] = []
    w_rep: List[float] = []

    for i in range(len(df)):
        x_i = X_base[i]
        n = int(exposures[i])
        d = int(defaults[i])
        nd = n - d

        if d > 0:
            X_rep.append(x_i)
            y_rep.append(1)
            w_rep.append(d)
        if nd > 0:
            X_rep.append(x_i)
            y_rep.append(0)
            w_rep.append(nd)

    X_rep_arr = np.vstack(X_rep)
    y_rep_arr = np.array(y_rep, dtype=int)
    w_rep_arr = np.array(w_rep, dtype=float)

    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X_rep_arr, y_rep_arr, sample_weight=w_rep_arr)

    return lr, df, X_base, grades, macro_cols_centered


# ---------------------------------------------------------------------------
# Macro model diagnostics
# ---------------------------------------------------------------------------

def evaluate_macro_model(
    df: pd.DataFrame,
    X_base: np.ndarray,
    lr: LogisticRegression,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Evaluate the aggregated macro model and produce:
      - global metrics (pseudo-R2, Brier-like, corr, etc.)
      - a coefficients table (names will be generic, mapping is handled outside).
    """
    exposures = df["exposures"].values.astype(float)
    defaults = df["defaults"].values.astype(float)
    dr_obs = df["dr"].values.astype(float)

    p_hat = lr.predict_proba(X_base)[:, 1]

    w = exposures
    mean_dr = float(np.average(dr_obs, weights=w))
    mean_phat = float(np.average(p_hat, weights=w))

    mse = float(np.average((p_hat - dr_obs) ** 2, weights=w))

    y = dr_obs
    y_bar = mean_dr
    p_bar = mean_phat
    cov = float(np.average((y - y_bar) * (p_hat - p_bar), weights=w))
    var_y = float(np.average((y - y_bar) ** 2, weights=w))
    var_p = float(np.average((p_hat - p_bar) ** 2, weights=w))
    if var_y > 0 and var_p > 0:
        corr = float(cov / np.sqrt(var_y * var_p))
    else:
        corr = float("nan")

    eps = 1e-12
    ll = float(
        (defaults * np.log(np.clip(p_hat, eps, 1 - eps))
         + (exposures - defaults) * np.log(np.clip(1 - p_hat, eps, 1 - eps))).sum()
    )
    ll_null = float(
        (defaults * np.log(np.clip(mean_dr, eps, 1 - eps))
         + (exposures - defaults) * np.log(np.clip(1 - mean_dr, eps, 1 - eps))).sum()
    )
    pseudo_r2 = 1.0 - ll / ll_null if ll_null != 0 else float("nan")

    metrics = {
        "exposures_total": float(w.sum()),
        "mean_dr": mean_dr,
        "mean_pred": mean_phat,
        "mse_brier_like": mse,
        "corr_dr_pred": corr,
        "loglik": ll,
        "loglik_null": ll_null,
        "pseudo_r2_mcfadden": pseudo_r2,
    }

    # Coefficients: naming will be done outside
    coef = lr.coef_[0]
    intercept = float(lr.intercept_[0])

    coef_table: List[Dict[str, float]] = [{"name": "intercept", "coef": intercept}]
    for i, c in enumerate(coef):
        coef_table.append({"name": f"beta_{i}", "coef": float(c)})

    return metrics, coef_table


# ---------------------------------------------------------------------------
# PD TTC macro per grade
# ---------------------------------------------------------------------------

def compute_pd_ttc_macro_per_grade(
    lr: LogisticRegression,
    grades: np.ndarray,
    n_macros: int,
) -> Dict[int, float]:
    """
    Compute PD_TTC_macro per grade for neutral macro scenario.

    Neutral macro:
      - all centered macro variables set to 0
      - only the dummy corresponding to grade g set to 1.
    """
    n_grades = len(grades)
    n_features = n_grades + n_macros

    pd_ttc_macro: Dict[int, float] = {}

    for idx_g, g in enumerate(grades):
        x_row = np.zeros(n_features, dtype=float)
        x_row[idx_g] = 1.0  # grade dummy

        p = float(lr.predict_proba(x_row.reshape(1, -1))[:, 1][0])
        pd_ttc_macro[int(g)] = p

    return pd_ttc_macro


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_pd_ttc_macro(
    pd_ttc_macro: Dict[int, float],
    neutral_macro: Dict[str, float],
    out_path: Path,
    metrics: Dict[str, float] | None = None,
    coef_table: List[Dict[str, float]] | None = None,
) -> None:
    """
    Save PD_TTC_macro per grade, neutral macro scenario, and (optionally)
    macro-model diagnostics (metrics + coefficients) to JSON.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "pd_ttc_macro": [
            {"grade": int(g), "pd_ttc_macro": float(p)}
            for g, p in sorted(pd_ttc_macro.items())
        ],
        "neutral_macro": neutral_macro,
    }

    if metrics is not None:
        payload["macro_model_metrics"] = metrics

    if coef_table is not None:
        payload["macro_model_coefficients"] = coef_table

    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate macro-based TTC PD per grade (pd_ttc_macro).")
    p.add_argument(
        "--train-scored",
        required=True,
        help="Path to train_scored.parquet (must contain grade, target, time_col).",
    )
    p.add_argument(
        "--val-scored",
        required=True,
        help="Path to validation_scored.parquet (must contain grade, target, time_col).",
    )
    p.add_argument(
        "--macro-dir",
        default="data/raw/macro",
        help="Directory containing macro CSV files from FRED.",
    )
    p.add_argument(
        "--target",
        default="default_24m",
        help="Name of the binary default column in scored datasets.",
    )
    p.add_argument(
        "--time-col",
        default="vintage",
        help="Name of the time column (will be converted to quarterly period).",
    )
    p.add_argument(
        "--out-json",
        default="artifacts/pd_ttc_macro.json",
        help="Output JSON file for pd_ttc_macro per grade.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_scored = load_scored(args.train_scored, target=args.target, time_col=args.time_col, grade_col="grade")
    val_scored = load_scored(args.val_scored, target=args.target, time_col=args.time_col, grade_col="grade")

    # (grade, period) panel
    panel = build_grade_time_panel(train_scored, val_scored, target=args.target, time_col=args.time_col)

    # Macro panel (quarterly, transformed & centered)
    macro_panel, macro_cols_centered, neutral_macro = prepare_macro_panel(Path(args.macro_dir))

    # Fit macro TTC model
    lr_macro, df_merge, X_base, grades, macro_cols_centered = fit_macro_ttc_model(
        panel, macro_panel, macro_cols_centered
    )

    # PD_TTC_macro per grade (neutral macro)
    pd_ttc_macro = compute_pd_ttc_macro_per_grade(
        lr_macro, grades, n_macros=len(macro_cols_centered)
    )

    # Diagnostics
    metrics, coef_table = evaluate_macro_model(df_merge, X_base, lr_macro)

    print("Macro TTC model diagnostics:")
    for k, v in metrics.items():
        print(f"  {k:20s} : {v:.6f}")

    # Save JSON
    out_path = Path(args.out_json)
    save_pd_ttc_macro(
        pd_ttc_macro=pd_ttc_macro,
        neutral_macro=neutral_macro,
        out_path=out_path,
        metrics=metrics,
        coef_table=coef_table,
    )

    print(f"✔ pd_ttc_macro saved to: {out_path}")
    for g, p in sorted(pd_ttc_macro.items()):
        print(f"  grade {g:2d} -> PD_TTC_macro = {p:.6f}")


if __name__ == "__main__":
    main()
