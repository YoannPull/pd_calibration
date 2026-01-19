#!/usr/bin/env python3
# src/apply_model.py
# -*- coding: utf-8 -*-

"""
APPLY MODEL — FULL PIPELINE (OOS)
================================

Goal
----
Replay the *full* preprocessing + scoring pipeline on a new dataset (e.g., OOS),
using artifacts learned on the training sample:

  1) Imputation          (imputer.joblib)
  2) Binning             (bins.json -> load_bins_json + transform_with_learned_bins)
  3) WOE + interactions  (woe_maps stored in model_best.joblib)
  4) Feature selection   (kept_features stored in model_best.joblib)
  5) LR + calibration    (best_lr + model_pd stored in model_best.joblib)
  6) TTC score + grade   (risk_buckets.json)

Output
------
A scored dataset containing (at least):
- an identifier column,
- TTC score,
- predicted PD,
- grade (master scale bucket).

Notes
-----
- The script reuses the *exact* binning helpers from features.binning to ensure
  train/test consistency.
- A small import fallback is provided to support running the file as a standalone
  script without setting PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# --- Binning: reuse EXACTLY the helpers from the binning module ---
try:
    from features.binning import load_bins_json, transform_with_learned_bins
except ImportError:
    # Fallback if executed as a script without PYTHONPATH=src
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from features.binning import load_bins_json, transform_with_learned_bins


# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Score scaling (same convention as train_model.py)
# ---------------------------------------------------------------------------

def scale_score(log_odds, base_points=600, base_odds=50, pdo=20):
    """
    Convert model log-odds into a points-based TTC score.

    Parameters
    ----------
    log_odds : array-like
        Log-odds from the logistic regression decision function.
    base_points : int
        Score assigned at the base odds.
    base_odds : float
        Odds corresponding to base_points (e.g., 50 means 50:1).
    pdo : float
        "Points to double the odds": score change corresponding to a doubling of odds.

    Returns
    -------
    np.ndarray[int]
        Integer TTC scores (rounded).
    """
    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)
    return np.round(offset - factor * log_odds).astype(int)


# ---------------------------------------------------------------------------
# Interaction terms (mirrors train_model.py)
# ---------------------------------------------------------------------------

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add handcrafted interaction terms on WOE-transformed variables.

    The interactions are only created if the required WOE columns exist.
    """
    if "credit_score_WOE" in df.columns and "original_cltv_WOE" in df.columns:
        df["inter_score_x_cltv"] = df["credit_score_WOE"] * df["original_cltv_WOE"]

    if "credit_score_WOE" in df.columns and "original_dti_WOE" in df.columns:
        df["inter_score_x_dti"] = df["credit_score_WOE"] * df["original_dti_WOE"]

    if "original_cltv_WOE" in df.columns and "original_dti_WOE" in df.columns:
        df["inter_cltv_x_dti"] = df["original_cltv_WOE"] * df["original_dti_WOE"]

    return df


# ---------------------------------------------------------------------------
# Apply WOE mapping (same logic as train_model.py)
# ---------------------------------------------------------------------------

def apply_woe(df: pd.DataFrame, woe_maps: dict, bin_suffix: str) -> pd.DataFrame:
    """
    Apply WOE encoding using precomputed WOE maps.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that contains binned columns (integer codes).
    woe_maps : dict
        Mapping describing, for each raw variable, the binned-to-WOE mapping and a
        default WOE value for missing/unseen bins.
        Expected structure (per variable):
          - info["default"]: float
          - info["map"]: dict {bin_code (str/int) -> woe_value (float)}
    bin_suffix : str
        Suffix used to name binned columns. The code supports both conventions:
        "{suffix}{raw}" and "{raw}{suffix}".

    Returns
    -------
    pd.DataFrame
        A dataframe with one WOE column per raw variable, named "{raw}_WOE".
    """
    cols = []

    for raw, info in woe_maps.items():
        # Handle both naming conventions for binned columns.
        bin1 = f"{bin_suffix}{raw}"
        bin2 = f"{raw}{bin_suffix}"
        colbin = bin1 if bin1 in df.columns else (bin2 if bin2 in df.columns else None)

        default = float(info["default"])
        mapping = {int(k): float(v) for k, v in info["map"].items()}

        if colbin is None:
            # No corresponding binned column -> use the global default WOE
            cols.append(pd.Series(default, index=df.index, name=f"{raw}_WOE"))
        else:
            # Map binned codes to WOE values; unseen codes fall back to default.
            cols.append(df[colbin].map(mapping).fillna(default).rename(f"{raw}_WOE"))

    return pd.concat(cols, axis=1)


# ---------------------------------------------------------------------------
# Buckets -> grade
# ---------------------------------------------------------------------------

def apply_buckets(scores: np.ndarray, edges: list[float]) -> np.ndarray:
    """
    Assign a risk grade based on TTC scores and master-scale edges.

    Convention (aligned with training):
      - grade = 1 : least risky (HIGHEST scores)
      - grade = n : most risky (LOWEST scores)

    Implementation detail:
    - np.digitize creates increasing bucket indices; we then reverse the numbering
      to match the "1 = best" convention.
    """
    # "Raw" buckets: 1 corresponds to the smallest scores (most risky)
    raw_bucket = np.digitize(scores, edges[1:], right=True) + 1

    # Total number of buckets (len(edges) = n_buckets + 1)
    n_buckets = len(edges) - 1

    # Final convention: 1 = least risky, n = most risky
    grade = n_buckets + 1 - raw_bucket
    return grade


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    """Define and parse CLI arguments."""
    p = argparse.ArgumentParser(description="Apply full credit-risk pipeline on new data (OOS).")
    p.add_argument("--data", required=True, help="Raw or labeled OOS file (parquet/csv).")
    p.add_argument("--out", required=True, help="Output scored file (parquet/csv based on extension).")
    p.add_argument("--imputer", required=True, help="Training-fitted imputer.joblib.")
    p.add_argument("--bins", required=True, help="bins.json (serialized binning definition).")
    p.add_argument("--model", required=True, help="model_best.joblib (LR + calibration + WOE artifacts).")
    p.add_argument("--buckets", required=True, help="risk_buckets.json (master scale edges).")
    p.add_argument("--target", default=None, help="Optional target column name (if present).")
    p.add_argument("--id-col", default="loan_sequence_number", help="Row identifier (propagated to output).")
    return p.parse_args()


def main():
    """Run the full preprocessing + scoring pipeline."""
    args = parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    print(f"[INFO] Loading raw data: {data_path}")
    df_raw = load_any(str(data_path))

    # ------------------------
    # 1) Imputation
    # ------------------------
    print(f"[INFO] Loading imputer: {args.imputer}")
    imputer = joblib.load(args.imputer)

    print("[INFO] Applying imputation (same transformer as training)...")
    X_imp = imputer.transform(df_raw)

    # Ensure we have a DataFrame with consistent column names.
    if isinstance(X_imp, pd.DataFrame):
        df_imp = X_imp
    else:
        if hasattr(imputer, "feature_names_in_"):
            cols = list(imputer.feature_names_in_)
        else:
            cols = list(df_raw.columns)
        df_imp = pd.DataFrame(X_imp, columns=cols, index=df_raw.index)

    # ------------------------
    # 2) Binning (reuse learned bins)
    # ------------------------
    print(f"[INFO] Loading learned bins from JSON: {args.bins}")
    learned_bins = load_bins_json(args.bins)

    print("[INFO] Applying learned bins (transform_with_learned_bins)...")
    df_binned = transform_with_learned_bins(df_imp, learned_bins)
    bin_suffix = learned_bins.bin_col_suffix

    # ------------------------
    # 3) WOE + interactions + feature selection
    # ------------------------
    print(f"[INFO] Loading model artifacts: {args.model}")
    pkg = joblib.load(args.model)
    best_lr = pkg["best_lr"]
    model_pd = pkg["model_pd"]
    woe_maps = pkg["woe_maps"]
    kept_features = pkg["kept_features"]

    print("[INFO] Applying WOE transformation...")
    df_woe = apply_woe(df_binned, woe_maps, bin_suffix=bin_suffix)
    df_woe = add_interactions(df_woe)

    # Keep only features selected during training; fill NA defensively.
    X = df_woe[kept_features].fillna(0).values

    # ------------------------
    # 4) TTC score + PD + grade
    # ------------------------
    print("[INFO] Computing TTC score and PD...")
    log_odds = best_lr.decision_function(X)
    score_ttc = scale_score(log_odds)
    pd_hat = model_pd.predict_proba(X)[:, 1]

    print(f"[INFO] Loading buckets (master scale): {args.buckets}")
    edges = json.loads(Path(args.buckets).read_text())["edges"]
    grade = apply_buckets(score_ttc, edges)

    # ------------------------
    # 5) Build output dataset
    # ------------------------
    print("[INFO] Building output DataFrame...")
    out_df = pd.DataFrame(index=df_raw.index)

    # Identifier column (or fallback to row index).
    if args.id_col in df_raw.columns:
        out_df[args.id_col] = df_raw[args.id_col]
    else:
        out_df[args.id_col] = df_raw.index

    # Propagate vintage if present (useful for monitoring / reporting).
    if "vintage" in df_raw.columns:
        out_df["vintage"] = df_raw["vintage"]

    # Core outputs
    out_df["score_ttc"] = score_ttc
    out_df["pd"] = pd_hat
    out_df["grade"] = grade

    # Optional: keep raw target as-is (not passed through imputation/binning).
    if args.target and args.target in df_raw.columns:
        out_df[args.target] = df_raw[args.target]

    # ------------------------
    # 6) Save
    # ------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in (".parquet", ".pq"):
        out_df.to_parquet(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False)

    print(f"✔ Scoring completed. Saved to: {out_path}")


if __name__ == "__main__":
    main()
