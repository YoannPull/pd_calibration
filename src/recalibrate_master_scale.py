#!/usr/bin/env python3
# src/recalibrate_master_scale.py
# -*- coding: utf-8 -*-

"""
recalibrate_grade_pd.py
======================

Type-(1) recalibration: update the *grade -> PD* lookup table.

Two estimators are supported per grade:
- mean     : raw grade PD (with optional pseudo-count smoothing)
- isotonic : monotone-smoothed grade PD via IsotonicRegression (increasing with grade)

Two aggregation modes are supported:
- pooled     : PD_k = sum(bad_k) / sum(count_k) across all observations in the input
- time_mean  : PD_k = average over time of DR_{k,t} (grade k, period t),
               useful if you want to "average through time" instead of pooling.

Optional time filtering:
- rolling window: --window-years 5   (requires --time-col)
- explicit bounds: --vintage-start / --vintage-end

Output
------
A JSON payload aligned with the `bucket_stats.json` style used elsewhere in the
pipeline, under the "train" key (list of per-grade records).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# =============================================================================
# I/O
# =============================================================================

def load_any(path: str) -> pd.DataFrame:
    """
    Load a dataset from Parquet or CSV (format inferred from extension).
    """
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


# =============================================================================
# Time parsing utilities
# =============================================================================

def parse_vintage_to_period(s: pd.Series, freq: str = "Q") -> pd.PeriodIndex:
    """
    Convert a "vintage" column to a pandas PeriodIndex.

    Supported formats:
    - Quarter strings: '2015Q1', '2015-Q1'
    - Parseable dates: '2015-03-31', etc. -> converted to quarters/months
    - Already a Period dtype

    Parameters
    ----------
    s : pd.Series
        Vintage-like column.
    freq : {"Q","M"}
        Target period frequency (quarters or months).

    Returns
    -------
    pd.PeriodIndex
        Period representation of the input.
    """
    if isinstance(s.dtype, pd.PeriodDtype):
        return s.astype("period[Q]").array

    s_str = s.astype(str)

    # Case 1: strings like "YYYYQn" or "YYYY-Qn"
    m = s_str.str.match(r"^\d{4}[- ]?Q[1-4]$")
    if m.all():
        cleaned = s_str.str.replace(" ", "", regex=False).str.replace("-", "", regex=False)
        return pd.PeriodIndex(cleaned, freq="Q")

    # Case 2: attempt datetime parsing then convert
    dt = pd.to_datetime(s_str, errors="coerce", utc=False)
    if dt.notna().mean() < 0.95:
        bad = s_str[dt.isna()].head(5).tolist()
        raise ValueError(
            f"Could not parse vintage as dates/quarters. Examples of non-parseable values: {bad}"
        )

    if freq.upper().startswith("Q"):
        return dt.dt.to_period("Q")
    if freq.upper().startswith("M"):
        return dt.dt.to_period("M")
    raise ValueError(f"Unsupported freq: {freq}")


# =============================================================================
# Grade reconstruction (optional)
# =============================================================================

def ensure_grade(
    df: pd.DataFrame,
    grade_col: str,
    score_col: str | None,
    buckets_json: str | None,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """
    Ensure that `grade_col` exists in the input.

    If grade is missing, reconstruct it from a score column and a bucket definition
    (risk_buckets.json) so that the convention matches training:
      - grade 1: least risky (highest scores)
      - grade N: most risky  (lowest scores)
    """
    if grade_col in df.columns:
        return df

    if score_col is None or buckets_json is None:
        raise ValueError(
            f"Missing '{grade_col}'. Provide --score-col and --buckets to reconstruct grades."
        )

    edges = json.loads(Path(buckets_json).read_text(encoding="utf-8"))["edges"]
    scores = df[score_col].to_numpy()

    # Raw bucket: 1 corresponds to lowest score range
    raw_grade = np.digitize(scores, np.array(edges)[1:], right=True) + 1

    # Final convention: 1 is least risky (invert order)
    grade = n_buckets + 1 - raw_grade

    out = df.copy()
    out[grade_col] = grade.astype(int)
    return out


# =============================================================================
# PD table computation
# =============================================================================

def compute_pd_table(
    df: pd.DataFrame,
    target_col: str,
    grade_col: str,
    method: str,
    aggregation: str,
    time_col: str | None,
    time_freq: str,
    smooth: float,
    score_col: str | None,
) -> dict:
    """
    Compute grade-level PDs and optionally apply isotonic monotone smoothing.

    Parameters
    ----------
    df : pd.DataFrame
        Input scored dataset (can be a concatenation of multiple files).
    target_col : str
        Binary target name (0/1).
    grade_col : str
        Grade column (integer).
    method : {"mean","isotonic"}
        PD estimator per grade (raw vs monotone-smoothed).
    aggregation : {"pooled","time_mean"}
        How to aggregate information across time.
    time_col : str or None
        Vintage/time column used for time_mean aggregation.
    time_freq : {"Q","M"}
        Period frequency used when parsing the time column.
    smooth : float
        Pseudo-count smoothing: PD = (bad + smooth) / (count + 2*smooth).
        Example: smooth=0.5 yields Jeffreys-like shrinkage.
    score_col : str or None
        If present, attach min/max score observed within each grade.

    Returns
    -------
    dict
        JSON-serializable payload aligned with `bucket_stats.json` structure.
    """
    d = df.copy()
    d[target_col] = d[target_col].astype(int)
    d[grade_col] = d[grade_col].astype(int)

    has_score = score_col is not None and score_col in d.columns

    # ------------------------------------------------------------------
    # Aggregation step: produce a grade-level table with count, bad, pd_raw
    # ------------------------------------------------------------------
    if aggregation == "pooled":
        g = d.groupby(grade_col, as_index=False).agg(
            count=(target_col, "size"),
            bad=(target_col, "sum"),
        )
        g["pd_raw"] = (g["bad"] + smooth) / (g["count"] + 2 * smooth)

        if has_score:
            score_stats = d.groupby(grade_col).agg(
                min_score=(score_col, "min"),
                max_score=(score_col, "max"),
            ).reset_index()
            g = g.merge(score_stats, on=grade_col, how="left")
        else:
            g["min_score"] = np.nan
            g["max_score"] = np.nan

    elif aggregation == "time_mean":
        if time_col is None:
            raise ValueError("aggregation=time_mean requires --time-col.")

        # Compute period label (quarter/month) and grade-time default rates
        d["_period"] = parse_vintage_to_period(d[time_col], freq=time_freq)

        gt = d.groupby([grade_col, "_period"], as_index=False).agg(
            count=(target_col, "size"),
            bad=(target_col, "sum"),
        )
        gt["dr"] = (gt["bad"] + smooth) / (gt["count"] + 2 * smooth)

        # Arithmetic mean of grade-time default rates (equal weight per period)
        g = gt.groupby(grade_col, as_index=False).agg(
            n_periods=("_period", "nunique"),
            count=("count", "sum"),  # total volume (useful as isotonic weights)
            bad=("bad", "sum"),
            pd_raw=("dr", "mean"),
        )

        if has_score:
            score_stats = d.groupby(grade_col).agg(
                min_score=(score_col, "min"),
                max_score=(score_col, "max"),
            ).reset_index()
            g = g.merge(score_stats, on=grade_col, how="left")
        else:
            g["min_score"] = np.nan
            g["max_score"] = np.nan

    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    # Sort grades to enforce the expected direction (1 -> least risky, N -> most risky)
    g = g.sort_values(grade_col).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Optional isotonic smoothing to enforce monotonic PDs across grades
    # ------------------------------------------------------------------
    if method == "mean":
        g["pd"] = g["pd_raw"]
    elif method == "isotonic":
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        x = g[grade_col].to_numpy(dtype=float)
        y = g["pd_raw"].to_numpy(dtype=float)
        w = g["count"].to_numpy(dtype=float)  # weights = exposure counts per grade
        g["pd"] = iso.fit_transform(x, y, sample_weight=w)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Monotonicity check: PD should increase with grade (grade↑ => PD↑)
    mono = bool(np.all(np.diff(g["pd"].to_numpy()) >= -1e-15))

    # ------------------------------------------------------------------
    # Build a `bucket_stats.json`-like payload (records under "train")
    # ------------------------------------------------------------------
    recs = []
    for _, r in g.iterrows():
        recs.append({
            "bucket": int(r[grade_col]),
            "count": int(r["count"]),
            "bad": int(r["bad"]),
            "min_score": None if pd.isna(r["min_score"]) else float(r["min_score"]),
            "max_score": None if pd.isna(r["max_score"]) else float(r["max_score"]),
            "pd_raw": float(r["pd_raw"]),
            "pd": float(r["pd"]),
        })

    payload = {
        "method": method,
        "aggregation": aggregation,
        "target": target_col,
        "grade_col": grade_col,
        "time_col": time_col,
        "time_freq": time_freq,
        "smooth": smooth,
        "n_obs": int(len(d)),
        "monotone_pd": mono,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train": recs,
    }
    return payload


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Type-(1) recalibration: update grade->PD table.")
    p.add_argument(
        "--scored",
        action="append",
        required=True,
        help="Path to a *_scored file (parquet/csv). Can be repeated.",
    )
    p.add_argument("--target", required=True, help="Binary default label column name")
    p.add_argument("--grade-col", default="grade", help="Grade column name")
    p.add_argument("--time-col", default=None, help="Vintage/time column (required for time_mean or time filters)")
    p.add_argument("--time-freq", default="Q", choices=["Q", "M"], help="Period frequency used for parsing time-col")
    p.add_argument("--method", default="mean", choices=["mean", "isotonic"], help="Raw vs isotonic-smoothed PDs")
    p.add_argument("--aggregation", default="pooled", choices=["pooled", "time_mean"], help="Pooling strategy")
    p.add_argument("--smooth", type=float, default=0.0, help="Pseudo-count smoothing (0.5 ~ Jeffreys-like)")
    p.add_argument("--out-json", required=True, help="Output JSON path")

    # Optional time filtering
    p.add_argument("--window-years", type=int, default=None, help="Keep last X years (requires --time-col)")
    p.add_argument("--vintage-start", default=None, help="Lower bound (e.g., 2015Q1 or a parseable date)")
    p.add_argument("--vintage-end", default=None, help="Upper bound (e.g., 2024Q4 or a parseable date)")

    # If grade is missing, reconstruct it
    p.add_argument("--score-col", default="score_ttc", help="Score column used to rebuild grade if needed")
    p.add_argument("--buckets", default=None, help="risk_buckets.json used to rebuild grade if needed")
    p.add_argument("--n-buckets", type=int, default=10, help="Number of grades/buckets used during training")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Load and concatenate inputs
    dfs = [load_any(p) for p in args.scored]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # Ensure grade exists (reconstruct if needed)
    df = ensure_grade(
        df,
        grade_col=args.grade_col,
        score_col=args.score_col,
        buckets_json=args.buckets,
        n_buckets=args.n_buckets,
    )

    # ------------------------------------------------------------------
    # Optional time filtering (rolling window or explicit bounds)
    # ------------------------------------------------------------------
    if args.time_col is not None and (args.window_years or args.vintage_start or args.vintage_end):
        per = parse_vintage_to_period(df[args.time_col], freq=args.time_freq)
        df = df.copy()
        df["_period"] = per

        if args.vintage_start:
            p0 = parse_vintage_to_period(pd.Series([args.vintage_start]), freq=args.time_freq)[0]
            df = df[df["_period"] >= p0]

        if args.vintage_end:
            p1 = parse_vintage_to_period(pd.Series([args.vintage_end]), freq=args.time_freq)[0]
            df = df[df["_period"] <= p1]

        if args.window_years:
            maxp = df["_period"].max()
            keep = args.window_years * (4 if args.time_freq == "Q" else 12)
            minp = (maxp - keep + 1)
            df = df[df["_period"] >= minp]

        df = df.drop(columns=["_period"], errors="ignore")

    # Compute recalibrated PD table
    payload = compute_pd_table(
        df=df,
        target_col=args.target,
        grade_col=args.grade_col,
        method=args.method,
        aggregation=args.aggregation,
        time_col=args.time_col,
        time_freq=args.time_freq,
        smooth=args.smooth,
        score_col=args.score_col,
    )

    # Write JSON output
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"✔ Recalibration written to: {outp}")
    print(f"  - method={payload['method']} aggregation={payload['aggregation']} monotone={payload['monotone_pd']}")


if __name__ == "__main__":
    main()
