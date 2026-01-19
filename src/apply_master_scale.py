#!/usr/bin/env python3
#src/apply_master_scale.py
# -*- coding: utf-8 -*-

"""
Attach a *master-scale* PD to observations based on their grade.

This script takes:
- an input dataset containing a grade column (e.g., 1..K),
- a JSON file containing bucket-level PD estimates (e.g., produced during calibration),
and outputs the same dataset with an additional PD column mapped from grade -> PD.

Typical use case
----------------
After calibrating / smoothing grade-level PDs (e.g., with isotonic regression), you
want to "push down" the calibrated PD to the observation level to obtain a scored
dataset with a consistent master scale.

Notes
-----
- The JSON is expected to contain a list under payload["train"], where each element
  provides at least:
    - "bucket": the grade/bucket identifier (integer)
    - "pd": the (possibly smoothed) PD for that bucket
- If --replace-pd is enabled, the script overwrites an existing column named "pd"
  with the master-scale PD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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


def save_any(df: pd.DataFrame, path: Path) -> None:
    """
    Save a dataset to disk (parquet or csv based on the file extension).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Attach master-scale PDs based on grades.")
    ap.add_argument("--in-scored", required=True, help="Input file (parquet/csv) with at least a grade column")
    ap.add_argument("--out", required=True, help="Output file (parquet/csv)")
    ap.add_argument("--bucket-stats", required=True, help="Path to bucket_stats_recalibrated.json")
    ap.add_argument("--grade-col", default="grade", help="Name of the grade column in the input dataset")
    ap.add_argument("--pd-col-out", default="pd_ms", help="Name of the master-scale PD column to create")
    ap.add_argument(
        "--replace-pd",
        action="store_true",
        help="If set, overwrite the existing 'pd' column with the master-scale PD",
    )
    args = ap.parse_args()

    # 1) Load input data
    df = load_any(args.in_scored)

    if args.grade_col not in df.columns:
        raise ValueError(f"Missing grade column: {args.grade_col}")

    # 2) Load bucket-level PDs and build a grade -> PD mapping
    payload = json.loads(Path(args.bucket_stats).read_text(encoding="utf-8"))

    # Expected structure: payload["train"] is a list of dicts with keys "bucket" and "pd".
    grade_to_pd = {int(r["bucket"]): float(r["pd"]) for r in payload["train"]}

    # 3) Apply mapping to create the master-scale PD column
    out = df.copy()
    out[args.pd_col_out] = out[args.grade_col].map(grade_to_pd)

    # 4) Safety check: fail fast if some grades are not present in the mapping
    miss = out[args.pd_col_out].isna().mean()
    if miss > 0:
        raise ValueError(f"{miss:.2%} of rows have a grade with no mapping in bucket-stats.")

    # 5) Optional: overwrite an existing 'pd' column
    if args.replace_pd:
        out["pd"] = out[args.pd_col_out]

    # 6) Save results
    outp = Path(args.out)
    save_any(out, outp)

    print(f"✔ Master scale applied: {outp} (column={args.pd_col_out})")


if __name__ == "__main__":
    main()
