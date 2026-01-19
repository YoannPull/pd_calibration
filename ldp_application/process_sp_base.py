#!/usr/bin/env python3
# ldp_application/process_sp_base.py
# -*- coding: utf-8 -*-
"""
Code translated from the R code of Charles-Emmanuel Prost (Square Management)

Goal:
- Process / clean S&P corporate ratings from the raw ratingshistory CSV.
- NO NACE/country enrichment is performed here.
- Output is a MONTHLY SNAPSHOT (1 row per obligor x month, keeping the most recent rating in the month),
  formatted like `data_rating_corporate.xlsx` (schema only):

    rating_agency_name | rating | rating_action_date | legal_entity_identifier | obligor_name
    year_month | year | pays | nace

Notes:
- We DO NOT generate XLSX (CSV only) to avoid openpyxl/zip timeouts.
- `legal_entity_identifier`, `pays`, `nace` are not available in the raw file, so we output them as empty (NA).
- Rating bucketing is robust to substring issues (AAA vs AA) by matching from the START of the string.

Run:
  poetry run python ldp_application/process_sp_base.py \
    --raw ldp_application/data/raw/20220601_SP_Ratings_Services_Corporate.csv \
    --out_csv  ldp_application/data/processed/sp_corporate_monthly.csv \
    --start 2010-01-01 --end 2021-07-01
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


AGENCY_SP = "Standard & Poor's Ratings Services"

# Regex patterns matched from the START of the rating string (prevents AAA -> AA bug)
# We intentionally map CCC/CC/C* into the coarse "C" bucket.
_BUCKET_REGEX = [
    ("AAA", re.compile(r"^AAA")),
    ("AA",  re.compile(r"^AA")),
    ("A",   re.compile(r"^A")),
    ("BBB", re.compile(r"^BBB")),
    ("BB",  re.compile(r"^BB")),
    ("B",   re.compile(r"^B")),
    ("D",   re.compile(r"^D")),
    ("NR",  re.compile(r"^NR")),
    ("C",   re.compile(r"^(CCC|CC|C)")),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean S&P ratings and export a monthly snapshot in corporate.xlsx-like (schema) format (CSV only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw", type=str, required=True, help="Path to raw S&P Corporate CSV (ratingshistory format)")
    p.add_argument("--out_csv", type=str, required=True, help="Output CSV path (monthly snapshot)")
    p.add_argument(
        "--start",
        type=str,
        default="2010-01-01",
        help="Inclusive start date filter applied on year_month",
    )
    p.add_argument(
        "--end",
        type=str,
        default="2021-07-01",
        help="Exclusive end date filter applied on year_month",
    )
    return p.parse_args()


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Load the raw ratings CSV with safe dtypes."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path, sep=",", header=0, dtype=str, low_memory=False)


def month_floor(dt: pd.Series) -> pd.Series:
    """Floor datetimes to month start (YYYY-MM-01)."""
    return dt.dt.to_period("M").dt.to_timestamp()


def july_to_june_year(dt: pd.Series) -> pd.Series:
    """
    July–June year convention:
      - If month <= 6 -> year = calendar year
      - Else          -> year = calendar year + 1
    """
    return np.where(dt.dt.month <= 6, dt.dt.year, dt.dt.year + 1)


def bucket_rating(raw_rating: str) -> str:
    """
    Convert a raw S&P rating string to a coarse bucket.

    Examples:
      - "AAA" / "AAA-" -> "AAA"
      - "AA+" / "AA"   -> "AA"
      - "BBB-"         -> "BBB"
      - "CCC+" / "CC"  -> "C"
      - "NRprelim"     -> "NR"
      - missing/unknown -> "Other"
    """
    if pd.isna(raw_rating):
        return "Other"
    s = str(raw_rating).strip().upper()
    s = s.split()[0]  # keep first token-like part

    for label, rx in _BUCKET_REGEX:
        if rx.search(s):
            return label
    return "Other"


def clean_event_panel(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Clean the *event-level* panel (rating actions).

    Steps:
    1) Filter to S&P + issuer credit rating + local currency LT
    2) Bucket ratings
    3) Parse dates, drop missing obligor, drop NR
    4) If (obligor, date) has D + another rating, keep only D
    5) Build year_month and filter [start, end)
    6) Build July–June year
    7) Keep only first default per obligor and drop post-default ratings
    """
    required_cols = [
        "rating_agency_name",
        "rating",
        "rating_action_date",
        "rating_type",
        "rating_sub_type",
        "obligor_name",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw CSV: {missing}")

    df = df.copy()

    # 1) Restrict to S&P and to a consistent issuer rating definition
    df = df[df["rating_agency_name"] == AGENCY_SP].copy()
    df = df[
        (df["rating_type"] == "Issuer Credit Rating")
        & (df["rating_sub_type"] == "Local Currency LT")
    ].copy()

    # 2) Bucket ratings (robust AAA/AA handling)
    df["rating"] = df["rating"].map(bucket_rating)

    # 3) Parse rating_action_date
    df["rating_action_date"] = pd.to_datetime(df["rating_action_date"], errors="coerce")
    df = df[~df["rating_action_date"].isna()].copy()

    # Drop empty obligors
    df["obligor_name"] = df["obligor_name"].fillna("").astype(str)
    df = df[df["obligor_name"].str.strip() != ""].copy()

    # Drop NR
    df = df[df["rating"] != "NR"].copy()

    # 4) Same-day duplicates: if D is present, keep only D
    g = df.groupby(["obligor_name", "rating_action_date"])["rating"]
    has_d = g.transform(lambda s: (s == "D").any())
    df = df[~(has_d & (df["rating"] != "D"))].copy()

    # 5) year_month filtering
    df["year_month"] = month_floor(df["rating_action_date"])
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    df = df[(df["year_month"] >= start_dt) & (df["year_month"] < end_dt)].copy()

    # 6) Sort and compute July–June year
    df = df.sort_values(["obligor_name", "rating_action_date"]).copy()
    df["year"] = july_to_june_year(df["rating_action_date"]).astype(int)

    # 7) Default handling: keep first D and drop post-default ratings
    default_date = (
        df.loc[df["rating"] == "D"]
        .groupby("obligor_name")["rating_action_date"]
        .min()
        .rename("default_date")
    )
    df = df.merge(default_date, on="obligor_name", how="left")
    df["is_first_default"] = (df["rating"] == "D") & (df["rating_action_date"] == df["default_date"])
    df = df[~((df["rating"] == "D") & (~df["is_first_default"]))].copy()

    df["post_default"] = (~df["default_date"].isna()) & (df["rating_action_date"] > df["default_date"])
    df = df[~df["post_default"]].copy()

    # Keep only needed columns
    df = df[
        ["rating_agency_name", "rating", "rating_action_date", "obligor_name", "year_month", "year"]
    ].copy()

    return df


def monthly_snapshot(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Create a monthly snapshot:
    - 1 row per obligor_name x year_month
    - keep the most recent rating_action_date within the month
    """
    df = df_events.sort_values(["obligor_name", "year_month", "rating_action_date"]).copy()
    snap = (
        df.groupby(["obligor_name", "year_month"], as_index=False)
        .tail(1)
        .copy()
    )
    return snap


def format_like_corporate_xlsx_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder and add missing columns to match the *schema* of corporate.xlsx:
      rating_agency_name, rating, rating_action_date, legal_entity_identifier,
      obligor_name, year_month, year, pays, nace
    """
    out = df.copy()

    # Add missing columns as NA (because we do NOT enrich here)
    out["legal_entity_identifier"] = pd.NA
    out["pays"] = pd.NA
    out["nace"] = pd.NA

    cols = [
        "rating_agency_name",
        "rating",
        "rating_action_date",
        "legal_entity_identifier",
        "obligor_name",
        "year_month",
        "year",
        "pays",
        "nace",
    ]
    return out[cols].copy()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw)
    out_csv = Path(args.out_csv)

    raw = load_raw(raw_path)

    # Clean event-level panel
    events = clean_event_panel(raw, start=args.start, end=args.end)

    print("\n[Events] Rows:", f"{len(events):,}")
    print("[Events] Unique obligors:", f"{events['obligor_name'].nunique():,}")
    print("[Events] Rating distribution (events):")
    print(events["rating"].value_counts(dropna=False))

    # Build monthly snapshot
    snap = monthly_snapshot(events)

    print("\n[Monthly snapshot] Rows:", f"{len(snap):,}")
    print("[Monthly snapshot] Unique obligors:", f"{snap['obligor_name'].nunique():,}")
    print("[Monthly snapshot] Rating distribution (snapshot):")
    print(snap["rating"].value_counts(dropna=False))

    # Format to corporate.xlsx-like schema
    snap_fmt = format_like_corporate_xlsx_schema(snap)

    # Write outputs (CSV only)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    snap_fmt.to_csv(out_csv, index=False)

    print("\n[OK] Wrote:")
    print(" -", out_csv)


if __name__ == "__main__":
    main()
