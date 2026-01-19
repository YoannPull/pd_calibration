#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S&P (monthly snapshot / corporate.xlsx-like) — PER GRADE benchmark tables

Input (new format):
  rating_agency_name | rating | rating_action_date | legal_entity_identifier | obligor_name
  year_month | year | pays | nace

Outputs (outdir):
  - sp_grade_table_YYYY.csv (one per OOS year)
  - sp_grade_tables_{OOS_STARTeff}_{OOS_ENDeff}.csv (combined)
  - sp_grade_overview.json (meta/config)

Key features:
  - Builds issuer-year cohorts (first rating observation in each "year"),
    labels default within HORIZON_MONTHS (default 12).
  - Drops starting-in-default, keeps only observable horizons
  - TTC sources:
      * is     : TTC estimated on IS window (pooled / jeffreys_mean / hybrid)
      * sp2012 : external TTC from S&P Annual Study 2012 (WLTA 1981–2012, Table 4)
  - Adds TTC_KEY column (so you can append newer TTC tables later)
  - If TTC_SOURCE=sp2012: clips OOS_START >= 2012 (warns)

Important adaptation to new data:
  - The input may be .xlsx OR .csv.
  - If a 'year' column exists, we use it (this matches the "corporate.xlsx" schema).
    Otherwise we fall back to rating_action_date.dt.year (calendar year).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

# -------------------------------------------------------------------
# Import interval functions from your repo file: experiments/stats/intervals.py
# -------------------------------------------------------------------
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stats.intervals import (  # noqa: E402
    jeffreys_alpha2,
    approx_normal,
    exact_cp,
    jeffreys_pvalue_unilateral,
)

# ============================================================
# External TTC tables (S&P Annual Study 2012)
#   Table 4 "Weighted long-term average" over 1981–2012
#   Values below are decimals (percent / 100)
# ============================================================
SP2012_TTC_KEY = "SP_Annual_Study_2012_1981_2012_Table4_WLTA"
SP2012_TTC_MAJOR = {
    "AAA": 0.00 / 100.0,
    "AA": 0.02 / 100.0,
    "A": 0.07 / 100.0,
    "BBB": 0.22 / 100.0,
    "BB": 0.86 / 100.0,
    "B": 4.28 / 100.0,
    "CCC/C": 26.85 / 100.0,
}


# ============================================================
# Helpers
# ============================================================
def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _upper_set(xs: Iterable[str] | None) -> set[str]:
    return {str(x).strip().upper() for x in (xs or []) if str(x).strip()}


def add_intervals_and_pval(tab: pd.DataFrame, *, conf_level: float = 0.95) -> pd.DataFrame:
    """
    tab must contain columns: n, d, TTC
    adds: jeff_LB/UB, normal_LB/UB, cp_LB/UB, IC_* strings, p_jeff_P_p_lt_TTC
    """
    tab = tab.copy()

    jeff = tab.apply(lambda r: jeffreys_alpha2(int(r["n"]), int(r["d"]), confidence_level=conf_level), axis=1)
    tab["jeff_LB"], tab["jeff_UB"] = zip(*jeff)

    norm_ci = tab.apply(lambda r: approx_normal(int(r["n"]), int(r["d"]), confidence_level=conf_level), axis=1)
    tab["normal_LB"], tab["normal_UB"] = zip(*norm_ci)
    tab["normal_LB"] = tab["normal_LB"].map(_clip01)
    tab["normal_UB"] = tab["normal_UB"].map(_clip01)

    cp = tab.apply(lambda r: exact_cp(int(r["n"]), int(r["d"]), confidence_level=conf_level), axis=1)
    tab["cp_LB"], tab["cp_UB"] = zip(*cp)

    tab["p_jeff_P_p_lt_TTC"] = tab.apply(
        lambda r: jeffreys_pvalue_unilateral(int(r["n"]), int(r["d"]), float(r["TTC"]), tail="lower")
        if pd.notna(r["TTC"])
        else np.nan,
        axis=1,
    )

    tab["IC_Jeffreys"] = tab.apply(lambda r: f"[{r['jeff_LB']:.6f}, {r['jeff_UB']:.6f}]", axis=1)
    tab["IC_Normal"] = tab.apply(lambda r: f"[{r['normal_LB']:.6f}, {r['normal_UB']:.6f}]", axis=1)
    tab["IC_CP"] = tab.apply(lambda r: f"[{r['cp_LB']:.6f}, {r['cp_UB']:.6f}]", axis=1)

    return tab


# ------------------------------------------------------------
# Grade ordering helper (best effort)
# ------------------------------------------------------------
BASE_ORDER = [
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B",
    "B-",
    "CCC+",
    "CCC",
    "CCC-",
    "CC",
    "C",
]
GRADE_RANK = {g: i for i, g in enumerate(BASE_ORDER)}


def grade_rank(g: str) -> int:
    g = str(g).strip().upper()
    if g == "RD":
        return len(BASE_ORDER) + 1
    if g == "D":
        return len(BASE_ORDER) + 2
    return int(GRADE_RANK.get(g, 10_000))


def ttc_estimate(sum_d: int, sum_n: int, estimator: str) -> float:
    if sum_n <= 0:
        return np.nan

    est = str(estimator).strip().lower()
    if est == "pooled":
        return float(sum_d) / float(sum_n)
    if est == "jeffreys_mean":
        return (float(sum_d) + 0.5) / (float(sum_n) + 1.0)
    if est == "pooled_if_nonzero_else_jeffreys":
        if int(sum_d) > 0:
            return float(sum_d) / float(sum_n)
        return (float(sum_d) + 0.5) / (float(sum_n) + 1.0)

    raise ValueError("ttc_estimator must be in {'pooled','jeffreys_mean','pooled_if_nonzero_else_jeffreys'}")


# ============================================================
# Mapping hook (notched grade -> major grade used in external TTC tables)
# Must return one of: {"AAA","AA","A","BBB","BB","B","CCC/C"} or None
# ============================================================
def map_notch_to_sp_major(grade: str) -> str | None:
    g = str(grade).strip().upper()
    if g == "AAA":
        return "AAA"
    if g in {"AA+", "AA", "AA-"}:
        return "AA"
    if g in {"A+", "A", "A-"}:
        return "A"
    if g in {"BBB+", "BBB", "BBB-"}:
        return "BBB"
    if g in {"BB+", "BB", "BB-"}:
        return "BB"
    if g in {"B+", "B", "B-"}:
        return "B"
    if g in {"CCC+", "CCC", "CCC-", "CC", "C"}:
        return "CCC/C"
    return None


def build_external_ttc_map_sp2012(grades: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for g in grades:
        major = map_notch_to_sp_major(g)
        out[g] = SP2012_TTC_MAJOR.get(major, np.nan)
    return out


# ============================================================
# Input loader (xlsx or csv)
# ============================================================
def load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suf == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input extension {path.suffix}. Use .xlsx or .csv")

    return df


# ============================================================
# Cohort builder
# ============================================================
def build_sp_cohorts(
    input_path: Path,
    *,
    agency: str,
    horizon_months: int,
    default_rating: str,
    grade_whitelist: set[str] | None,
    prefer_year_column: bool = True,
) -> pd.DataFrame:
    """
    Build issuer-year cohorts from the monthly snapshot.

    - cohort year is:
        * df['year'] if present and prefer_year_column=True
        * else rating_action_date.dt.year
    - cohort grade is the FIRST observation in that year (per obligor)
    - default is flagged if a 'D' occurs within (cohort_date, cohort_date + horizon]
    """
    df = load_snapshot(input_path)

    needed = {"rating_agency_name", "rating_action_date", "rating", "obligor_name"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in snapshot: {sorted(missing)}")

    df = df[df["rating_agency_name"] == agency].copy()

    df["rating_action_date"] = pd.to_datetime(df["rating_action_date"], errors="coerce")
    df = df.dropna(subset=["rating_action_date"]).copy()
    df["rating"] = df["rating"].astype(str).str.strip().str.upper()

    # obligor_id = LEI if available else obligor_name__country
    lei = df.get("legal_entity_identifier", pd.Series([pd.NA] * len(df))).astype("string")
    lei = lei.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "none": pd.NA})

    pays = df.get("pays", pd.Series([pd.NA] * len(df))).astype("string")
    pays = pays.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "none": pd.NA}).fillna("XX").str.strip()

    obligor_name = df.get("obligor_name", pd.Series([pd.NA] * len(df))).astype("string")
    obligor_name = obligor_name.fillna("UNKNOWN").str.strip()

    fallback = obligor_name + "__" + pays
    df["obligor_id"] = np.where(lei.notna(), lei, fallback)

    # Year definition: prefer 'year' column if provided by snapshot
    if prefer_year_column and ("year" in df.columns):
        y = pd.to_numeric(df["year"], errors="coerce")
        if y.notna().any():
            df["year"] = y.astype("Int64").astype(int)  # robust across pandas versions
        else:
            df["year"] = df["rating_action_date"].dt.year.astype(int)
    else:
        df["year"] = df["rating_action_date"].dt.year.astype(int)

    df = df.sort_values(["obligor_id", "rating_action_date"]).copy()

    offset = DateOffset(months=int(horizon_months))
    default_rating = str(default_rating).strip().upper()

    def per_obligor(grp: pd.DataFrame) -> pd.DataFrame:
        oid = str(getattr(grp, "name", "UNKNOWN_OBLIGOR"))

        dates = grp["rating_action_date"].to_numpy(dtype="datetime64[ns]")
        ratings = grp["rating"].to_numpy()

        d_pos = np.where(ratings == default_rating)[0]
        d_dates = dates[d_pos] if len(d_pos) else np.array([], dtype="datetime64[ns]")
        last_date = dates[-1]

        # First observation in each year defines the cohort entry for that year
        cohorts = grp.sort_values("rating_action_date").drop_duplicates("year", keep="first")[
            ["year", "rating_action_date", "rating"]
        ]

        rows = []
        for _, r in cohorts.iterrows():
            cohort_date = r["rating_action_date"]
            cohort_date64 = cohort_date.to_datetime64()
            horizon_end64 = (cohort_date + offset).to_datetime64()

            if len(d_dates):
                idx = np.searchsorted(d_dates, cohort_date64, side="right")
                default_h = bool(idx < len(d_dates) and d_dates[idx] <= horizon_end64)
            else:
                default_h = False

            observable_h = bool((last_date >= horizon_end64) or default_h)

            rows.append(
                {
                    "obligor_id": oid,
                    "year": int(r["year"]),
                    "cohort_date": cohort_date,
                    "grade": str(r["rating"]).strip().upper(),
                    "default_12m": int(default_h),
                    "observable_12m": int(observable_h),
                }
            )
        return pd.DataFrame(rows)

    # pandas compatibility (include_groups exists only in newer versions)
    try:
        cohorts = (
            df.groupby("obligor_id", group_keys=False)
            .apply(per_obligor, include_groups=False)  # type: ignore[call-arg]
            .reset_index(drop=True)
        )
    except TypeError:
        cohorts = df.groupby("obligor_id", group_keys=False).apply(per_obligor).reset_index(drop=True)

    # Drop starting in default and keep only observable horizons
    cohorts = cohorts[cohorts["grade"] != default_rating].copy()
    cohorts = cohorts[cohorts["observable_12m"] == 1].copy()

    if grade_whitelist is not None:
        wl = {str(g).strip().upper() for g in grade_whitelist}
        cohorts = cohorts[cohorts["grade"].isin(wl)].copy()

    return cohorts


# ============================================================
# TTC + OOS tables
# ============================================================
def build_ttc_is_per_grade(cohorts: pd.DataFrame, *, is_start: int, is_end: int, ttc_estimator: str) -> pd.DataFrame:
    coh_is = cohorts[(cohorts["year"] >= int(is_start)) & (cohorts["year"] <= int(is_end))].copy()
    if coh_is.empty:
        raise ValueError("IS window yields empty cohorts after filtering.")

    ttc = (
        coh_is.groupby("grade", as_index=False).agg(
            n_is=("obligor_id", "nunique"),
            d_is=("default_12m", "sum"),
        )
    )
    ttc["TTC_IS"] = ttc.apply(lambda r: ttc_estimate(int(r["d_is"]), int(r["n_is"]), ttc_estimator), axis=1)
    ttc["rank"] = ttc["grade"].map(grade_rank).astype(int)
    ttc = ttc.sort_values(["rank", "grade"]).drop(columns=["rank"]).reset_index(drop=True)
    return ttc


def build_oos_tables_per_grade(
    cohorts: pd.DataFrame,
    *,
    oos_start: int,
    oos_end: int,
    ttc_map: dict[str, float],
    conf_level: float,
    ttc_key: str,
    drop_without_ttc: bool,
) -> dict[int, pd.DataFrame]:
    tables: dict[int, pd.DataFrame] = {}
    for y in range(int(oos_start), int(oos_end) + 1):
        df_y = cohorts[cohorts["year"] == y].copy()
        if df_y.empty:
            tables[y] = pd.DataFrame()
            continue

        tab = (
            df_y.groupby("grade", as_index=False)
            .agg(
                n=("obligor_id", "nunique"),
                d=("default_12m", "sum"),
            )
        )
        tab["default_rate"] = np.where(tab["n"] > 0, tab["d"] / tab["n"], np.nan)
        tab["TTC"] = tab["grade"].map(ttc_map)
        tab["TTC_KEY"] = str(ttc_key)

        if drop_without_ttc:
            tab = tab[pd.notna(tab["TTC"])].copy()

        tab = add_intervals_and_pval(tab, conf_level=float(conf_level))

        tab["rank"] = tab["grade"].map(grade_rank).fillna(10_000).astype(int)
        tab = tab.sort_values(["rank", "grade"]).drop(columns=["rank"]).reset_index(drop=True)
        tables[y] = tab

    return tables


def save_tables(tables: dict[int, pd.DataFrame], outdir: Path, *, oos_start: int, oos_end: int) -> Path:
    _safe_mkdir(outdir)
    combined = []
    for y in range(int(oos_start), int(oos_end) + 1):
        tab = tables.get(y, pd.DataFrame())
        if tab is None or tab.empty:
            continue
        tab2 = tab.copy()
        tab2.insert(0, "year", int(y))
        combined.append(tab2)
        tab2.to_csv(outdir / f"sp_grade_table_{y}.csv", index=False)

    combined_path = outdir / f"sp_grade_tables_{oos_start}_{oos_end}.csv"
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(combined_path, index=False)
    else:
        pd.DataFrame().to_csv(combined_path, index=False)
    return combined_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S&P per-grade OOS tables with TTC benchmarks (snapshot input).")
    parser.add_argument("--file", type=str, default="ldp_application/data/processed/sp_corporate_monthly.xlsx")
    parser.add_argument("--outdir", type=str, default="ldp_application/outputs/sp_grade_is_oos")

    parser.add_argument("--agency", type=str, default="Standard & Poor's Ratings Services")
    parser.add_argument("--horizon-months", type=int, default=12)
    parser.add_argument("--default-rating", type=str, default="D")
    parser.add_argument("--confidence-level", type=float, default=0.95)

    parser.add_argument("--is-start-year", type=int, default=2010)
    parser.add_argument("--is-end-year", type=int, default=2018)
    parser.add_argument("--oos-start-year", type=int, default=2019)
    parser.add_argument("--oos-end-year", type=int, default=2020)

    parser.add_argument("--grade-whitelist", type=str, nargs="*", default=None)

    parser.add_argument("--ttc-source", type=str, default="is", choices=["is", "sp2012"])
    parser.add_argument(
        "--ttc-estimator",
        type=str,
        default="pooled",
        choices=["pooled", "jeffreys_mean", "pooled_if_nonzero_else_jeffreys"],
        help="Only used if --ttc-source is 'is'",
    )
    parser.add_argument("--drop-grades-without-ttc", action="store_true")
    parser.add_argument("--keep-grades-without-ttc", action="store_true")

    # New: prefer year column from snapshot (recommended)
    parser.add_argument(
        "--prefer-year-column",
        action="store_true",
        help="If set, use the input 'year' column when available (recommended for corporate.xlsx-like data).",
    )
    parser.add_argument(
        "--no-prefer-year-column",
        action="store_true",
        help="If set, ignore input 'year' column and use rating_action_date.year (calendar year).",
    )

    args = parser.parse_args()

    input_path = Path(args.file).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    _safe_mkdir(outdir)

    wl = _upper_set(args.grade_whitelist) if args.grade_whitelist else None

    prefer_year = bool(args.prefer_year_column) and (not bool(args.no_prefer_year_column))

    cohorts = build_sp_cohorts(
        input_path,
        agency=str(args.agency),
        horizon_months=int(args.horizon_months),
        default_rating=str(args.default_rating),
        grade_whitelist=wl,
        prefer_year_column=prefer_year,
    )
    if cohorts.empty:
        print("[ERR] cohorts empty after filtering.", file=sys.stderr)
        return 2

    all_grades = sorted(cohorts["grade"].unique().tolist(), key=grade_rank)

    ttc_source = str(args.ttc_source).strip().lower()
    drop_without_ttc = bool(args.drop_grades_without_ttc) and (not bool(args.keep_grades_without_ttc))

    # Decide OOS effective window + TTC map + key
    if ttc_source == "is":
        ttc_is = build_ttc_is_per_grade(
            cohorts,
            is_start=int(args.is_start_year),
            is_end=int(args.is_end_year),
            ttc_estimator=str(args.ttc_estimator),
        )
        ttc_map = dict(zip(ttc_is["grade"], ttc_is["TTC_IS"]))
        ttc_key = f"IS_{args.is_start_year}_{args.is_end_year}_{args.ttc_estimator}"
        oos_start_eff = int(args.oos_start_year)
        oos_end_eff = int(args.oos_end_year)

    else:  # sp2012
        oos_start_eff = max(int(args.oos_start_year), 2012)
        oos_end_eff = int(args.oos_end_year)
        if int(args.oos_start_year) < 2012:
            print(f"[WARN] TTC_SOURCE=sp2012 => OOS_START clipped {args.oos_start_year} -> {oos_start_eff}")
        ttc_map = build_external_ttc_map_sp2012(all_grades)
        ttc_key = SP2012_TTC_KEY

    if oos_end_eff < oos_start_eff:
        print(f"[ERR] Invalid OOS window after clipping: {oos_start_eff}..{oos_end_eff}", file=sys.stderr)
        return 2

    tables = build_oos_tables_per_grade(
        cohorts,
        oos_start=oos_start_eff,
        oos_end=oos_end_eff,
        ttc_map=ttc_map,
        conf_level=float(args.confidence_level),
        ttc_key=ttc_key,
        drop_without_ttc=drop_without_ttc,
    )

    combined_path = save_tables(tables, outdir, oos_start=oos_start_eff, oos_end=oos_end_eff)

    meta = {
        "input_file": str(input_path),
        "outdir": str(outdir),
        "agency": str(args.agency),
        "horizon_months": int(args.horizon_months),
        "default_rating": str(args.default_rating),
        "confidence_level": float(args.confidence_level),
        "prefer_year_column": bool(prefer_year),
        "is_window": [int(args.is_start_year), int(args.is_end_year)],
        "oos_window_requested": [int(args.oos_start_year), int(args.oos_end_year)],
        "oos_window_effective": [int(oos_start_eff), int(oos_end_eff)],
        "ttc_source": ttc_source,
        "ttc_estimator": str(args.ttc_estimator),
        "ttc_key": str(ttc_key),
        "drop_grades_without_ttc": bool(drop_without_ttc),
        "n_cohort_rows": int(len(cohorts)),
        "n_obligors": int(cohorts["obligor_id"].nunique()),
        "years_min": int(cohorts["year"].min()),
        "years_max": int(cohorts["year"].max()),
        "combined_csv": str(combined_path),
    }
    with (outdir / "sp_grade_overview.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== S&P grade tables ===")
    print(f"Input:  {input_path}")
    print(f"Outdir: {outdir}")
    print(f"TTC:    {ttc_source} | key={ttc_key}")
    if ttc_source == "is":
        print(f"IS:     {args.is_start_year}..{args.is_end_year} | estimator={args.ttc_estimator}")
    print(f"OOS:    {oos_start_eff}..{oos_end_eff}")
    print(f"Written combined: {combined_path}")
    print("Also written:")
    print(" - sp_grade_table_YYYY.csv (per year)")
    print(" - sp_grade_overview.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
