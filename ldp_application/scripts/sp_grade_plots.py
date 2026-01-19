#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S&P — PER GRADE plots from the combined tables CSV.

Reads combined CSV produced by sp_grade_tables.py:
  - expects columns: year, grade, default_rate, TTC, jeff_LB, jeff_UB, n
  - optional: TTC_KEY (used for labels)

Outputs (outdir):
  - plots_timeseries/sp_grade_timeseries_{GRADE}.png
  - plots_timeseries/sp_grade_timeseries_all_grades.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Grade ordering helper (same as tables)
# ------------------------------------------------------------
BASE_ORDER = [
    "AAA",
    "AA+", "AA", "AA-",
    "A+", "A", "A-",
    "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-",
    "B+", "B", "B-",
    "CCC+", "CCC", "CCC-",
    "CC", "C",
]
GRADE_RANK = {g: i for i, g in enumerate(BASE_ORDER)}


def grade_rank(g: str) -> int:
    g = str(g).strip().upper()
    if g == "RD":
        return len(BASE_ORDER) + 1
    if g == "D":
        return len(BASE_ORDER) + 2
    return int(GRADE_RANK.get(g, 10_000))


def plot_timeseries(
    df_long: pd.DataFrame,
    outdir: Path,
    *,
    p_max: float | None,
    min_total_n: int,
    conf_level: float,
    ttc_label: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    needed = {"year", "grade", "default_rate", "TTC", "jeff_LB", "jeff_UB", "n"}
    missing = needed - set(df_long.columns)
    if missing:
        raise ValueError(f"Missing columns in combined CSV: {sorted(missing)}")

    df_long = df_long.copy()
    df_long["rank"] = df_long["grade"].map(grade_rank).fillna(10_000).astype(int)
    df_long = df_long.sort_values(["rank", "grade", "year"]).drop(columns=["rank"])

    grade_order = df_long["grade"].dropna().unique().tolist()
    grade_order = sorted(grade_order, key=grade_rank)

    tot_n = df_long.groupby("grade", as_index=True)["n"].sum().to_dict()
    keep_grades = [g for g in grade_order if int(tot_n.get(g, 0)) >= int(min_total_n)]

    # Per-grade plots
    for g in keep_grades:
        dg = df_long[df_long["grade"] == g].copy()
        if dg.empty:
            continue
        dg = dg.sort_values("year")

        years = dg["year"].to_numpy()
        dr = dg["default_rate"].to_numpy(dtype=float)
        lb = dg["jeff_LB"].to_numpy(dtype=float)
        ub = dg["jeff_UB"].to_numpy(dtype=float)
        ttc = dg["TTC"].to_numpy(dtype=float)

        plt.figure(figsize=(11, 5))
        plt.plot(years, dr, marker="o", linewidth=1.6, label="Default rate (12m, OOS)")
        plt.plot(years, ttc, marker="s", linewidth=1.6, label=ttc_label)
        plt.fill_between(years, lb, ub, alpha=0.25, label=f"Jeffreys {int(conf_level*100)}% CI")

        plt.title(f"{g} — OOS default rate vs TTC + Jeffreys CI")
        plt.xlabel("Year")
        plt.ylabel("Probability")
        plt.grid(True, alpha=0.25)
        if p_max is not None:
            plt.ylim(0.0, float(p_max))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(outdir / f"sp_grade_timeseries_{g}.png", dpi=220)
        plt.close()

    # Combined plot
    plt.figure(figsize=(13, 7))
    for g in keep_grades:
        dg = df_long[df_long["grade"] == g].copy()
        if dg.empty:
            continue
        dg = dg.sort_values("year")

        years = dg["year"].to_numpy()
        dr = dg["default_rate"].to_numpy(dtype=float)
        lb = dg["jeff_LB"].to_numpy(dtype=float)
        ub = dg["jeff_UB"].to_numpy(dtype=float)
        ttc = dg["TTC"].to_numpy(dtype=float)

        plt.plot(years, dr, marker="o", linewidth=1.3, label=f"{g} default rate")
        plt.fill_between(years, lb, ub, alpha=0.10)
        plt.plot(years, ttc, linestyle="--", linewidth=1.0, alpha=0.7, label=f"{g} TTC")

    plt.title("OOS default rates by grade (with Jeffreys CI) + TTC")
    plt.xlabel("Year")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.25)
    if p_max is not None:
        plt.ylim(0.0, float(p_max))
    plt.legend(loc="best", ncols=3, frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "sp_grade_timeseries_all_grades.png", dpi=220)
    plt.close()

    print(f"[OK] Plots saved in: {outdir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot S&P grade timeseries from combined tables CSV.")
    parser.add_argument("--tables-csv", type=str, required=True, help="Combined CSV produced by sp_grade_tables.py")
    parser.add_argument("--outdir", type=str, default="ldp_application/outputs/sp_grade_is_oos/plots_timeseries")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--p-max", type=float, default=None)
    parser.add_argument("--min-total-n", type=int, default=1)
    parser.add_argument("--ttc-label", type=str, default=None, help="Override TTC label in plots")

    args = parser.parse_args()

    tables_csv = Path(args.tables_csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not tables_csv.exists():
        raise FileNotFoundError(f"Missing tables CSV: {tables_csv}")

    df = pd.read_csv(tables_csv)
    if df.empty:
        print("[WARN] tables CSV empty, nothing to plot.")
        return 0

    # Build a default TTC label from TTC_KEY (if unique)
    ttc_label = args.ttc_label
    if not ttc_label:
        if "TTC_KEY" in df.columns and df["TTC_KEY"].nunique(dropna=True) == 1:
            ttc_label = f"TTC ({df['TTC_KEY'].dropna().iloc[0]})"
        else:
            ttc_label = "TTC"

    plot_timeseries(
        df_long=df,
        outdir=outdir,
        p_max=args.p_max,
        min_total_n=int(args.min_total_n),
        conf_level=float(args.confidence_level),
        ttc_label=str(ttc_label),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
