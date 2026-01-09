# -*- coding: utf-8 -*-
"""
Build compact scenario tables for the Beta–Binomial robustness experiment.

- Reads:   data/beta_binom_results.csv
- Writes: tables/beta_binom_scenarios_long.csv
          tables/beta_binom_scenarios_table.csv

Enhancements vs the draft:
- Adds Monte Carlo standard error for coverage: coverage_se
- Adds +/- 1.96*SE bands: coverage_ci_lo/coverage_ci_hi
- Also exports SD (sqrt(var)) for bounds/length: lb_sd, ub_sd, len_sd
- Robust column fallbacks (reject rate, length column names)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Labels (as they appear in beta_binom_results.csv)
# ---------------------------------------------------------------------
METHOD_LABELS: Dict[str, str] = {
    "jeffreys": "Jeffreys",
    "cp": "Clopper–Pearson",
    "normal": "Normal Approximation",
}

# ---------------------------------------------------------------------
# Scenarios (n, np_target, rho)
# Here we keep a minimal robustness design: i.i.d. vs clustered, baseline vs low-default.
# ---------------------------------------------------------------------
SCENARIOS: List[dict] = [
    {"scenario": "Baseline",                "n": 1000, "np_target": 10.0, "rho": 0.00},  # p=0.01
    {"scenario": "Baseline (clustered)",    "n": 1000, "np_target": 10.0, "rho": 0.05},  # p=0.01
    {"scenario": "Low Default",             "n": 1000, "np_target": 1.0,  "rho": 0.00},  # p=0.001
    {"scenario": "Low Default (clustered)", "n": 1000, "np_target": 1.0,  "rho": 0.05},  # p=0.001
]

DECIMALS = 3
Z_95 = 1.96


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _fmt_interval(len_mean: float, lb_mean: float, ub_mean: float) -> str:
    return (
        f"({len_mean:.{DECIMALS}f}) "
        f"[{lb_mean:.{DECIMALS}f}, {ub_mean:.{DECIMALS}f}]"
    )


def _get_col(df: pd.DataFrame, primary: str, fallback: str) -> str:
    if primary in df.columns:
        return primary
    if fallback in df.columns:
        return fallback
    raise KeyError(f"Missing both '{primary}' and '{fallback}' in results CSV.")


def _get_first_existing(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def _require_cols(df: pd.DataFrame, cols: List[str], context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = "Missing columns in beta_binom_results.csv: " + ", ".join(missing)
        if context:
            msg += f". {context}"
        raise KeyError(msg)


def _mc_se_coverage(cov: float, n_sim: int) -> float:
    # Binomial Monte Carlo SE for a proportion estimate
    if n_sim <= 0:
        return float("nan")
    cov = float(np.clip(cov, 0.0, 1.0))
    return float(np.sqrt(cov * (1.0 - cov) / n_sim))


# ---------------------------------------------------------------------
# Core: build long + wide tables
# ---------------------------------------------------------------------
def build_tables(df: pd.DataFrame, scenarios: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Allow backward compatibility
    len_col = _get_first_existing(df, ["len_mean", "avg_length", "length_mean", "interval_len_mean"])
    reject_col = _get_first_existing(df, ["reject_star_rate", "reject_rate", "rejection_rate"])

    # Required base columns
    _require_cols(
        df,
        ["n", "np_target", "rho", "p_true", "conf_level", "n_sim", "method", "coverage"],
        context="Check sim_beta_binom.py output schema.",
    )

    # Required bound/length dispersion columns for SD/Var reporting
    _require_cols(
        df,
        ["lb_mean", "ub_mean", "lb_var", "ub_var", "len_var"],
        context="Did you update sim_beta_binom.py to store bound means/vars?",
    )

    long_rows: List[dict] = []

    for sc in scenarios:
        sub = df[
            (df["n"] == sc["n"])
            & (np.isclose(df["np_target"], sc["np_target"]))
            & (np.isclose(df["rho"], sc["rho"]))
        ].copy()

        if sub.empty:
            raise ValueError(
                f"No rows found for scenario={sc}. "
                f"Available n: {sorted(df['n'].unique())}, "
                f"np_target: {sorted(df['np_target'].unique())[:10]}..., "
                f"rho: {sorted(df['rho'].unique())}"
            )

        for method in sorted(sub["method"].unique()):
            if method not in METHOD_LABELS:
                continue

            row = sub[sub["method"] == method].iloc[0]

            coverage = float(row["coverage"])
            n_sim = int(row["n_sim"])
            coverage_se = _mc_se_coverage(coverage, n_sim)

            lb_var = float(row["lb_var"])
            ub_var = float(row["ub_var"])
            len_var = float(row["len_var"])

            # guard tiny negative variances from numerical rounding
            lb_sd = float(np.sqrt(max(lb_var, 0.0)))
            ub_sd = float(np.sqrt(max(ub_var, 0.0)))
            len_sd = float(np.sqrt(max(len_var, 0.0)))

            ci_lo = float(np.clip(coverage - Z_95 * coverage_se, 0.0, 1.0))
            ci_hi = float(np.clip(coverage + Z_95 * coverage_se, 0.0, 1.0))

            long_rows.append(
                {
                    "scenario": sc["scenario"],
                    "n": int(row["n"]),
                    "np_target": float(row["np_target"]),
                    "p_true": float(row["p_true"]),
                    "rho": float(row["rho"]),
                    "conf_level": float(row["conf_level"]),
                    "n_sim": n_sim,
                    "method": METHOD_LABELS[method],
                    "coverage": coverage,
                    "coverage_se": coverage_se,
                    "coverage_ci_lo": ci_lo,
                    "coverage_ci_hi": ci_hi,
                    "reject_rate": float(row[reject_col]),
                    "lb_mean": float(row["lb_mean"]),
                    "ub_mean": float(row["ub_mean"]),
                    "lb_var": lb_var,
                    "ub_var": ub_var,
                    "len_mean": float(row[len_col]),
                    "len_var": len_var,
                    "lb_sd": lb_sd,
                    "ub_sd": ub_sd,
                    "len_sd": len_sd,
                }
            )

    df_long = pd.DataFrame(long_rows)

    # Wide table for paper / export
    base_cols = ["scenario", "n", "np_target", "p_true", "rho", "conf_level", "n_sim"]
    wide = (
        df_long[base_cols]
        .drop_duplicates()
        .sort_values(["scenario", "n", "rho"])
        .copy()
    )

    # Merge method-specific blocks
    for method_label in sorted(df_long["method"].unique()):
        subm = df_long[df_long["method"] == method_label].copy()
        subm["interval"] = subm.apply(
            lambda r: _fmt_interval(r["len_mean"], r["lb_mean"], r["ub_mean"]), axis=1
        )

        # Keep compact but informative: interval, coverage, MC SE, reject_rate, and SD of length (optional)
        keep = base_cols + [
            "interval",
            "coverage",
            "coverage_se",
            "coverage_ci_lo",
            "coverage_ci_hi",
            "reject_rate",
            "lb_sd",
            "ub_sd",
            "len_sd",
        ]

        subm = subm[keep].rename(
            columns={
                "interval": f"{method_label}",
                "coverage": f"{method_label}__coverage",
                "coverage_se": f"{method_label}__coverage_se",
                "coverage_ci_lo": f"{method_label}__coverage_ci_lo",
                "coverage_ci_hi": f"{method_label}__coverage_ci_hi",
                "reject_rate": f"{method_label}__reject_rate",
                "lb_sd": f"{method_label}__lb_sd",
                "ub_sd": f"{method_label}__ub_sd",
                "len_sd": f"{method_label}__len_sd",
            }
        )

        wide = wide.merge(subm, on=base_cols, how="left")

    return df_long, wide


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    out_dir = base_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_path = data_dir / "beta_binom_results.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {df_path}")

    df = pd.read_csv(df_path)

    df_long, df_wide = build_tables(df, SCENARIOS)

    out_long = out_dir / "beta_binom_scenarios_long.csv"
    out_wide = out_dir / "beta_binom_scenarios_table.csv"

    df_long.to_csv(out_long, index=False)
    df_wide.to_csv(out_wide, index=False)

    print(f"Saved: {out_long}")
    print(f"Saved: {out_wide}")


if __name__ == "__main__":
    main()
