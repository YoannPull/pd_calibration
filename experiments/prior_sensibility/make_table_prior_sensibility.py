# -*- coding: utf-8 -*-
"""
Build scenario tables from prior sensitivity CSV (LONG format).

- tail=two_sided: reports coverage + interval summary
- tail=upper: reports conservatism_mean + upper bound summary
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


SCENARIOS = [
    {"scenario": "Baseline",     "n": 1000,  "p": 0.01},
    {"scenario": "Low Default",  "n": 1000,  "p": 0.001},
    {"scenario": "Small Sample", "n": 100,   "p": 0.01},
    {"scenario": "Large Sample", "n": 10000, "p": 0.01},
]

DECIMALS = 6
PICK_NEAREST_P = True


def _pick_row(df: pd.DataFrame, *, n: int, p: float, prior: str, tail: str) -> pd.Series:
    sub = df[(df["n"] == n) & (df["prior"] == prior) & (df["tail"] == tail)].copy()
    if sub.empty:
        raise ValueError(f"No rows for n={n}, prior={prior}, tail={tail}")

    if not PICK_NEAREST_P:
        hit = sub[np.isclose(sub["p"].values, p)]
        if hit.empty:
            raise ValueError(f"p={p} not found for n={n} (try PICK_NEAREST_P=True)")
        return hit.iloc[0]

    idx = (sub["p"] - p).abs().idxmin()
    return df.loc[idx]


def _safe_sqrt(v: float) -> float:
    return float(np.sqrt(max(float(v), 0.0)))


def _fmt_interval(len_mean: float, lb_mean: float, ub_mean: float) -> str:
    return f"({len_mean:.{DECIMALS}f}) [{lb_mean:.{DECIMALS}f}, {ub_mean:.{DECIMALS}f}]"


def build_tables(df: pd.DataFrame, scenarios: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "n", "p", "conf_level", "prior", "tail",
        "lb_mean", "ub_mean", "len_mean",
        "lb_var", "ub_var", "len_var",
        "n_sims",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

    priors = sorted(df["prior"].unique())
    tails = sorted(df["tail"].unique())

    long_rows: list[dict] = []

    for sc in scenarios:
        for tail in tails:
            for prior in priors:
                row = _pick_row(df, n=sc["n"], p=sc["p"], prior=prior, tail=tail)

                S = int(row["n_sims"]) if "n_sims" in row.index else np.nan
                lb_mean = float(row["lb_mean"])
                ub_mean = float(row["ub_mean"])
                len_mean = float(row["len_mean"])
                lb_var = float(row["lb_var"])
                ub_var = float(row["ub_var"])
                len_var = float(row["len_var"])

                lb_sd = _safe_sqrt(lb_var)
                ub_sd = _safe_sqrt(ub_var)
                len_sd = _safe_sqrt(len_var)

                denom = float(np.sqrt(S)) if np.isfinite(S) and S > 0 else float("nan")
                lb_se = lb_sd / denom if np.isfinite(denom) else float("nan")
                ub_se = ub_sd / denom if np.isfinite(denom) else float("nan")
                len_se = len_sd / denom if np.isfinite(denom) else float("nan")

                # tail-specific metric
                coverage = float(row["coverage"]) if "coverage" in row.index else np.nan
                conserv = float(row["conservatism_mean"]) if "conservatism_mean" in row.index else np.nan

                long_rows.append(
                    {
                        "scenario": sc["scenario"],
                        "n": int(sc["n"]),
                        "p_target": float(sc["p"]),
                        "p_used": float(row["p"]),
                        "conf_level": float(row["conf_level"]),
                        "tail": str(tail),
                        "prior": str(prior),
                        "S": S,

                        "coverage": coverage,
                        "conservatism_mean": conserv,

                        "lb_mean": lb_mean,
                        "ub_mean": ub_mean,
                        "len_mean": len_mean,
                        "lb_var": lb_var,
                        "ub_var": ub_var,
                        "len_var": len_var,
                        "lb_sd": lb_sd,
                        "ub_sd": ub_sd,
                        "len_sd": len_sd,
                        "lb_se": lb_se,
                        "ub_se": ub_se,
                        "len_se": len_se,
                    }
                )

    df_long = pd.DataFrame(long_rows)

    # Wide: one row per (scenario, tail), columns per prior
    base_cols = ["scenario", "n", "p_target", "p_used", "conf_level", "tail", "S"]
    wide = df_long[base_cols].drop_duplicates().sort_values(["scenario", "tail", "n"]).copy()

    for prior in priors:
        subp = df_long[df_long["prior"] == prior].copy()
        subp["interval"] = subp.apply(
            lambda r: _fmt_interval(r["len_mean"], r["lb_mean"], r["ub_mean"]), axis=1
        )

        # choose metric column
        # - two_sided: coverage
        # - upper: conservatism_mean
        metric_name = "coverage"  # default
        if (subp["tail"].unique() == np.array(["upper"])).all():
            metric_name = "conservatism_mean"

        # keep both; user can select in LaTeX as needed
        keep = base_cols + ["interval", "coverage", "conservatism_mean"]
        subp = subp[keep].rename(
            columns={
                "interval": f"{prior}",
                "coverage": f"{prior}__coverage",
                "conservatism_mean": f"{prior}__conservatism_mean",
            }
        )
        wide = wide.merge(subp, on=base_cols, how="left")

    return df_long, wide


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    out_dir = base_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "prior_sensibility_all_n.csv")

    df_long, df_wide = build_tables(df, SCENARIOS)

    out_long = out_dir / "prior_sens_scenarios_long.csv"
    out_wide = out_dir / "prior_sens_scenarios_table.csv"

    df_long.to_csv(out_long, index=False)
    df_wide.to_csv(out_wide, index=False)

    print(f"Saved: {out_long}")
    print(f"Saved: {out_wide}")
