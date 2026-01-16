# experiments/binom_coverage/make_table_binom.py
# -*- coding: utf-8 -*-
"""
Build "paper-ready" scenario tables from the aggregated binomial Monte Carlo CSV.

Adds:
- SD of bounds and length: sd = sqrt(var)
- MC standard errors (SE) of reported means: se = sd / sqrt(S)

Assumptions:
- The input CSV contains columns like:
    p, n, conf_level,
    lb_mean_<suffix>, ub_mean_<suffix>, len_mean_<suffix>,
    lb_var_<suffix>,  ub_var_<suffix>,  len_var_<suffix>,
    coverage_<suffix>
- Optionally, the CSV may contain a column with the number of MC replications
  (e.g., "n_sims", "S", "mc_reps"). If not, we fall back to MC_REPS_DEFAULT.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# Mapping entre "méthodes papier" et suffixes/colonnes dans ton CSV binomial
METHODS = {
    "Jeffreys": {
        "suffix": "jeff",
        "coverage": "coverage_jeff",
    },
    "Clopper–Pearson": {
        "suffix": "exact",
        "coverage": "coverage_exact",
    },
    "Normal Approximation": {
        "suffix": "approx",
        "coverage": "coverage_approx",
    },
    # Si tu veux aussi les one-sided / ECB, décommente :
    # "Jeffreys ECB (upper)": {"suffix": "ecb", "coverage": "coverage_ecb"},
    # "Exact CP upper": {"suffix": "exact_unil", "coverage": "coverage_exact_unil"},
    # "Normal upper": {"suffix": "approx_unil", "coverage": "coverage_approx_unil"},
}

# Scénarios = couples (n, p) + un nom
SCENARIOS = [
    {"scenario": "Baseline",     "n": 1000,  "p": 0.01},
    {"scenario": "Low Default",  "n": 1000,  "p": 0.001},
    {"scenario": "Small Sample", "n": 100,   "p": 0.01},
    {"scenario": "Large Sample", "n": 10000, "p": 0.01},
]

# Format d'affichage pour la table "papier"
DECIMALS = 6
PICK_NEAREST_P = True  # utile si p n'est pas exactement dans la grille

# Nombre de réplications MC (fallback si absent du CSV)
MC_REPS_DEFAULT = 10_000

# Noms possibles de colonnes "nb de sims" dans le CSV
MC_REPS_COL_CANDIDATES = ("n_sims", "S", "mc_reps", "mc_sims", "n_replications")


def _get_mc_reps_from_row(row: pd.Series) -> int:
    """
    Return the number of Monte Carlo replications S.
    Looks for common column names; falls back to MC_REPS_DEFAULT.
    """
    for c in MC_REPS_COL_CANDIDATES:
        if c in row.index:
            try:
                val = int(row[c])
                if val > 0:
                    return val
            except Exception:
                pass
    return MC_REPS_DEFAULT


def _pick_row_for_scenario(df: pd.DataFrame, n: int, p: float) -> pd.Series:
    sub = df[df["n"] == n].copy()
    if sub.empty:
        raise ValueError(f"No rows for n={n} in CSV")

    if not PICK_NEAREST_P:
        # tentative match exact (risque float)
        hit = sub[np.isclose(sub["p"].values, p)]
        if hit.empty:
            raise ValueError(f"p={p} not found for n={n} (try PICK_NEAREST_P=True)")
        return hit.iloc[0]

    # prend le p le plus proche dans la grille
    idx = (sub["p"] - p).abs().idxmin()
    return df.loc[idx]


def _fmt_interval(len_mean: float, lb_mean: float, ub_mean: float) -> str:
    return (
        f"({len_mean:.{DECIMALS}f}) "
        f"[{lb_mean:.{DECIMALS}f}, {ub_mean:.{DECIMALS}f}]"
    )


def _safe_sqrt_var(v: float) -> float:
    """
    sqrt(max(v,0)) to avoid NaNs from tiny negative numerical noise.
    """
    return float(np.sqrt(max(float(v), 0.0)))


def build_tables(df: pd.DataFrame, scenarios: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne:
      - df_long: une ligne par scenario × méthode (toutes les stats en colonnes)
      - df_wide: format "papier" (1 ligne par scenario, colonnes par méthode)

    df_long includes:
      - means (lb_mean, ub_mean, len_mean)
      - variances (lb_var, ub_var, len_var)
      - SD (lb_sd, ub_sd, len_sd)
      - MC SE of means (lb_se, ub_se, len_se)
      - coverage
      - S (MC replications)
    """
    # Basic sanity checks
    required_cols = {"n", "p", "conf_level"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    long_rows: list[dict] = []

    for sc in scenarios:
        row = _pick_row_for_scenario(df, sc["n"], sc["p"])

        # on garde le p réellement utilisé (peut différer si on a pris le plus proche)
        p_used = float(row["p"])
        conf = float(row["conf_level"])
        S = _get_mc_reps_from_row(row)

        for method_name, meta in METHODS.items():
            suf = meta["suffix"]

            # Required per-method columns
            cols_needed = [
                f"lb_mean_{suf}", f"ub_mean_{suf}", f"len_mean_{suf}",
                f"lb_var_{suf}",  f"ub_var_{suf}",  f"len_var_{suf}",
                meta["coverage"],
            ]
            missing_m = [c for c in cols_needed if c not in row.index]
            if missing_m:
                raise ValueError(
                    f"Missing columns for method '{method_name}' (suffix='{suf}'): {missing_m}"
                )

            lb_mean = float(row[f"lb_mean_{suf}"])
            ub_mean = float(row[f"ub_mean_{suf}"])
            lb_var  = float(row[f"lb_var_{suf}"])
            ub_var  = float(row[f"ub_var_{suf}"])
            len_mean = float(row[f"len_mean_{suf}"])
            len_var  = float(row[f"len_var_{suf}"])
            cov = float(row[meta["coverage"]])

            # SDs
            lb_sd  = _safe_sqrt_var(lb_var)
            ub_sd  = _safe_sqrt_var(ub_var)
            len_sd = _safe_sqrt_var(len_var)

            # MC standard errors of the reported means
            denom = float(np.sqrt(S))
            lb_se  = lb_sd / denom
            ub_se  = ub_sd / denom
            len_se = len_sd / denom

            long_rows.append(
                {
                    "scenario": sc["scenario"],
                    "n": int(sc["n"]),
                    "p_target": float(sc["p"]),
                    "p_used": p_used,
                    "conf_level": conf,
                    "S": int(S),
                    "method": method_name,

                    "coverage": cov,

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

    # Table "papier" (wide)
    base_cols = ["scenario", "n", "p_target", "p_used", "conf_level", "S"]
    wide = df_long[base_cols].drop_duplicates().sort_values(["scenario", "n"]).copy()

    for method_name in METHODS.keys():
        subm = df_long[df_long["method"] == method_name].copy()

        # colonne intervalle comme l'image
        subm["interval"] = subm.apply(
            lambda r: _fmt_interval(r["len_mean"], r["lb_mean"], r["ub_mean"]), axis=1
        )

        # merge sur les colonnes scenario
        cols_to_merge = base_cols + [
            "interval",
            "coverage",
            "lb_var", "ub_var", "len_var",
            "lb_sd", "ub_sd", "len_sd",
            "lb_se", "ub_se", "len_se",
        ]
        subm = subm[cols_to_merge].rename(
            columns={
                "interval": f"{method_name}",
                "coverage": f"{method_name}__coverage",

                "lb_var": f"{method_name}__lb_var",
                "ub_var": f"{method_name}__ub_var",
                "len_var": f"{method_name}__len_var",

                "lb_sd": f"{method_name}__lb_sd",
                "ub_sd": f"{method_name}__ub_sd",
                "len_sd": f"{method_name}__len_sd",

                "lb_se": f"{method_name}__lb_se",
                "ub_se": f"{method_name}__ub_se",
                "len_se": f"{method_name}__len_se",
            }
        )
        wide = wide.merge(subm, on=base_cols, how="left")

    return df_long, wide


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    out_dir = base_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "binom_coverage_all_n.csv"
    df = pd.read_csv(csv_path)

    df_long, df_wide = build_tables(df, SCENARIOS)

    out_long = out_dir / "binom_scenarios_long.csv"
    out_wide = out_dir / "binom_scenarios_table.csv"

    df_long.to_csv(out_long, index=False)
    df_wide.to_csv(out_wide, index=False)

    print(f"Saved: {out_long}")
    print(f"Saved: {out_wide}")
