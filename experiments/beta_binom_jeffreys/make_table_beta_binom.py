from pathlib import Path
import numpy as np
import pandas as pd

# Méthodes telles qu'elles apparaissent dans beta_binom_results.csv
METHOD_LABELS = {
    "jeffreys": "Jeffreys",
    "cp": "Clopper–Pearson",
    "normal": "Normal Approximation",
}

# Scénarios = choix (n, np_target, rho)
# Ajuste librement selon tes besoins.
SCENARIOS = [
    {"scenario": "Baseline",     "n": 100, "np_target": 1.0,  "rho": 0.00},
    {"scenario": "Low Default",  "n": 100, "np_target": 0.4,  "rho": 0.00},
    {"scenario": "Small Sample", "n": 50,  "np_target": 1.0,  "rho": 0.00},
    {"scenario": "Clustered",    "n": 100, "np_target": 1.0,  "rho": 0.10},
]

DECIMALS = 3


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


def build_tables(df: pd.DataFrame, scenarios: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Colonnes length : compat si tu as gardé avg_length
    len_col = _get_col(df, "len_mean", "avg_length")

    # Si tu n'as pas encore ajouté les stats de bornes/variances, ça plantera ici
    required_cols = ["lb_mean", "ub_mean", "lb_var", "ub_var", "len_var"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing columns in beta_binom_results.csv: "
            + ", ".join(missing)
            + ". Did you update sim_beta_binom.py to store bound means/vars?"
        )

    long_rows = []

    for sc in scenarios:
        # filtre scénario
        sub = df[
            (df["n"] == sc["n"])
            & (np.isclose(df["np_target"], sc["np_target"]))
            & (np.isclose(df["rho"], sc["rho"]))
        ].copy()

        if sub.empty:
            raise ValueError(f"No rows found for scenario={sc}")

        # on s'assure d'avoir les 3 méthodes
        for method in sorted(sub["method"].unique()):
            if method not in METHOD_LABELS:
                continue

            row = sub[sub["method"] == method].iloc[0]

            long_rows.append(
                {
                    "scenario": sc["scenario"],
                    "n": int(row["n"]),
                    "np_target": float(row["np_target"]),
                    "p_true": float(row["p_true"]),
                    "rho": float(row["rho"]),
                    "conf_level": float(row["conf_level"]),
                    "n_sim": int(row["n_sim"]),
                    "method": METHOD_LABELS[method],
                    "coverage": float(row["coverage"]),
                    "reject_rate": float(row["reject_star_rate"]),
                    "lb_mean": float(row["lb_mean"]),
                    "ub_mean": float(row["ub_mean"]),
                    "lb_var": float(row["lb_var"]),
                    "ub_var": float(row["ub_var"]),
                    "len_mean": float(row[len_col]),
                    "len_var": float(row["len_var"]),
                }
            )

    df_long = pd.DataFrame(long_rows)

    # Table "papier"
    base_cols = ["scenario", "n", "np_target", "p_true", "rho", "conf_level", "n_sim"]
    wide = df_long[base_cols].drop_duplicates().sort_values(["scenario", "n", "rho"]).copy()

    for method_label in sorted(df_long["method"].unique()):
        subm = df_long[df_long["method"] == method_label].copy()
        subm["interval"] = subm.apply(
            lambda r: _fmt_interval(r["len_mean"], r["lb_mean"], r["ub_mean"]), axis=1
        )

        subm = subm[base_cols + ["interval", "coverage", "reject_rate", "lb_var", "ub_var", "len_var"]].rename(
            columns={
                "interval": f"{method_label}",
                "coverage": f"{method_label}__coverage",
                "reject_rate": f"{method_label}__reject_rate",
                "lb_var": f"{method_label}__lb_var",
                "ub_var": f"{method_label}__ub_var",
                "len_var": f"{method_label}__len_var",
            }
        )

        wide = wide.merge(subm, on=base_cols, how="left")

    return df_long, wide


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    out_dir = base_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_path = data_dir / "beta_binom_results.csv"
    df = pd.read_csv(df_path)

    df_long, df_wide = build_tables(df, SCENARIOS)

    out_long = out_dir / "beta_binom_scenarios_long.csv"
    out_wide = out_dir / "beta_binom_scenarios_table.csv"

    df_long.to_csv(out_long, index=False)
    df_wide.to_csv(out_wide, index=False)

    print(f"Saved: {out_long}")
    print(f"Saved: {out_wide}")
