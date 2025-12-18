# experiments/binom_coverage/make_table_binom.py

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
# Ajuste librement : l'idée est juste de donner une petite liste de "cas"
SCENARIOS = [
    {"scenario": "Baseline",     "n": 1000, "p": 0.01},
    {"scenario": "Low Default",  "n": 1000, "p": 0.001},
    {"scenario": "Small Sample", "n": 100,  "p": 0.01},
    {"scenario": "Large Sample", "n": 10000, "p": 0.01},
]

# Format d'affichage pour la table "papier"
DECIMALS = 6  
PICK_NEAREST_P = True  # utile si p n'est pas exactement dans la grille


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


def build_tables(df: pd.DataFrame, scenarios: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne:
      - df_long: une ligne par scenario × méthode (toutes les stats en colonnes)
      - df_wide: format "papier" (1 ligne par scenario, colonnes par méthode)
    """
    long_rows = []

    for sc in scenarios:
        row = _pick_row_for_scenario(df, sc["n"], sc["p"])

        # on garde le p réellement utilisé (peut différer si on a pris le plus proche)
        p_used = float(row["p"])
        conf = float(row["conf_level"])

        for method_name, meta in METHODS.items():
            suf = meta["suffix"]

            lb_mean = float(row[f"lb_mean_{suf}"])
            ub_mean = float(row[f"ub_mean_{suf}"])
            lb_var  = float(row[f"lb_var_{suf}"])
            ub_var  = float(row[f"ub_var_{suf}"])
            len_mean = float(row[f"len_mean_{suf}"])
            len_var  = float(row[f"len_var_{suf}"])
            cov = float(row[meta["coverage"]])

            long_rows.append(
                {
                    "scenario": sc["scenario"],
                    "n": int(sc["n"]),
                    "p_target": float(sc["p"]),
                    "p_used": p_used,
                    "conf_level": conf,
                    "method": method_name,
                    "coverage": cov,
                    "lb_mean": lb_mean,
                    "ub_mean": ub_mean,
                    "lb_var": lb_var,
                    "ub_var": ub_var,
                    "len_mean": len_mean,
                    "len_var": len_var,
                }
            )

    df_long = pd.DataFrame(long_rows)

    # Table "papier" (wide)
    base_cols = ["scenario", "n", "p_target", "p_used", "conf_level"]
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
            "lb_var",
            "ub_var",
            "len_var",
        ]
        subm = subm[cols_to_merge].rename(
            columns={
                "interval": f"{method_name}",
                "coverage": f"{method_name}__coverage",
                "lb_var": f"{method_name}__lb_var",
                "ub_var": f"{method_name}__ub_var",
                "len_var": f"{method_name}__len_var",
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
