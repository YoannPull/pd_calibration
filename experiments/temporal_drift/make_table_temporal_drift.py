from pathlib import Path
import numpy as np
import pandas as pd

METHOD_LABELS = {
    "jeffreys": "Jeffreys",
    "cp": "Clopper–Pearson",
    "normal": "Normal Approximation",
}

# Si None: auto-génère 3 scénarios (T0, T0+1, T) pour chaque p_hat trouvé
SCENARIOS = None

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


def default_scenarios_from_df(df: pd.DataFrame) -> list[dict]:
    scenarios = []
    for p_hat in sorted(df["p_hat"].unique()):
        sub = df[df["p_hat"] == p_hat]
        T0 = int(sub["T0"].iloc[0])
        T = int(sub["T"].iloc[0])

        scenarios.extend(
            [
                {"scenario": "Pre-drift",   "p_hat": float(p_hat), "t": T0},
                {"scenario": "Drift onset", "p_hat": float(p_hat), "t": T0 + 1},
                {"scenario": "End",         "p_hat": float(p_hat), "t": T},
            ]
        )
    return scenarios


def build_tables(df: pd.DataFrame, scenarios: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    len_col = _get_col(df, "len_mean", "avg_length")

    required_cols = ["lb_mean", "ub_mean", "lb_var", "ub_var", "len_var"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing columns in temporal_drift_results.csv: "
            + ", ".join(missing)
            + ". Did you update sim_temporal_drift.py to store bound means/vars?"
        )

    long_rows = []

    for sc in scenarios:
        sub = df[
            (np.isclose(df["p_hat"], sc["p_hat"]))
            & (df["t"] == sc["t"])
        ].copy()

        if sub.empty:
            raise ValueError(f"No rows found for scenario={sc}")

        # paramètres globaux (identiques pour toutes méthodes à ce t)
        T0 = int(sub["T0"].iloc[0])
        T = int(sub["T"].iloc[0])
        n = int(sub["n"].iloc[0])
        delta = float(sub["delta"].iloc[0])
        conf = float(sub["conf_level"].iloc[0])
        alpha_nom = float(sub["alpha_nominal"].iloc[0])
        p_true = float(sub["p_true"].iloc[0])
        phase = str(sub["phase"].iloc[0])

        for method in sorted(sub["method"].unique()):
            if method not in METHOD_LABELS:
                continue

            row = sub[sub["method"] == method].iloc[0]

            long_rows.append(
                {
                    "scenario": sc["scenario"],
                    "t": int(sc["t"]),
                    "phase": phase,
                    "p_hat": float(sc["p_hat"]),
                    "delta": float(delta),
                    "p_true": float(p_true),
                    "T": int(T),
                    "T0": int(T0),
                    "n": int(n),
                    "conf_level": float(conf),
                    "alpha_nominal": float(alpha_nom),
                    "n_sim": int(row["n_sim"]),
                    "method": METHOD_LABELS[method],
                    "coverage": float(row["coverage"]),
                    "reject_rate": float(row["reject_rate"]),
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
    base_cols = ["scenario", "p_hat", "t", "phase", "p_true", "delta", "n", "T", "T0", "conf_level", "alpha_nominal", "n_sim"]
    wide = df_long[base_cols].drop_duplicates().sort_values(["p_hat", "t", "scenario"]).copy()

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

    df_path = data_dir / "temporal_drift_results.csv"
    df = pd.read_csv(df_path)

    scenarios = default_scenarios_from_df(df) if SCENARIOS is None else SCENARIOS

    df_long, df_wide = build_tables(df, scenarios)

    out_long = out_dir / "temporal_drift_scenarios_long.csv"
    out_wide = out_dir / "temporal_drift_scenarios_table.csv"

    df_long.to_csv(out_long, index=False)
    df_wide.to_csv(out_wide, index=False)

    print(f"Saved: {out_long}")
    print(f"Saved: {out_wide}")
