# experiments/beta_binom_jeffreys/sim_beta_binom.py
# -*- coding: utf-8 -*-
"""
Beta–Binomial robustness simulations (Binomial-based intervals under over-dispersion).

This script supports TWO designs:

(A) CURVES design (legacy):
    - grid over p_true (many values) for each n and rho
    - used to plot coverage vs p (coverage curves)

(B) SCENARIOS design (paper compact):
    - a small set of scenarios with fixed (n, p_true) and varying rho
    - used to build compact tables and robustness-vs-rho plots

Outputs are written to:
  data/beta_binom_results_curves.csv
  data/beta_binom_results_scenarios.csv

You can also set WRITE_COMBINED=True to save a combined CSV.

Run:
  poetry run python experiments/beta_binom_jeffreys/sim_beta_binom.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from experiments.stats.intervals import jeffreys_alpha2, approx_normal, exact_cp

# -----------------------------
# Global config
# -----------------------------
SEED: int = 123
N_SIM: int = 10_000
CONF_LEVEL: float = 0.95

RHO_GRID: list[float] = [0.00, 0.01, 0.05, 0.10]

# (A) CURVES design: for coverage-vs-p plots
NS_CURVES: list[int] = [100, 1000, 10_000]
P_TRUES_CURVES: np.ndarray = np.linspace(0.001, 0.1, 30)

# (B) SCENARIOS design: for compact tables/robustness-vs-rho plots
SCENARIOS: list[dict] = [
    {"scenario": "Baseline",    "n": 1000, "p_true": 0.01,  "rhos": RHO_GRID},
    {"scenario": "Low Default", "n": 1000, "p_true": 0.001, "rhos": RHO_GRID},
]

WRITE_COMBINED: bool = True


# -----------------------------
# Model helpers
# -----------------------------
def beta_binom_params(p: float, rho: float) -> tuple[float, float]:
    """Beta(alpha,beta) parameters for mean p and ICC rho (rho in (0,1))."""
    if rho <= 0.0 or rho >= 1.0:
        raise ValueError("rho must be in (0, 1).")
    total = (1.0 - rho) / rho
    alpha = p * total
    beta_param = (1.0 - p) * total
    return alpha, beta_param


def simulate_beta_binomial(
    n: int,
    p: float,
    rho: float,
    n_sim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw default counts under Binomial if rho=0 else Beta–Binomial."""
    if rho == 0.0:
        return rng.binomial(n, p, size=n_sim)
    alpha, beta_param = beta_binom_params(p, rho)
    theta = rng.beta(alpha, beta_param, size=n_sim)
    return rng.binomial(n, theta)


def _summarize_bounds(lb: np.ndarray, ub: np.ndarray) -> dict[str, float]:
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    length = ub - lb
    return {
        "lb_mean": float(np.mean(lb)),
        "ub_mean": float(np.mean(ub)),
        "lb_var": float(np.var(lb, ddof=0)),
        "ub_var": float(np.var(ub, ddof=0)),
        "len_mean": float(np.mean(length)),
        "len_var": float(np.var(length, ddof=0)),
    }


def _coverage(lb: np.ndarray, ub: np.ndarray, p_true: float) -> float:
    return float(((lb <= p_true) & (p_true <= ub)).mean())


def simulate_comparison_beta_binom(
    *,
    n: int,
    p_true: float,
    rho: float,
    n_sim: int,
    confidence_level: float,
    rng: np.random.Generator,
    p_star: float | None = None,
) -> dict[str, dict[str, float]]:
    """
    Generate D under Beta–Binomial, then compute Binomial-based intervals as-if i.i.d. Binomial.
    """
    if p_star is None:
        p_star = p_true

    d_samples = simulate_beta_binomial(n=n, p=p_true, rho=rho, n_sim=n_sim, rng=rng)

    lb_j = np.empty(n_sim, dtype=float)
    ub_j = np.empty(n_sim, dtype=float)
    lb_cp = np.empty(n_sim, dtype=float)
    ub_cp = np.empty(n_sim, dtype=float)
    lb_n = np.empty(n_sim, dtype=float)
    ub_n = np.empty(n_sim, dtype=float)

    for i, d in enumerate(d_samples):
        a, b = jeffreys_alpha2(n, int(d), confidence_level)
        lb_j[i], ub_j[i] = a, b

        a, b = exact_cp(n, int(d), confidence_level)
        lb_cp[i], ub_cp[i] = a, b

        a, b = approx_normal(n, int(d), confidence_level)
        lb_n[i] = max(0.0, a)
        ub_n[i] = min(1.0, b)

    results: dict[str, dict[str, float]] = {
        "jeffreys": {
            "coverage": _coverage(lb_j, ub_j, p_true),
            "avg_length": float(np.mean(ub_j - lb_j)),
            "reject_star_rate": float(((p_star < lb_j) | (p_star > ub_j)).mean()),
            **_summarize_bounds(lb_j, ub_j),
        },
        "cp": {
            "coverage": _coverage(lb_cp, ub_cp, p_true),
            "avg_length": float(np.mean(ub_cp - lb_cp)),
            "reject_star_rate": float(((p_star < lb_cp) | (p_star > ub_cp)).mean()),
            **_summarize_bounds(lb_cp, ub_cp),
        },
        "normal": {
            "coverage": _coverage(lb_n, ub_n, p_true),
            "avg_length": float(np.mean(ub_n - lb_n)),
            "reject_star_rate": float(((p_star < lb_n) | (p_star > ub_n)).mean()),
            **_summarize_bounds(lb_n, ub_n),
        },
    }
    return results


# -----------------------------
# Runners
# -----------------------------
def run_curves(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict] = []
    for n in NS_CURVES:
        for p_true in P_TRUES_CURVES:
            for rho in RHO_GRID:
                print(f"[CURVES] n={n} p_true={p_true:.6f} rho={rho:.3f}")
                res = simulate_comparison_beta_binom(
                    n=n,
                    p_true=float(p_true),
                    rho=float(rho),
                    n_sim=N_SIM,
                    p_star=float(p_true),
                    confidence_level=CONF_LEVEL,
                    rng=rng,
                )
                for method, stats in res.items():
                    rows.append(
                        {
                            "design": "curves",
                            "scenario": "Grid",
                            "n": int(n),
                            "p_true": float(p_true),
                            "rho": float(rho),
                            "method": method,
                            "coverage": float(stats["coverage"]),
                            "avg_length": float(stats["avg_length"]),
                            "reject_star_rate": float(stats["reject_star_rate"]),
                            "lb_mean": float(stats["lb_mean"]),
                            "ub_mean": float(stats["ub_mean"]),
                            "lb_var": float(stats["lb_var"]),
                            "ub_var": float(stats["ub_var"]),
                            "len_mean": float(stats["len_mean"]),
                            "len_var": float(stats["len_var"]),
                            "conf_level": float(CONF_LEVEL),
                            "n_sim": int(N_SIM),
                        }
                    )
    return pd.DataFrame(rows)


def run_scenarios(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict] = []
    for sc in SCENARIOS:
        scenario_name = str(sc["scenario"])
        n = int(sc["n"])
        p_true = float(sc["p_true"])
        for rho in [float(x) for x in sc["rhos"]]:
            print(f"[SCENARIOS] scenario={scenario_name} n={n} p_true={p_true:.6f} rho={rho:.3f}")
            res = simulate_comparison_beta_binom(
                n=n,
                p_true=p_true,
                rho=rho,
                n_sim=N_SIM,
                p_star=p_true,
                confidence_level=CONF_LEVEL,
                rng=rng,
            )
            for method, stats in res.items():
                rows.append(
                    {
                        "design": "scenarios",
                        "scenario": scenario_name,
                        "n": n,
                        "p_true": p_true,
                        "rho": float(rho),
                        "method": method,
                        "coverage": float(stats["coverage"]),
                        "avg_length": float(stats["avg_length"]),
                        "reject_star_rate": float(stats["reject_star_rate"]),
                        "lb_mean": float(stats["lb_mean"]),
                        "ub_mean": float(stats["ub_mean"]),
                        "lb_var": float(stats["lb_var"]),
                        "ub_var": float(stats["ub_var"]),
                        "len_mean": float(stats["len_mean"]),
                        "len_var": float(stats["len_var"]),
                        "conf_level": float(CONF_LEVEL),
                        "n_sim": int(N_SIM),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    df_curves = run_curves(rng)
    out_curves = data_dir / "beta_binom_results_curves.csv"
    df_curves.to_csv(out_curves, index=False)
    print(f"[OK] Saved: {out_curves}")

    df_scen = run_scenarios(rng)
    out_scen = data_dir / "beta_binom_results_scenarios.csv"
    df_scen.to_csv(out_scen, index=False)
    print(f"[OK] Saved: {out_scen}")

    if WRITE_COMBINED:
        df_all = pd.concat([df_curves, df_scen], axis=0, ignore_index=True)
        out_all = data_dir / "beta_binom_results.csv"
        df_all.to_csv(out_all, index=False)
        print(f"[OK] Saved combined: {out_all}")


if __name__ == "__main__":
    main()
