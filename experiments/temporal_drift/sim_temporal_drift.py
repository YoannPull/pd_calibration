# experiments/temporal_drift/sim_temporal_drift.py
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path

from experiments.stats.intervals import (
    jeffreys_alpha2,
    exact_cp,
    approx_normal,
    in_interval,
)

# =========================
# Scenarios to run
# =========================
SCENARIOS = [
    {"scenario": "Baseline",     "n": 1000,  "p": 0.01},
    {"scenario": "Low Default",  "n": 1000,  "p": 0.001},
    {"scenario": "Small Sample", "n": 100,   "p": 0.01},
    {"scenario": "Large Sample", "n": 10000, "p": 0.01},
]


def slugify(name: str) -> str:
    """
    Safe filename slug: 'Low Default' -> 'low_default'
    """
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def fmt_float_for_path(x: float) -> str:
    """
    Safe float formatter for filenames: 0.001 -> '0p001'
    """
    s = f"{x:.6g}"
    return s.replace(".", "p")


def temporal_pd_path(T: int, T0: int, p_hat: float, delta: float) -> np.ndarray:
    t = np.arange(1, T + 1)
    p_true = np.empty(T, dtype=float)

    p_true[t <= T0] = p_hat
    mask = t > T0
    if np.any(mask):
        p_true[mask] = p_hat + delta * (t[mask] - T0) / (T - T0)

    return p_true


def monte_carlo_temporal_drift(
    T: int = 60,
    T0: int = 24,
    n: int = 100,
    p_hat: float = 0.01,
    delta: float = 0.01,
    n_sim: int = 10_000,
    confidence_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng()

    # guardrails
    if not (0.0 < p_hat < 1.0):
        raise ValueError("p_hat must be in (0,1)")
    if not (0.0 <= p_hat + delta <= 1.0):
        raise ValueError("p_hat + delta must be in [0,1]")
    if not (1 <= T0 < T):
        raise ValueError("Need 1 <= T0 < T")

    p_true_t = temporal_pd_path(T=T, T0=T0, p_hat=p_hat, delta=delta)

    methods = ["jeffreys", "cp", "normal"]
    m_index = {name: i for i, name in enumerate(methods)}
    M = len(methods)

    coverage_counts = np.zeros((M, T), dtype=float)
    reject_counts = np.zeros((M, T), dtype=float)

    lb_sum = np.zeros((M, T), dtype=float)
    lb_sumsq = np.zeros((M, T), dtype=float)
    ub_sum = np.zeros((M, T), dtype=float)
    ub_sumsq = np.zeros((M, T), dtype=float)

    len_sum = np.zeros((M, T), dtype=float)
    len_sumsq = np.zeros((M, T), dtype=float)

    alpha = 1.0 - confidence_level

    for _ in range(n_sim):
        d_t = rng.binomial(n, p_true_t)

        for t_idx in range(T):
            p_true = float(p_true_t[t_idx])
            d = int(d_t[t_idx])

            # Jeffreys
            m = m_index["jeffreys"]
            lb, ub = jeffreys_alpha2(n, d, confidence_level)
            length = ub - lb
            coverage_counts[m, t_idx] += in_interval(lb, ub, p_true)
            reject_counts[m, t_idx] += int((p_hat < lb) or (p_hat > ub))
            lb_sum[m, t_idx] += lb
            lb_sumsq[m, t_idx] += lb * lb
            ub_sum[m, t_idx] += ub
            ub_sumsq[m, t_idx] += ub * ub
            len_sum[m, t_idx] += length
            len_sumsq[m, t_idx] += length * length

            # Exact CP
            m = m_index["cp"]
            lb, ub = exact_cp(n, d, confidence_level)
            length = ub - lb
            coverage_counts[m, t_idx] += in_interval(lb, ub, p_true)
            reject_counts[m, t_idx] += int((p_hat < lb) or (p_hat > ub))
            lb_sum[m, t_idx] += lb
            lb_sumsq[m, t_idx] += lb * lb
            ub_sum[m, t_idx] += ub
            ub_sumsq[m, t_idx] += ub * ub
            len_sum[m, t_idx] += length
            len_sumsq[m, t_idx] += length * length

            # Normal approx (clipped)
            m = m_index["normal"]
            lb, ub = approx_normal(n, d, confidence_level)
            lb = max(0.0, lb)
            ub = min(1.0, ub)
            length = ub - lb
            coverage_counts[m, t_idx] += in_interval(lb, ub, p_true)
            reject_counts[m, t_idx] += int((p_hat < lb) or (p_hat > ub))
            lb_sum[m, t_idx] += lb
            lb_sumsq[m, t_idx] += lb * lb
            ub_sum[m, t_idx] += ub
            ub_sumsq[m, t_idx] += ub * ub
            len_sum[m, t_idx] += length
            len_sumsq[m, t_idx] += length * length

    coverage_rates = coverage_counts / n_sim
    reject_rates = reject_counts / n_sim

    lb_mean = lb_sum / n_sim
    ub_mean = ub_sum / n_sim
    len_mean = len_sum / n_sim

    lb_var = (lb_sumsq / n_sim) - lb_mean**2
    ub_var = (ub_sumsq / n_sim) - ub_mean**2
    len_var = (len_sumsq / n_sim) - len_mean**2

    rows = []
    times = np.arange(1, T + 1)
    phases = np.where(times <= T0, "pre_drift", "post_drift")

    for method in methods:
        m = m_index[method]
        for t_idx, t in enumerate(times):
            rows.append(
                {
                    "t": int(t),
                    "phase": phases[t_idx],
                    "method": method,
                    "p_true": float(p_true_t[t_idx]),
                    "coverage": float(coverage_rates[m, t_idx]),
                    "avg_length": float(len_mean[m, t_idx]),
                    "reject_rate": float(reject_rates[m, t_idx]),
                    "lb_mean": float(lb_mean[m, t_idx]),
                    "ub_mean": float(ub_mean[m, t_idx]),
                    "lb_var": float(lb_var[m, t_idx]),
                    "ub_var": float(ub_var[m, t_idx]),
                    "len_mean": float(len_mean[m, t_idx]),
                    "len_var": float(len_var[m, t_idx]),
                    "T": int(T),
                    "T0": int(T0),
                    "n": int(n),
                    "p_hat": float(p_hat),
                    "delta": float(delta),
                    "n_sim": int(n_sim),
                    "conf_level": float(confidence_level),
                    "alpha_nominal": float(alpha),
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Global parameters
    T = 68
    T0 = 24
    n_sim = 10_000
    conf = 0.95

    # Drift definition: delta = k_delta * p_hat (relative drift)
    k_delta = 1.0

    rng = np.random.default_rng(123)

    all_dfs = []

    for sc in SCENARIOS:
        scenario_name = sc["scenario"]
        scenario_slug = slugify(scenario_name)
        n = int(sc["n"])
        p_hat = float(sc["p"])

        delta = k_delta * p_hat
        delta = min(delta, 1.0 - p_hat)

        df_sc = monte_carlo_temporal_drift(
            T=T,
            T0=T0,
            n=n,
            p_hat=p_hat,
            delta=delta,
            n_sim=n_sim,
            confidence_level=conf,
            rng=rng,
        )

        # add scenario metadata
        df_sc["scenario"] = scenario_name
        df_sc["scenario_slug"] = scenario_slug
        df_sc["scenario_n"] = n
        df_sc["scenario_p"] = p_hat

        # Save ONE CSV per scenario (named by scenario)
        out_path = data_dir / (
            f"temporal_drift_{scenario_slug}"
            f"_n{n}_p{fmt_float_for_path(p_hat)}"
            f"_T{T}_T0{T0}_nsim{n_sim}.csv"
        )
        df_sc.to_csv(out_path, index=False)
        print(f"Saved scenario '{scenario_name}' -> {out_path}")

        all_dfs.append(df_sc)

    # Optional: also save a combined file (handy for global plots)
    df_all = pd.concat(all_dfs, ignore_index=True)
    out_all = data_dir / "temporal_drift_all_scenarios.csv"
    df_all.to_csv(out_all, index=False)
    print(f"\nSaved combined results -> {out_all}")

    # Console summary: mean reject rates pre/post drift per scenario
    summary = (
        df_all.groupby(["scenario", "method", "phase"])["reject_rate"]
        .mean()
        .reset_index()
        .pivot_table(index=["scenario", "method"], columns="phase", values="reject_rate")
        .sort_index()
    )
    print("\nMean rejection rate of H0: p = p_hat (pre/post drift):")
    print(summary)
