# -*- coding: utf-8 -*-
"""
Prior sensitivity for Binomial calibration:
- two_sided: coverage of equal-tailed Bayesian credible intervals under different Beta priors
- upper (one-sided): study "conservatism" via E[ F_post(p_true) ] (posterior CDF at the true p)

Outputs LONG dataframe (one row per n × p × prior × tail) with:
- two_sided: coverage + bound stats
- upper: conservatism_mean/var + bound stats (coverage is NOT the focus)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import beta as beta_dist


# -------------------------
# Priors to compare
# -------------------------
def get_priors(eps: float = 1e-6) -> dict[str, tuple[float, float]]:
    """
    Returns a dict prior_name -> (alpha, beta)

    - Jeffreys: Beta(1/2, 1/2)
    - Laplace:  Beta(1, 1)
    - Haldane:  Beta(0, 0) improper -> approximated by (eps, eps)
    """
    return {
        "Jeffreys (1/2,1/2)": (0.5, 0.5),
        "Laplace (1,1)": (1.0, 1.0),
        f"Haldane (~{eps:g},{eps:g})": (eps, eps),
    }


# -------------------------
# Helpers
# -------------------------
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
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    return float(((lb <= p_true) & (p_true <= ub)).mean())


def beta_equal_tailed_interval(
    *,
    n: int,
    d: np.ndarray,
    conf: float,
    alpha_prior: float,
    beta_prior: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Equal-tailed Bayesian credible interval.
    Posterior: Beta(d + a0, n-d + b0)
    """
    a_post = d.astype(float) + float(alpha_prior)
    b_post = (n - d).astype(float) + float(beta_prior)
    q = (1.0 - conf) / 2.0
    lb = beta_dist.ppf(q, a_post, b_post)
    ub = beta_dist.ppf(1.0 - q, a_post, b_post)
    return lb, ub


def beta_upper_bound(
    *,
    n: int,
    d: np.ndarray,
    conf: float,
    alpha_prior: float,
    beta_prior: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Upper one-sided bound:
      LB = 0
      UB = quantile(conf) of posterior
    """
    a_post = d.astype(float) + float(alpha_prior)
    b_post = (n - d).astype(float) + float(beta_prior)
    ub = beta_dist.ppf(conf, a_post, b_post)
    lb = np.zeros_like(ub, dtype=float)
    return lb, ub


def posterior_cdf_at_ptrue(
    *,
    n: int,
    d: np.ndarray,
    p_true: float,
    alpha_prior: float,
    beta_prior: float,
) -> np.ndarray:
    """
    Returns F_post(p_true) for each replication, where
    post = Beta(d + a0, n-d + b0).
    """
    a_post = d.astype(float) + float(alpha_prior)
    b_post = (n - d).astype(float) + float(beta_prior)
    return beta_dist.cdf(float(p_true), a_post, b_post)


def simulate_prior_sensibility_for_n(
    *,
    n: int,
    n_simulation: int,
    p_values: np.ndarray,
    confidence_level: float,
    priors: dict[str, tuple[float, float]],
    rng: np.random.Generator,
    tails: tuple[str, ...] = ("two_sided", "upper"),
) -> pd.DataFrame:
    """
    LONG output:
      n, p, conf_level, n_sims, prior, prior_a, prior_b, tail,
      coverage (only meaningful for two_sided),
      conservatism_mean/var (only for upper),
      lb_mean, ub_mean, lb_var, ub_var, len_mean, len_var
    """
    rows: list[dict] = []

    for p in tqdm(p_values, desc=f"Prior sensitivity (n={n})"):
        d = rng.binomial(n, p, size=n_simulation)

        for prior_name, (a0, b0) in priors.items():
            for tail in tails:
                if tail == "two_sided":
                    lb, ub = beta_equal_tailed_interval(
                        n=n, d=d, conf=confidence_level, alpha_prior=a0, beta_prior=b0
                    )
                    cov = _coverage(lb, ub, p)
                    cons_mean, cons_var = np.nan, np.nan

                elif tail == "upper":
                    lb, ub = beta_upper_bound(
                        n=n, d=d, conf=confidence_level, alpha_prior=a0, beta_prior=b0
                    )
                    # Conservatism metric: E[ F_post(p_true) ]
                    cons = posterior_cdf_at_ptrue(
                        n=n, d=d, p_true=float(p), alpha_prior=a0, beta_prior=b0
                    )
                    cons_mean = float(np.mean(cons))
                    cons_var = float(np.var(cons, ddof=0))
                    cov = np.nan  # not the focus here

                else:
                    raise ValueError(f"Unknown tail: {tail}")

                s = _summarize_bounds(lb, ub)

                rows.append(
                    {
                        "n": int(n),
                        "p": float(p),
                        "conf_level": float(confidence_level),
                        "n_sims": int(n_simulation),
                        "prior": str(prior_name),
                        "prior_a": float(a0),
                        "prior_b": float(b0),
                        "tail": str(tail),

                        "coverage": float(cov) if np.isfinite(cov) else np.nan,
                        "conservatism_mean": float(cons_mean) if np.isfinite(cons_mean) else np.nan,
                        "conservatism_var": float(cons_var) if np.isfinite(cons_var) else np.nan,
                        **s,
                    }
                )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n_simulation = 10_000
    ns = [50, 100, 1000, 10_000]
    confidence_level = 0.95

    # main grid
    p_values = np.linspace(0.001, 0.1, 300)

    # rare-event grid
    n_ldp = 1000
    p_values_ldp = np.linspace(0.0001, 0.005, 300)

    priors = get_priors(eps=1e-6)

    rng = np.random.default_rng(123)
    dfs = []
    for n in ns:
        dfs.append(
            simulate_prior_sensibility_for_n(
                n=n,
                n_simulation=n_simulation,
                p_values=p_values,
                confidence_level=confidence_level,
                priors=priors,
                rng=rng,
                tails=("two_sided", "upper"),
            )
        )

    df_all = pd.concat(dfs, ignore_index=True)
    out_path = data_dir / "prior_sensibility_all_n.csv"
    df_all.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    rng_ldp = np.random.default_rng(456)
    df_ldp = simulate_prior_sensibility_for_n(
        n=n_ldp,
        n_simulation=n_simulation,
        p_values=p_values_ldp,
        confidence_level=confidence_level,
        priors=priors,
        rng=rng_ldp,
        tails=("two_sided", "upper"),
    )
    out_path_ldp = data_dir / f"prior_sensibility_ldp_n{n_ldp}.csv"
    df_ldp.to_csv(out_path_ldp, index=False)
    print(f"Saved: {out_path_ldp}")
