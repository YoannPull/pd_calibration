# experiments/beta_binom_jeffreys/sim_beta_binom.py
import numpy as np
import pandas as pd
from pathlib import Path

from experiments.stats.intervals import (
    jeffreys_alpha2,
    approx_normal,
    exact_cp,
)


def beta_binom_params(p, rho):
    """
    Paramètres (alpha, beta) d'une Beta pour obtenir une Beta-Binomiale
    de moyenne p et corrélation intra-classe rho.
    """
    if rho <= 0 or rho >= 1:
        raise ValueError("rho doit être dans (0, 1).")
    total = (1 - rho) / rho
    alpha = p * total
    beta_param = (1 - p) * total
    return alpha, beta_param


def simulate_beta_binomial(n, p, rho, n_sim, rng=None):
    """
    Simule d selon :
      - Binomiale(n, p) si rho = 0
      - Beta-Binomiale(n, alpha, beta) si 0 < rho < 1
    """
    if rng is None:
        rng = np.random.default_rng()

    if rho == 0:
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
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    return float(((lb <= p_true) & (p_true <= ub)).mean())


def simulate_comparison_beta_binom(
    n,
    p_true,
    rho,
    n_sim=10_000,
    p_star=None,
    confidence_level=0.95,
    rng=None,
):
    """
    Compare Jeffreys / Clopper-Pearson / Normale sous un DGP Beta-Binomial,
    en appliquant les procédures comme si le modèle était Binomial(n, p).

    Enregistre en plus:
      - moyenne/variance des bornes LB/UB
      - moyenne/variance de la longueur
    """
    if rng is None:
        rng = np.random.default_rng()
    if p_star is None:
        p_star = p_true

    d_samples = simulate_beta_binomial(n, p_true, rho, n_sim, rng=rng)

    # On calcule toutes les bornes en arrays (plus rapide, plus simple)
    lb_j = np.empty(n_sim, dtype=float)
    ub_j = np.empty(n_sim, dtype=float)
    lb_cp = np.empty(n_sim, dtype=float)
    ub_cp = np.empty(n_sim, dtype=float)
    lb_n = np.empty(n_sim, dtype=float)
    ub_n = np.empty(n_sim, dtype=float)

    for i, d in enumerate(d_samples):
        a, b = jeffreys_alpha2(n, d, confidence_level)
        lb_j[i], ub_j[i] = a, b

        a, b = exact_cp(n, d, confidence_level)
        lb_cp[i], ub_cp[i] = a, b

        a, b = approx_normal(n, d, confidence_level)
        lb_n[i] = max(0.0, a)
        ub_n[i] = min(1.0, b)

    results = {}

    # Jeffreys
    results["jeffreys"] = {
        "coverage": _coverage(lb_j, ub_j, p_true),
        "avg_length": float(np.mean(ub_j - lb_j)),
        "reject_star_rate": float(((p_star < lb_j) | (p_star > ub_j)).mean()),
        **_summarize_bounds(lb_j, ub_j),
    }

    # CP
    results["cp"] = {
        "coverage": _coverage(lb_cp, ub_cp, p_true),
        "avg_length": float(np.mean(ub_cp - lb_cp)),
        "reject_star_rate": float(((p_star < lb_cp) | (p_star > ub_cp)).mean()),
        **_summarize_bounds(lb_cp, ub_cp),
    }

    # Normal approx
    results["normal"] = {
        "coverage": _coverage(lb_n, ub_n, p_true),
        "avg_length": float(np.mean(ub_n - lb_n)),
        "reject_star_rate": float(((p_star < lb_n) | (p_star > ub_n)).mean()),
        **_summarize_bounds(lb_n, ub_n),
    }

    return results


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ns = [50, 100, 200, 1000, 10_000]
    np_targets = [
        0.001,
        0.01, 0.02, 0.03, 0.04, 0.05,
        0.07, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.60, 0.70, 0.80, 0.90, 1.00,
        1.25, 1.50, 1.75, 2.00, 2.50,
        3.00, 3.50, 4.00, 5.00, 6.00,
        7.50, 10.00,
    ]  # valeurs de n*p
    rhos = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.20]
    n_sim = 10_000
    conf = 0.95

    rows = []
    rng = np.random.default_rng(123)

    for n in ns:
        for np_target in np_targets:
            p_true = np_target / n
            for rho in rhos:
                print(f"Simu: n={n}, p_true={p_true}, n*p={np_target}, rho={rho}")
                res = simulate_comparison_beta_binom(
                    n=n,
                    p_true=p_true,
                    rho=rho,
                    n_sim=n_sim,
                    p_star=p_true,
                    confidence_level=conf,
                    rng=rng,
                )
                for method, stats in res.items():
                    rows.append(
                        {
                            "n": n,
                            "p_true": p_true,
                            "np_target": np_target,
                            "rho": rho,
                            "method": method,
                            "coverage": stats["coverage"],
                            "avg_length": stats["avg_length"],  # gardé pour compat
                            "reject_star_rate": stats["reject_star_rate"],
                            # nouvelles colonnes
                            "lb_mean": stats["lb_mean"],
                            "ub_mean": stats["ub_mean"],
                            "lb_var": stats["lb_var"],
                            "ub_var": stats["ub_var"],
                            "len_mean": stats["len_mean"],
                            "len_var": stats["len_var"],
                            "conf_level": conf,
                            "n_sim": n_sim,
                        }
                    )

    df = pd.DataFrame(rows)
    out_path = data_dir / "beta_binom_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Résultats sauvegardés dans {out_path}")
