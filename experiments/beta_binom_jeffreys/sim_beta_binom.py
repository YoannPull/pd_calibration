# experiments/beta_binom_jeffreys/sim_beta_binom.py
import numpy as np
import pandas as pd
from pathlib import Path

from intervals import (
    jeffreys_alpha2,
    approx_normal,
    exact_cp,
    in_interval,
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

    # Cas Binomial pur : pas de surdispersion
    if rho == 0:
        d = rng.binomial(n, p, size=n_sim)
        return d

    # Cas surdispersé : Beta-Binomiale
    alpha, beta_param = beta_binom_params(p, rho)
    theta = rng.beta(alpha, beta_param, size=n_sim)
    d = rng.binomial(n, theta)
    return d



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
    """
    if rng is None:
        rng = np.random.default_rng()
    if p_star is None:
        p_star = p_true

    d_samples = simulate_beta_binomial(n, p_true, rho, n_sim, rng=rng)

    results = {
        "jeffreys": {"cover": 0, "length_sum": 0.0, "reject_star": 0},
        "cp":       {"cover": 0, "length_sum": 0.0, "reject_star": 0},
        "normal":   {"cover": 0, "length_sum": 0.0, "reject_star": 0},
    }

    for d in d_samples:
        # Jeffreys
        lb_j, ub_j = jeffreys_alpha2(n, d, confidence_level)
        results["jeffreys"]["cover"] += in_interval(lb_j, ub_j, p_true)
        results["jeffreys"]["length_sum"] += (ub_j - lb_j)
        results["jeffreys"]["reject_star"] += int((p_star < lb_j) or (p_star > ub_j))

        # Clopper–Pearson exact
        lb_cp, ub_cp = exact_cp(n, d, confidence_level)
        results["cp"]["cover"] += in_interval(lb_cp, ub_cp, p_true)
        results["cp"]["length_sum"] += (ub_cp - lb_cp)
        results["cp"]["reject_star"] += int((p_star < lb_cp) or (p_star > ub_cp))

        # Approximation normale
        lb_n, ub_n = approx_normal(n, d, confidence_level)
        lb_n = max(0.0, lb_n)
        ub_n = min(1.0, ub_n)
        results["normal"]["cover"] += in_interval(lb_n, ub_n, p_true)
        results["normal"]["length_sum"] += (ub_n - lb_n)
        results["normal"]["reject_star"] += int((p_star < lb_n) or (p_star > ub_n))

    for key, stats in results.items():
        stats["coverage"] = stats["cover"] / n_sim
        stats["avg_length"] = stats["length_sum"] / n_sim
        stats["reject_star_rate"] = stats["reject_star"] / n_sim
        del stats["cover"], stats["length_sum"], stats["reject_star"]

    return results


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ns = [50, 100, 200]   # par ex.
    np_targets = [0.001, 0.005,
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.07, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.60, 0.70, 0.80, 0.90, 1.00,
    1.25, 1.50, 1.75, 2.00, 2.50,
    3.00, 3.50, 4.00, 5.00, 6.00,
    7.50, 10.00,
]
   # valeurs de n*p
    rhos = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.20]          
    n_sim = 5000
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
                    p_star=p_true,          # cas calibré en moyenne
                    confidence_level=conf,
                    rng=rng,
                )
                for method, stats in res.items():
                    rows.append({
                        "n": n,
                        "p_true": p_true,
                        "np_target": np_target,
                        "rho": rho,
                        "method": method,
                        "coverage": stats["coverage"],
                        "avg_length": stats["avg_length"],
                        "reject_star_rate": stats["reject_star_rate"],
                        "conf_level": conf,
                        "n_sim": n_sim,
                    })

    df = pd.DataFrame(rows)
    out_path = data_dir / "beta_binom_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Résultats sauvegardés dans {out_path}")
