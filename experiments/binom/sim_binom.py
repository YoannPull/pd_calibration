# experiments/binom_coverage/sim_binom_coverage.py

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from experiments.common.intervals import (
    jeffreys_alpha2,
    approx_normal,
    exact_cp,
    jeffreys_ecb,
    approx_normal_unilateral,
    exact_cp_unilateral,
    in_interval,
)


def simulate_binomial_coverage(
    n: int,
    n_simulation: int,
    p_values: np.ndarray,
    confidence_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Simule la couverture de différentes procédures d'IC binomiaux
    pour une grille de p dans [0,1].

    Retourne un DataFrame avec une ligne par valeur de p, contenant :
      - n, p, conf_level
      - coverage_jeff, coverage_exact, coverage_approx,
        coverage_ecb, coverage_approx_unil, coverage_exact_unil
    """
    if rng is None:
        rng = np.random.default_rng()

    coverage_rates_jeff = []
    coverage_rates_approx = []
    coverage_rates_exact = []
    coverage_rates_ecb = []
    coverage_rates_approx_unil = []
    coverage_rates_exact_unil = []

    for p in tqdm(p_values, desc="Binomial coverage simulations"):
        # d ~ Bin(n, p) répété n_simulation fois
        binomial_samples = rng.binomial(n, p, size=n_simulation)

        # Intervalles pour chaque tirage d_k
        ub_jeffreys_ecb_values = [
            jeffreys_ecb(n, d_k, confidence_level) for d_k in binomial_samples
        ]

        lb_exact_cp_values, ub_exact_cp_values = zip(
            *[exact_cp(n, d_k, confidence_level) for d_k in binomial_samples]
        )

        lb_approx_normal_values, ub_approx_normal_values = zip(
            *[approx_normal(n, d_k, confidence_level) for d_k in binomial_samples]
        )

        lb_jeffreys_2_values, ub_jeffreys_2_values = zip(
            *[jeffreys_alpha2(n, d_k, confidence_level) for d_k in binomial_samples]
        )

        lb_exact_cp_unil_values, ub_exact_cp_unil_values = zip(
            *[
                exact_cp_unilateral(
                    n_k=n, d_k=d_k, confidence_level=confidence_level, tail="upper"
                )
                for d_k in binomial_samples
            ]
        )

        lb_approx_normal_unil_values, ub_approx_normal_unil_values = zip(
            *[
                approx_normal_unilateral(
                    n_k=n, d_k=d_k, confidence_level=confidence_level, tail="upper"
                )
                for d_k in binomial_samples
            ]
        )

        # Couverture (bilatérale)
        coverage_rates_jeff.append(
            np.mean(
                [
                    in_interval(lb, ub, p)
                    for lb, ub in zip(lb_jeffreys_2_values, ub_jeffreys_2_values)
                ]
            )
        )
        coverage_rates_approx.append(
            np.mean(
                [
                    in_interval(lb, ub, p)
                    for lb, ub in zip(lb_approx_normal_values, ub_approx_normal_values)
                ]
            )
        )
        coverage_rates_exact.append(
            np.mean(
                [
                    in_interval(lb, ub, p)
                    for lb, ub in zip(lb_exact_cp_values, ub_exact_cp_values)
                ]
            )
        )

        # Couverture unilatérale [0, ub] (ECb Jeffreys, CP upper, Normal upper)
        coverage_rates_ecb.append(
            np.mean([in_interval(0.0, ub, p) for ub in ub_jeffreys_ecb_values])
        )
        coverage_rates_exact_unil.append(
            np.mean(
                [
                    in_interval(lb, ub, p)
                    for lb, ub in zip(lb_exact_cp_unil_values, ub_exact_cp_unil_values)
                ]
            )
        )
        coverage_rates_approx_unil.append(
            np.mean(
                [
                    in_interval(lb, ub, p)
                    for lb, ub in zip(
                        lb_approx_normal_unil_values, ub_approx_normal_unil_values
                    )
                ]
            )
        )

    df = pd.DataFrame(
        {
            "n": n,
            "p": p_values,
            "conf_level": confidence_level,
            "coverage_jeff": coverage_rates_jeff,
            "coverage_exact": coverage_rates_exact,
            "coverage_approx": coverage_rates_approx,
            "coverage_ecb": coverage_rates_ecb,
            "coverage_approx_unil": coverage_rates_approx_unil,
            "coverage_exact_unil": coverage_rates_exact_unil,
        }
    )

    return df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Paramètres de la simulation (tu peux les adapter)
    n_simulation = 10_000
    n = 100
    p_values = np.linspace(0.001, 0.1, 300)
    confidence_level = 0.95

    rng = np.random.default_rng(123)

    df_coverages = simulate_binomial_coverage(
        n=n,
        n_simulation=n_simulation,
        p_values=p_values,
        confidence_level=confidence_level,
        rng=rng,
    )

    out_path = data_dir / f"binom_coverage_n{n}.csv"
    df_coverages.to_csv(out_path, index=False)
    print(f"Résultats sauvegardés dans {out_path}")
