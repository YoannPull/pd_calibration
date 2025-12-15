# experiments/binom_coverage/sim_binom.py

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from experiments.stats.intervals import (
    jeffreys_alpha2,
    approx_normal,
    exact_cp,
    jeffreys_ecb,
    approx_normal_unilateral,
    exact_cp_unilateral,
)


def _summarize_bounds(lb: np.ndarray, ub: np.ndarray) -> dict[str, float]:
    """
    Calcule des stats sur les bornes et la longueur.
    Variance = np.var(..., ddof=0) (variance "population").
    """
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


def simulate_binomial_coverage_for_n(
    n: int,
    n_simulation: int,
    p_values: np.ndarray,
    confidence_level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simule la couverture pour un n donné, sur toute la grille de p.
    Retourne un DataFrame avec colonnes :
      n, p, conf_level,
      coverage_*,
      lb_mean_*, ub_mean_*, lb_var_*, ub_var_*,
      len_mean_*, len_var_*.
    """

    # Méthodes et colonnes de sortie (noms courts et stables)
    methods_cols = {
        "jeff": "jeffreys_equal_tailed",
        "exact": "clopper_pearson_equal_tailed",
        "approx": "normal_equal_tailed",
        "ecb": "jeffreys_ecb_upper",
        "exact_unil": "clopper_pearson_upper",
        "approx_unil": "normal_upper",
    }

    # Stockage par colonne
    out = {
        "n": [],
        "p": [],
        "conf_level": [],
        # coverages
        "coverage_jeff": [],
        "coverage_exact": [],
        "coverage_approx": [],
        "coverage_ecb": [],
        "coverage_exact_unil": [],
        "coverage_approx_unil": [],
    }

    # Ajout des colonnes de stats pour chaque méthode
    for key in methods_cols.keys():
        out[f"lb_mean_{key}"] = []
        out[f"ub_mean_{key}"] = []
        out[f"lb_var_{key}"] = []
        out[f"ub_var_{key}"] = []
        out[f"len_mean_{key}"] = []
        out[f"len_var_{key}"] = []

    for p in tqdm(p_values, desc=f"Binomial coverage (n={n})"):
        binomial_samples = rng.binomial(n, p, size=n_simulation)

        # --- Intervalles (vectorisés via compréhension -> arrays) ---
        # Bilatéraux
        lb_exact_cp, ub_exact_cp = zip(*[exact_cp(n, d_k, confidence_level) for d_k in binomial_samples])
        lb_approx, ub_approx = zip(*[approx_normal(n, d_k, confidence_level) for d_k in binomial_samples])
        lb_jeff, ub_jeff = zip(*[jeffreys_alpha2(n, d_k, confidence_level) for d_k in binomial_samples])

        # Unilatéraux (upper)
        lb_exact_u, ub_exact_u = zip(*[
            exact_cp_unilateral(n_k=n, d_k=d_k, confidence_level=confidence_level, tail="upper")
            for d_k in binomial_samples
        ])
        lb_approx_u, ub_approx_u = zip(*[
            approx_normal_unilateral(n_k=n, d_k=d_k, confidence_level=confidence_level, tail="upper")
            for d_k in binomial_samples
        ])

        # ECB: upper uniquement -> on prend LB=0
        ub_ecb = np.array([jeffreys_ecb(n, d_k, confidence_level) for d_k in binomial_samples], dtype=float)
        lb_ecb = np.zeros_like(ub_ecb)

        # Cast en np.array une fois
        lb_exact_cp = np.asarray(lb_exact_cp, dtype=float)
        ub_exact_cp = np.asarray(ub_exact_cp, dtype=float)
        lb_approx = np.asarray(lb_approx, dtype=float)
        ub_approx = np.asarray(ub_approx, dtype=float)
        lb_jeff = np.asarray(lb_jeff, dtype=float)
        ub_jeff = np.asarray(ub_jeff, dtype=float)
        lb_exact_u = np.asarray(lb_exact_u, dtype=float)
        ub_exact_u = np.asarray(ub_exact_u, dtype=float)
        lb_approx_u = np.asarray(lb_approx_u, dtype=float)
        ub_approx_u = np.asarray(ub_approx_u, dtype=float)

        # --- Coverage ---
        out["n"].append(n)
        out["p"].append(float(p))
        out["conf_level"].append(float(confidence_level))

        out["coverage_jeff"].append(_coverage(lb_jeff, ub_jeff, p))
        out["coverage_exact"].append(_coverage(lb_exact_cp, ub_exact_cp, p))
        out["coverage_approx"].append(_coverage(lb_approx, ub_approx, p))
        out["coverage_ecb"].append(_coverage(lb_ecb, ub_ecb, p))
        out["coverage_exact_unil"].append(_coverage(lb_exact_u, ub_exact_u, p))
        out["coverage_approx_unil"].append(_coverage(lb_approx_u, ub_approx_u, p))

        # --- Stats bornes/longueur ---
        stats = {
            "jeff": _summarize_bounds(lb_jeff, ub_jeff),
            "exact": _summarize_bounds(lb_exact_cp, ub_exact_cp),
            "approx": _summarize_bounds(lb_approx, ub_approx),
            "ecb": _summarize_bounds(lb_ecb, ub_ecb),
            "exact_unil": _summarize_bounds(lb_exact_u, ub_exact_u),
            "approx_unil": _summarize_bounds(lb_approx_u, ub_approx_u),
        }

        for key, s in stats.items():
            out[f"lb_mean_{key}"].append(s["lb_mean"])
            out[f"ub_mean_{key}"].append(s["ub_mean"])
            out[f"lb_var_{key}"].append(s["lb_var"])
            out[f"ub_var_{key}"].append(s["ub_var"])
            out[f"len_mean_{key}"].append(s["len_mean"])
            out[f"len_var_{key}"].append(s["len_var"])

    return pd.DataFrame(out)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Grille commune à tous les n
    n_simulation = 1000
    ns = [50, 100, 200]
    p_values = np.linspace(0.001, 0.1, 300)
    confidence_level = 0.95

    rng = np.random.default_rng(123)

    dfs = []
    for n in ns:
        df_n = simulate_binomial_coverage_for_n(
            n=n,
            n_simulation=n_simulation,
            p_values=p_values,
            confidence_level=confidence_level,
            rng=rng,
        )
        dfs.append(df_n)

    df_all = pd.concat(dfs, ignore_index=True)

    out_path = data_dir / "binom_coverage_all_n.csv"
    df_all.to_csv(out_path, index=False)
    print(f"Résultats sauvegardés dans {out_path}")
