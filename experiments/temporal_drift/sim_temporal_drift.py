# experiments/temporal_drift/sim_temporal_drift.py

import numpy as np
import pandas as pd
from pathlib import Path

from experiments.stats.intervals import (
    jeffreys_alpha2,
    exact_cp,
    approx_normal,
    in_interval,
)


def temporal_pd_path(T, T0, p_hat, delta):
    """
    Construit la trajectoire temporelle de la vraie PD p(t) avec drift linéaire.

    - T   : nombre total de périodes (t = 1,...,T)
    - T0  : période de début du drift (jusqu'à T0 : p(t) = p_hat)
    - p_hat : PD de modèle (constante dans le temps)
    - delta : amplitude totale du drift (p(T) = p_hat + delta)

    Retourne : tableau numpy de taille T, p_true[t-1] = p(t).
    """
    t = np.arange(1, T + 1)

    p_true = np.empty(T, dtype=float)
    p_true[t <= T0] = p_hat

    mask_drift = t > T0
    if np.any(mask_drift):
        p_true[mask_drift] = p_hat + delta * (t[mask_drift] - T0) / (T - T0)

    return p_true


def monte_carlo_temporal_drift(
    T=60,
    T0=24,
    n=100,
    p_hat=0.01,
    delta=0.01,
    n_sim=10_000,
    confidence_level=0.95,
    rng=None,
):
    """
    Monte Carlo pour la simulation de drift temporel.

    À chaque réplication :
      - On génère une trajectoire p_true(t) avec drift (ou non) autour de p_hat.
      - On simule d(t) ~ Bin(n, p_true(t)) pour t = 1,...,T.
      - À chaque t, on applique plusieurs procédures (Jeffreys, CP, Normale)
        pour tester H0: p = p_hat et construire un intervalle pour p.

    On agrège ensuite, pour chaque méthode et chaque t :
      - couverture empirique de p_true(t),
      - longueur moyenne + variance de longueur,
      - moyenne + variance des bornes (LB/UB),
      - fréquence de rejet de H0: p = p_hat.

    Retourne :
      Un DataFrame avec une ligne par (méthode, t) contenant les statistiques.
    """
    if rng is None:
        rng = np.random.default_rng()

    p_true_t = temporal_pd_path(T=T, T0=T0, p_hat=p_hat, delta=delta)

    methods = ["jeffreys", "cp", "normal"]
    m_index = {name: i for i, name in enumerate(methods)}
    M = len(methods)

    # Accumulateurs (méthode, temps)
    coverage_counts = np.zeros((M, T), dtype=float)
    reject_counts = np.zeros((M, T), dtype=float)

    # Bornes : sommes et sommes des carrés
    lb_sum = np.zeros((M, T), dtype=float)
    lb_sumsq = np.zeros((M, T), dtype=float)
    ub_sum = np.zeros((M, T), dtype=float)
    ub_sumsq = np.zeros((M, T), dtype=float)

    # Longueur : somme et somme des carrés
    len_sum = np.zeros((M, T), dtype=float)
    len_sumsq = np.zeros((M, T), dtype=float)

    alpha = 1.0 - confidence_level

    for _ in range(n_sim):
        d_t = rng.binomial(n, p_true_t)

        for t_idx in range(T):
            p_true = p_true_t[t_idx]
            d = d_t[t_idx]

            # -----------------------------
            # Jeffreys (intervalle central)
            # -----------------------------
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

            # -----------------------------
            # Clopper–Pearson exact
            # -----------------------------
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

            # -----------------------------
            # Approximation normale (clippée)
            # -----------------------------
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

    # Moyennes
    coverage_rates = coverage_counts / n_sim
    reject_rates = reject_counts / n_sim

    lb_mean = lb_sum / n_sim
    ub_mean = ub_sum / n_sim
    len_mean = len_sum / n_sim

    # Variances (population) : Var(X)=E[X^2]-E[X]^2
    lb_var = (lb_sumsq / n_sim) - lb_mean**2
    ub_var = (ub_sumsq / n_sim) - ub_mean**2
    len_var = (len_sumsq / n_sim) - len_mean**2

    # Mise en forme en DataFrame
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
                    "avg_length": float(len_mean[m, t_idx]),  # compat plots existants
                    "reject_rate": float(reject_rates[m, t_idx]),
                    # Nouvelles stats
                    "lb_mean": float(lb_mean[m, t_idx]),
                    "ub_mean": float(ub_mean[m, t_idx]),
                    "lb_var": float(lb_var[m, t_idx]),
                    "ub_var": float(ub_var[m, t_idx]),
                    "len_mean": float(len_mean[m, t_idx]),
                    "len_var": float(len_var[m, t_idx]),
                    # Paramètres de simu
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

    # Paramètres globaux
    T = 68
    T0 = 24
    n = 100
    n_sim = 20_000
    conf = 0.95

    # Plusieurs niveaux de PD de modèle (p_hat)
    # Reco: quelques valeurs couvrant PD faible -> moyenne
    p_hats = [0.005, 0.01, 0.02]

    # Drift relatif: delta = k * p_hat (ici: fin à 2x p_hat)
    k_delta = 1.0

    rng = np.random.default_rng(123)

    dfs = []
    for p_hat in p_hats:
        delta = k_delta * p_hat

        df_hat = monte_carlo_temporal_drift(
            T=T,
            T0=T0,
            n=n,
            p_hat=p_hat,
            delta=delta,
            n_sim=n_sim,
            confidence_level=conf,
            rng=rng,
        )
        dfs.append(df_hat)

    df = pd.concat(dfs, ignore_index=True)

    out_path = data_dir / "temporal_drift_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Résultats sauvegardés dans {out_path}")

    # Petit résumé console : moyenne des taux de rejet avant/après drift par méthode et p_hat
    summary = (
        df.groupby(["p_hat", "method", "phase"])["reject_rate"]
        .mean()
        .reset_index()
        .pivot(index=["p_hat", "method"], columns="phase", values="reject_rate")
    )
    print("\nTaux de rejet moyen de H0: p = p_hat (par méthode, avant/après drift, par p_hat) :")
    print(summary)
