# experiments/temporal_drift/sim_temporal_drift.py

import numpy as np
import pandas as pd
from pathlib import Path

from experiments.beta_binom_jeffreys.intervals import (
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
    # avant ou à T0 : modèle bien calibré
    p_true[t <= T0] = p_hat
    # après T0 : drift linéaire
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
      - la couverture empirique de p_true(t),
      - la longueur moyenne de l'intervalle,
      - la fréquence de rejet de H0: p = p_hat.

    Retourne :
      Un DataFrame avec une ligne par (méthode, t) contenant les statistiques
      agrégées + les paramètres de la simulation.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Trajectoire déterministe de la vraie PD
    p_true_t = temporal_pd_path(T=T, T0=T0, p_hat=p_hat, delta=delta)

    methods = ["jeffreys", "cp", "normal"]
    m_index = {name: i for i, name in enumerate(methods)}
    M = len(methods)

    # Accumulateurs : (méthode, temps)
    coverage_counts = np.zeros((M, T), dtype=float)
    length_sums = np.zeros((M, T), dtype=float)
    reject_counts = np.zeros((M, T), dtype=float)

    alpha = 1.0 - confidence_level

    for sim in range(n_sim):
        # Simule une trajectoire de défauts d(t) ~ Bin(n, p_true(t))
        d_t = rng.binomial(n, p_true_t)

        for t_idx in range(T):
            p_true = p_true_t[t_idx]
            d = d_t[t_idx]

            # Jeffreys (intervalle central)
            m = m_index["jeffreys"]
            lb, ub = jeffreys_alpha2(n, d, confidence_level)
            coverage_counts[m, t_idx] += in_interval(lb, ub, p_true)
            length_sums[m, t_idx] += (ub - lb)
            reject_counts[m, t_idx] += int((p_hat < lb) or (p_hat > ub))

            # Clopper–Pearson exact
            m = m_index["cp"]
            lb_cp, ub_cp = exact_cp(n, d, confidence_level)
            coverage_counts[m, t_idx] += in_interval(lb_cp, ub_cp, p_true)
            length_sums[m, t_idx] += (ub_cp - lb_cp)
            reject_counts[m, t_idx] += int((p_hat < lb_cp) or (p_hat > ub_cp))

            # Approximation normale
            m = m_index["normal"]
            lb_n, ub_n = approx_normal(n, d, confidence_level)
            lb_n = max(0.0, lb_n)
            ub_n = min(1.0, ub_n)
            coverage_counts[m, t_idx] += in_interval(lb_n, ub_n, p_true)
            length_sums[m, t_idx] += (ub_n - lb_n)
            reject_counts[m, t_idx] += int((p_hat < lb_n) or (p_hat > ub_n))

    # Moyennes sur les réplications
    coverage_rates = coverage_counts / n_sim
    avg_lengths = length_sums / n_sim
    reject_rates = reject_counts / n_sim

    # Mise en forme en DataFrame
    rows = []
    times = np.arange(1, T + 1)
    phases = np.where(times <= T0, "pre_drift", "post_drift")

    for method in methods:
        m = m_index[method]
        for t_idx, t in enumerate(times):
            rows.append(
                {
                    "t": t,
                    "phase": phases[t_idx],
                    "method": method,
                    "p_true": p_true_t[t_idx],
                    "coverage": coverage_rates[m, t_idx],
                    "avg_length": avg_lengths[m, t_idx],
                    "reject_rate": reject_rates[m, t_idx],
                    "T": T,
                    "T0": T0,
                    "n": n,
                    "p_hat": p_hat,
                    "delta": delta,
                    "n_sim": n_sim,
                    "conf_level": confidence_level,
                    "alpha_nominal": alpha,
                }
            )

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # Répertoire de base = dossier du script
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Paramètres de la simulation (à adapter à ton papier)
    T = 60          # par ex. 60 mois
    T0 = 24         # drift à partir de t > 24
    n = 100         # taille du portefeuille par période
    p_hat = 0.01    # PD de modèle
    delta = 0.01    # drift total : p(T) = 0.02 ici
    n_sim = 20_000
    conf = 0.95

    df = monte_carlo_temporal_drift(
        T=T,
        T0=T0,
        n=n,
        p_hat=p_hat,
        delta=delta,
        n_sim=n_sim,
        confidence_level=conf,
    )

    out_path = data_dir / "temporal_drift_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Résultats sauvegardés dans {out_path}")

    # Petit résumé console : moyenne des taux de rejet avant/après drift par méthode
    summary = (
        df.groupby(["method", "phase"])["reject_rate"]
        .mean()
        .reset_index()
        .pivot(index="method", columns="phase", values="reject_rate")
    )
    print("\nTaux de rejet moyen de H0: p = p_hat (par méthode, avant/après drift) :")
    print(summary)
