# intervals.py
import numpy as np
from scipy.stats import beta, norm, binom
from scipy.optimize import bisect


def jeffreys_alpha2(n_k, d_k, confidence_level=0.95):
    """
    Intervalle de Jeffreys bilatéral :
    prior Beta(1/2, 1/2), posterior Beta(d_k + 1/2, n_k - d_k + 1/2),
    intervalle crédible central.
    """
    alpha_prior = 0.5
    beta_prior = 0.5

    alpha_posterior = d_k + alpha_prior
    beta_posterior = n_k - d_k + beta_prior

    lower_bound = beta.ppf((1 - confidence_level) / 2, alpha_posterior, beta_posterior)
    upper_bound = beta.ppf(1 - (1 - confidence_level) / 2, alpha_posterior, beta_posterior)

    return lower_bound, upper_bound


def jeffreys_billa(n_k, d_k, confidence_level=0.95):
    """
    Intervalle de Jeffreys basé sur la région de densité la plus élevée (HDR).
    """
    alpha_prior = 0.5
    beta_prior = 0.5

    alpha_posterior = d_k + alpha_prior
    beta_posterior = n_k - d_k + beta_prior

    # Grille sur [0, 1]
    x = np.linspace(0, 1, 100_000)
    density = beta.pdf(x, alpha_posterior, beta_posterior)

    # Seuil tel que l'intégrale de la densité au-dessus du seuil = niveau de confiance
    def hdr_threshold(threshold):
        mask = density >= threshold
        return np.trapz(density[mask], x[mask]) - confidence_level

    threshold = bisect(hdr_threshold, 0, density.max())
    hdr_region = x[density >= threshold]
    hdr_lower, hdr_upper = hdr_region.min(), hdr_region.max()

    return hdr_lower, hdr_upper


def approx_normal(n_k, d_k, confidence_level=0.95):
    """
    Intervalle de confiance bilatéral par approximation normale autour de p̂ = d/n.
    (Non corrigé de continuité, à tronquer si besoin dans [0,1].)
    """
    if n_k == 0:
        raise ValueError("n_k doit être > 0")

    p_hat = d_k / n_k
    se = np.sqrt(p_hat * (1 - p_hat) / n_k) if 0 < p_hat < 1 else 0.0
    z = norm.ppf(1 - (1 - confidence_level) / 2)

    lb = p_hat - z * se
    ub = p_hat + z * se

    return lb, ub


def exact_cp(n_k, d_k, confidence_level=0.95):
    """
    Intervalle de Clopper-Pearson bilatéral (exact) pour une proportion Binomiale.
    Formule classique via les quantiles de la loi Beta sur p.

    Référence :
      L = Beta^{-1}(alpha/2; d, n-d+1)   (si d > 0, sinon 0)
      U = Beta^{-1}(1 - alpha/2; d+1, n-d) (si d < n, sinon 1)
    """
    if n_k == 0:
        raise ValueError("n_k doit être > 0")

    alpha = 1 - confidence_level

    if d_k == 0:
        lower_bound = 0.0
    else:
        lower_bound = beta.ppf(alpha / 2, d_k, n_k - d_k + 1)

    if d_k == n_k:
        upper_bound = 1.0
    else:
        upper_bound = beta.ppf(1 - alpha / 2, d_k + 1, n_k - d_k)

    return lower_bound, upper_bound


def jeffreys_ecb(n_k, d_k, confidence_level=0.95):
    """
    Borne supérieure unilatérale Jeffreys (queue gauche) à niveau 'confidence_level'.

    On cherche u tel que P(p <= u | données) = confidence_level
    sous le posterior Beta(d+1/2, n-d+1/2).
    """
    alpha_prior = 0.5
    beta_prior = 0.5

    alpha_posterior = d_k + alpha_prior
    beta_posterior = n_k - d_k + beta_prior

    # borne supérieure unilatérale : quantile de la loi Beta postérieure
    upper = beta.ppf(confidence_level, alpha_posterior, beta_posterior)
    return upper


def exact_cp_unilateral(n_k, d_k, confidence_level=0.95, tail='lower'):
    """
    Intervalle de confiance unilatéral exact (Clopper-Pearson) via la loi Beta.

    - tail='lower' : [LB, 1]
    - tail='upper' : [0, UB]
    """
    if n_k == 0:
        raise ValueError("n_k doit être > 0")

    alpha = 1 - confidence_level

    if tail == 'lower':
        if d_k == 0:
            lb = 0.0
        else:
            lb = beta.ppf(alpha, d_k, n_k - d_k + 1)
        ub = 1.0

    elif tail == 'upper':
        lb = 0.0
        if d_k == n_k:
            ub = 1.0
        else:
            ub = beta.ppf(1 - alpha, d_k + 1, n_k - d_k)

    else:
        raise ValueError("tail doit être 'lower' ou 'upper'")

    return lb, ub


def approx_normal_unilateral(n_k, d_k, confidence_level=0.95, tail='lower'):
    """
    Intervalle de confiance unilatéral par approximation normale.

    - tail='lower' : [LB, 1]
    - tail='upper' : [0, UB]
    """
    if n_k == 0:
        raise ValueError("n_k doit être > 0")

    p_hat = d_k / n_k
    se = np.sqrt(p_hat * (1 - p_hat) / n_k) if 0 < p_hat < 1 else 0.0
    z = norm.ppf(confidence_level)

    if tail == 'lower':
        bound = p_hat - z * se
        lb, ub = bound, 1.0
    elif tail == 'upper':
        bound = p_hat + z * se
        lb, ub = 0.0, bound
    else:
        raise ValueError("tail doit être 'lower' ou 'upper'")

    return lb, ub


def in_interval(lb, ub, p_k):
    """
    Retourne 1 si p_k est dans [lb, ub], sinon 0.
    """
    return int(lb <= p_k <= ub)


def jeffreys_pvalue_unilateral(n_k, d_k, p0, tail="upper"):
    """
    p-value unilatérale sous posterior Jeffreys Beta(d+1/2, n-d+1/2).

    - tail="upper" : p-value = P(p >= p0 | data) = 1 - F(p0)
      (utile pour tester "sous-estimation" : H1 : p > p0)

    - tail="lower" : p-value = P(p <= p0 | data) = F(p0)
      (utile pour tester "sur-estimation" : H1 : p < p0)

    Retourne np.nan si non calculable.
    """
    if n_k <= 0:
        return np.nan
    if d_k < 0 or d_k > n_k:
        return np.nan
    if not np.isfinite(p0):
        return np.nan

    p0 = float(p0)
    if p0 <= 0.0:
        return 1.0 if tail == "upper" else 0.0
    if p0 >= 1.0:
        return 0.0 if tail == "upper" else 1.0

    a = d_k + 0.5
    b = (n_k - d_k) + 0.5
    cdf = beta.cdf(p0, a, b)

    if tail == "upper":
        return float(1.0 - cdf)
    elif tail == "lower":
        return float(cdf)
    else:
        raise ValueError("tail must be 'upper' or 'lower'")
