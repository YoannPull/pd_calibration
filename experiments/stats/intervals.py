# experiments/stats/intervals.py
import numpy as np
from scipy.stats import beta, norm
from scipy.optimize import bisect


# ----------------------------
# Generic Beta(a,b) priors
# ----------------------------
# Common "objective / reference" priors for a Bernoulli/Binomial proportion:
# - Jeffreys: Beta(1/2, 1/2)
# - Laplace (uniform): Beta(1, 1)
# - Haldane: Beta(0, 0) (improper; needs care at d=0 or d=n)
# - Perks: Beta(1, 1) sometimes called uniform too; keep explicit name if you want
# - (optional) "arcsine" is Jeffreys for Bernoulli
DEFAULT_PRIORS: dict[str, tuple[float, float]] = {
    "jeffreys": (0.5, 0.5),
    "laplace": (1.0, 1.0),
    "haldane": (0.0, 0.0),
    "perks": (1.0, 1.0),
}


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def beta_credible_interval(
    n_k: int,
    d_k: int,
    confidence_level: float = 0.95,
    a_prior: float = 0.5,
    b_prior: float = 0.5,
    *,
    kind: str = "equal_tail",
    eps: float = 1e-12,
) -> tuple[float, float]:
    """
    Generic Bayesian credible interval for Binomial proportion with Beta(a_prior, b_prior) prior.

    Posterior: Beta(d_k + a_prior, n_k - d_k + b_prior)

    kind:
      - "equal_tail": central equal-tailed interval
      - "hpd": highest posterior density region (HDR/HPD) via grid + threshold

    eps:
      - safety floor for improper priors / boundary cases.
        If a_prior==0 (Haldane), d_k=0 => posterior alpha=0 (invalid).
        We replace any non-positive posterior shape with eps.

    Notes:
      - This returns bounds clipped to [0,1].
      - For "hpd", a grid-based approximation is used (slower).
    """
    if n_k <= 0:
        raise ValueError("n_k must be > 0")
    if d_k < 0 or d_k > n_k:
        raise ValueError("d_k must be in [0, n_k]")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0,1)")

    a_post = float(d_k) + float(a_prior)
    b_post = float(n_k - d_k) + float(b_prior)

    # Safety for Haldane / numerical edge cases:
    if a_post <= 0.0:
        a_post = eps
    if b_post <= 0.0:
        b_post = eps

    alpha = 1.0 - float(confidence_level)

    if kind == "equal_tail":
        lb = beta.ppf(alpha / 2.0, a_post, b_post)
        ub = beta.ppf(1.0 - alpha / 2.0, a_post, b_post)
        return _clip01(lb), _clip01(ub)

    if kind == "hpd":
        # grid-based HPD (approx)
        x = np.linspace(0.0, 1.0, 200_001)
        dens = beta.pdf(x, a_post, b_post)

        def _mass_above(thr: float) -> float:
            mask = dens >= thr
            # trapezoidal integral of dens over masked points
            return float(np.trapz(dens[mask], x[mask]))

        # Find thr such that mass_above(thr) = confidence_level
        # bisection over [0, max_density]
        maxd = float(np.max(dens))
        if maxd <= 0.0:
            return 0.0, 1.0

        def f(thr: float) -> float:
            return _mass_above(thr) - confidence_level

        thr = bisect(f, 0.0, maxd, maxiter=100)
        region = x[dens >= thr]
        if region.size == 0:
            return 0.0, 1.0
        return float(region.min()), float(region.max())

    raise ValueError("kind must be 'equal_tail' or 'hpd'")


def beta_upper_bound(
    n_k: int,
    d_k: int,
    confidence_level: float = 0.95,
    a_prior: float = 0.5,
    b_prior: float = 0.5,
    *,
    eps: float = 1e-12,
) -> float:
    """
    One-sided upper bound u such that P(p <= u | data) = confidence_level
    under Beta(d+a, n-d+b) posterior. (Generalizes jeffreys_ecb to any prior.)
    """
    if n_k <= 0:
        raise ValueError("n_k must be > 0")
    if d_k < 0 or d_k > n_k:
        raise ValueError("d_k must be in [0, n_k]")

    a_post = float(d_k) + float(a_prior)
    b_post = float(n_k - d_k) + float(b_prior)

    if a_post <= 0.0:
        a_post = eps
    if b_post <= 0.0:
        b_post = eps

    u = beta.ppf(float(confidence_level), a_post, b_post)
    return _clip01(u)


# ----------------------------
# Your existing functions
# (kept for backward compatibility)
# ----------------------------
def jeffreys_alpha2(n_k, d_k, confidence_level=0.95):
    return beta_credible_interval(
        n_k=n_k, d_k=d_k, confidence_level=confidence_level, a_prior=0.5, b_prior=0.5, kind="equal_tail"
    )


def jeffreys_billa(n_k, d_k, confidence_level=0.95):
    return beta_credible_interval(
        n_k=n_k, d_k=d_k, confidence_level=confidence_level, a_prior=0.5, b_prior=0.5, kind="hpd"
    )


def approx_normal(n_k, d_k, confidence_level=0.95):
    if n_k == 0:
        raise ValueError("n_k doit être > 0")
    p_hat = d_k / n_k
    se = np.sqrt(p_hat * (1 - p_hat) / n_k) if 0 < p_hat < 1 else 0.0
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    lb = p_hat - z * se
    ub = p_hat + z * se
    return lb, ub


def exact_cp(n_k, d_k, confidence_level=0.95):
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
    return beta_upper_bound(
        n_k=n_k, d_k=d_k, confidence_level=confidence_level, a_prior=0.5, b_prior=0.5
    )


def exact_cp_unilateral(n_k, d_k, confidence_level=0.95, tail="lower"):
    if n_k == 0:
        raise ValueError("n_k doit être > 0")
    alpha = 1 - confidence_level

    if tail == "lower":
        if d_k == 0:
            lb = 0.0
        else:
            lb = beta.ppf(alpha, d_k, n_k - d_k + 1)
        ub = 1.0
    elif tail == "upper":
        lb = 0.0
        if d_k == n_k:
            ub = 1.0
        else:
            ub = beta.ppf(1 - alpha, d_k + 1, n_k - d_k)
    else:
        raise ValueError("tail doit être 'lower' ou 'upper'")
    return lb, ub


def approx_normal_unilateral(n_k, d_k, confidence_level=0.95, tail="lower"):
    if n_k == 0:
        raise ValueError("n_k doit être > 0")

    p_hat = d_k / n_k
    se = np.sqrt(p_hat * (1 - p_hat) / n_k) if 0 < p_hat < 1 else 0.0
    z = norm.ppf(confidence_level)

    if tail == "lower":
        bound = p_hat - z * se
        lb, ub = bound, 1.0
    elif tail == "upper":
        bound = p_hat + z * se
        lb, ub = 0.0, bound
    else:
        raise ValueError("tail doit être 'lower' ou 'upper'")
    return lb, ub


def in_interval(lb, ub, p_k):
    return int(lb <= p_k <= ub)


def jeffreys_pvalue_unilateral(n_k, d_k, p0, tail="upper"):
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
