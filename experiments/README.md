# PD Calibration – Simulation Experiments

This folder contains **Monte Carlo experiments** used to study the small-sample behaviour of binomial/Beta–binomial confidence intervals and PD validation tests.

All simulations are run from the project root via `make` and use the Python modules under `experiments/`.

---

## 1. Binomial coverage simulations

**Folder:** `experiments/binom_coverage/`  
**Scripts:**
- `sim_binom.py` – runs Monte Carlo simulations.
- `plot_binom.py` – generates coverage plots.

### What it does

For several sample sizes \(n\) and a grid of true default probabilities \(p\), we:

1. Simulate defaults  
   \[
   D \sim \text{Binomial}(n, p)
   \]
   over many replications.
2. Build confidence/credible intervals for \(p\) using:
   - Jeffreys equal-tailed interval,
   - exact Clopper–Pearson interval,
   - normal approximation,
   - Jeffreys ECB upper bound,
   - exact and normal one-sided upper intervals.
3. Estimate the **empirical coverage** of each method as a function of \(p\).
4. In addition, we store summary statistics of the intervals across replications:
   **mean/variance of the bounds** and **mean/variance of interval length**.

Results are saved as CSV in `experiments/binom_coverage/data/` and figures in  
`experiments/binom_coverage/figs/`.

### How to run

From the project root:

```bash
make binom_sim    # run Monte Carlo simulations
make binom_plots  # generate plots from CSV
make binom_all    # run both
````

---

## 2. Beta–binomial robustness simulations

**Folder:** `experiments/beta_binom_jeffreys/`
**Scripts:**

* `sim_beta_binom.py` – runs Monte Carlo simulations.
* `plot_beta_binom.py` – generates coverage plots.

### What it does

To assess robustness to **default clustering / over-dispersion**, we assume that defaults follow a **Beta–Binomial** model:

1. For each configuration ((n, p, \rho)), we simulate:

   * a latent PD (\theta \sim \text{Beta}(\alpha, \beta)) with mean (p) and intra-class correlation (\rho),
   * defaults (D \mid \theta \sim \text{Binomial}(n, \theta)).
2. We then **apply binomial-based intervals as if the data were i.i.d. Binomial**, and compare:

   * Jeffreys interval,
   * exact Clopper–Pearson,
   * normal approximation.
3. For each method we estimate:

   * empirical coverage of the true mean PD (p),
   * average interval length,
   * rejection rate of the model PD under calibration,
   * and (as in the binomial study) **mean/variance of the bounds** and **mean/variance of interval length**.

Results are saved in `experiments/beta_binom_jeffreys/data/`
and plots in `experiments/beta_binom_jeffreys/figs/`.

### How to run

```bash
make beta_binom_sim    # run Monte Carlo simulations
make beta_binom_plots  # generate plots from CSV
make beta_binom_all    # run both
```

---

## 3. Temporal drift experiment

**Folder:** `experiments/temporal_drift/`
**Scripts:**

* `sim_temporal_drift.py` – runs the temporal drift simulation.
* `plot_temporal_drift.py` – generates time-series plots.

### What it does

This experiment mimics an **iterative, period-by-period PD validation** under a drifting true PD:

1. We specify:

   * a fixed **model PD** (\hat p),
   * a **time-varying true PD** (p(t)) that equals (\hat p) up to time (T_0), then drifts linearly away from (\hat p) until time (T).
2. For each period (t = 1,\dots,T) and each replication, we simulate
   [
   D(t) \sim \text{Binomial}\bigl(n, p(t)\bigr)
   ]
   and build intervals for (p) using:

   * Jeffreys equal-tailed interval,
   * exact Clopper–Pearson,
   * normal approximation.
3. For each method and each time (t), we record:

   * **coverage** of the true (p(t)),
   * **rejection rate** of (H_0 : p = \hat p),
   * **mean/variance of the bounds** and **mean/variance of interval length**.
4. The experiment is run for several values of (\hat p) (different PD regimes), with a **relative drift** (\delta) (specified as a multiple of (\hat p)).

Results are saved in `experiments/temporal_drift/data/`
and plots in `experiments/temporal_drift/figs/`.

### How to run

```bash
make temporal_drift_sim    # run Monte Carlo simulations
make temporal_drift_plots  # generate plots from CSV
make temporal_drift_all    # run both
```

---

## 4. Run all simulations at once

To run **all** simulation blocks (binomial, Beta–binomial, temporal drift) and generate all figures in one command:

```bash
make sims_all
```

This will sequentially execute:

* `beta_binom_all`
* `temporal_drift_all`
* `binom_all`