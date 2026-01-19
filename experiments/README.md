# Experiments (Simulations)

This folder contains simulation studies used to benchmark PD calibration / interval methods
under controlled data-generating processes.

Each experiment typically follows the same structure:
- `sim_*.py` : generate simulation results (raw outputs saved to disk)
- `plot_*.py` : read saved results and create figures
- `make_table_*.py` : read saved results and create summary tables

All experiments are designed to be run from the **repository root** using the Makefile.

---

## How to run

From the repo root:

### Run a full experiment (sim + plots + tables)
```bash
make binom_all
make beta_binom_all
make temporal_drift_all
make prior_sens_all
````

### Run everything

```bash
make sims_all
```

### Run only one step

```bash
make binom_sim
make binom_plots
make binom_tables
```

---

## Available experiments

### 1) Binomial coverage (`experiments/binom_coverage/`)

Goal: evaluate coverage / error rates under the Binomial model (i.i.d. defaults).

Make targets:

```bash
make binom_sim
make binom_plots
make binom_tables
make binom_all
```

Python modules:

* `experiments.binom_coverage.sim_binom`
* `experiments.binom_coverage.plot_binom`
* `experiments.binom_coverage.make_table_binom`

---

### 2) Beta-Binomial robustness (`experiments/beta_binom_jeffreys/`)

Goal: test robustness to **default clustering / over-dispersion** (latent heterogeneity).

Make targets:

```bash
make beta_binom_sim
make beta_binom_plots
make beta_binom_tables
make beta_binom_all
```

Python modules:

* `experiments.beta_binom_jeffreys.sim_beta_binom`
* `experiments.beta_binom_jeffreys.plot_beta_binom`
* `experiments.beta_binom_jeffreys.make_table_beta_binom`

---

### 3) Temporal drift (`experiments/temporal_drift/`)

Goal: test robustness when the true PD changes over time (non-stationarity).

Make targets:

```bash
make temporal_drift_sim
make temporal_drift_plots
make temporal_drift_tables
make temporal_drift_all
```

Python modules:

* `experiments.temporal_drift.sim_temporal_drift`
* `experiments.temporal_drift.plot_temporal_drift`
* `experiments.temporal_drift.make_table_temporal_drift`

---

### 4) Prior sensitivity (`experiments/prior_sensibility/`)

Goal: compare results across different Beta priors (Jeffreys, Bayes-Laplace, etc.).

Make targets:

```bash
make prior_sens_sim
make prior_sens_plots
make prior_sens_tables
make prior_sens_all
```

Python modules:

* `experiments.prior_sensibility.sim_prior_sensibility`
* `experiments.prior_sensibility.plot_prior_sensibility`
* `experiments.prior_sensibility.make_table_prior_sensibility`

---

## Outputs

Depending on the experiment, outputs are saved to:

* `artifacts/` (recommended for final plots/tables), and/or
* experiment-specific output folders (if implemented inside each experiment).

Typical outputs:

* simulation result files (CSV/Parquet)
* figures (PNG/PDF)
* summary tables (CSV/LaTeX)

---

## Reproducibility guidelines

* Run experiments via `make ...` to ensure consistent paths.
* Fix random seeds inside `sim_*.py` (or log them).
* Store parameters used for each run (grid values, sample sizes, drift settings, etc.)
  alongside the saved results.

---

## Troubleshooting

* If plots/tables fail, run the simulation step first:

  ```bash
  make <experiment>_sim
  ```
* If you changed code and want to regenerate everything:

  ```bash
  make clean_all
  make sims_all
  ```