# Paper Replication Guide (replication/Makefile)

This document explains how to replicate the main results of the paper using the **replication-only**
Makefile located at `replication/Makefile`.

> Where to put this file: **`replication/README.md`** (next to `replication/Makefile`).

---

## 0) What is replicated?

The replication Makefile covers the full paper bundle:

- **(A) Empirical application #1 — Mortgage pipeline**
  Labels → Imputation (anti-leakage) → Binning → Model training + calibration + scoring
  → Report + OOS scoring + OOS vintage/grade report + paper-ready OOS backtest package.

- **(B) Empirical application #2 — S&P grades / LDP**
  Build a monthly snapshot from a raw ratings file → grade tables → time-series plots.

- **(C) Simulations**
  Binomial / Beta-Binomial / Temporal drift / Prior sensitivity (simulation + plots + tables).

Main entrypoint:
```bash
make -f replication/Makefile paper_all
```

---

## 1) About the root Makefile

The repository also includes a root-level `Makefile` intended for development and exploratory runs
(e.g., broader grids, alternative options, partial pipelines). To replicate the paper results,
use `replication/Makefile` only, as it pins the training settings and provides minimal paper entrypoints.

---

## 2) System requirements

* OS: Linux/macOS recommended
* GNU Make
* Python + Poetry (project-managed environment)

Install:

```bash
poetry install
make -f replication/Makefile help
```

### Determinism / threading (recommended for replication)

The Makefile enforces single-thread numerical backends by default:

* `OMP_NUM_THREADS=1`
* `MKL_NUM_THREADS=1`
* `OPENBLAS_NUM_THREADS=1`

You may override them, but **paper replication** should keep defaults.

---

## 3) Data requirements & expected layout

### 3.1 Freddie Mac mortgage data (Empirical app #1)

You must obtain the dataset from the official provider and comply with the terms.

Expected layout (high-level):

* `data/raw/mortgage_data/historical_data_YYYYQn/`

  * extracted quarterly text files

Replication produces processed data under:

* `data/processed/` (generated)

See:

* `DATA_DISCLAIMER.md`
* `data/README.md`

### 3.2 S&P corporate ratings file (Empirical app #2)

Expected raw file:

* `ldp_application/data/raw/20220601_SP_Ratings_Services_Corporate.csv`

The replication Makefile first builds a monthly snapshot:

* output: `ldp_application/data/processed/sp_corporate_monthly.csv`

---

## 4) Running the replication

### 4.1 Full paper bundle (A + B + C)

```bash
make -f replication/Makefile paper_all
```

### 4.2 Run components separately

(A) Mortgage pipeline:

```bash
make -f replication/Makefile pipeline
```

(B) S&P grades / LDP:

```bash
make -f replication/Makefile sp_grade_all
```

(C) Simulations:

```bash
make -f replication/Makefile sims_all
```

---

## 5) Main mortgage pipeline details (A)

### 5.1 Stages and outputs

Targets run in this order:

1. `labels`

* generates labels cohorts/windows
* output: `data/processed/default_labels/window=12m/`

2. `impute`

* fits imputer on Train and applies to all splits (anti-leakage)
* outputs:

  * `data/processed/imputed/`
  * `artifacts/imputer/`

3. `binning_fit`

* Max-Gini binning + monotonicity checks
* outputs:

  * `data/processed/binned/`
  * `artifacts/binning_maxgini/`

4. `model_train_final`

* deterministic “FINAL” model training with a **narrow C-grid around an anchor**
* outputs:

  * `artifacts/model_from_binned/` (model, calibration, bucket stats, etc.)
  * `data/processed/scored/` (train/validation scored)

5. `report`

* generates an HTML validation report (Train + Validation)
* output:

  * `reports/model_validation_report.html`

6. OOS diagnostics (included in `pipeline`)

* `oos_score` → `data/processed/scored/oos_scored.parquet`
* `oos_vintage_report` → `reports/vintage_grade_oos.html`
* `oos_backtest_full` → `outputs/oos_backtest/` (paper snapshot package + optional PDF)

### 5.2 C-grid replication logic (important)

Paper replication uses a **fast reproducible grid around `C_ANCHOR`**:

* `C_ANCHOR` (default: 468.633503)
* `C_EXP_SPAN` (default: 0.20 in log10 units)
* `C_NUM` (default: 13 grid points)

Override example:

```bash
make -f replication/Makefile model_train_final \
  C_ANCHOR=234.286842 C_EXP_SPAN=0.25 C_NUM=21
```

The Makefile computes:

* `C_MIN_EXP = log10(C_ANCHOR) - C_EXP_SPAN`
* `C_MAX_EXP = log10(C_ANCHOR) + C_EXP_SPAN`

---

## 6) S&P grades / LDP replication (B)

### 6.1 Build the monthly snapshot

```bash
make -f replication/Makefile sp_snapshot
```

### 6.2 Tables + plots

```bash
make -f replication/Makefile sp_grade_all
```

Outputs:

* tables: `ldp_application/outputs/sp_grade_is_oos/sp_grade_tables_*.csv`
* plots:  `ldp_application/outputs/sp_grade_is_oos/plots_timeseries/`

---

## 7) Simulations replication (C)

Run all simulation modules:

```bash
make -f replication/Makefile sims_all
```

Or one-by-one:

```bash
make -f replication/Makefile binom_all
make -f replication/Makefile beta_binom_all
make -f replication/Makefile temporal_drift_all
make -f replication/Makefile prior_sens_all
```

Expected outputs are written in the corresponding experiment output folders
(as defined in each module).

---

## 8) Output checklist (sanity checks)

After `paper_all`, you should typically have:

(A) Mortgage pipeline

* `reports/model_validation_report.html`
* `reports/vintage_grade_oos.html`
* `outputs/oos_backtest/`
* `artifacts/` (model, binning, imputer)
* `data/processed/` (labels, imputed, binned, scored)

(B) S&P grades / LDP

* `ldp_application/outputs/sp_grade_is_oos/sp_grade_tables_*.csv`
* `ldp_application/outputs/sp_grade_is_oos/plots_timeseries/`

(C) Simulations

* plots/tables generated by each experiment module

---

## 9) Troubleshooting

### Missing raw data

* If Freddie Mac files are missing: verify `data/raw/...` layout (see `data/README.md`).
* If S&P file is missing: place it under `ldp_application/data/raw/` as specified above.

### HTML reports do not open automatically

The Makefile uses `open` (macOS). On Linux, open the HTML directly in a browser.

### Non-deterministic results

* Keep `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
* Keep dependency versions pinned (`poetry.lock`)
* Avoid changing `C_ANCHOR`/grid unless explicitly intended

---

## 10) Cleaning replication outputs

```bash
make -f replication/Makefile clean_replication_outputs
```

This removes:

* `data/processed/`
* `artifacts/`
* `reports/`

