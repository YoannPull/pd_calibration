# Paper Replication Guide

This document explains how to replicate the main results of the paper using the **replication-only**
Makefile located at `replication/Makefile`.

---

## 0) What is replicated?

The replication Makefile covers the full paper bundle:

- **(A) Empirical application #1, Mortgage pipeline**  
  Labels, Imputation (anti-leakage), Binning, Model training, Calibration, Scoring, Report, OOS scoring, OOS vintage and grade report, and a paper-ready OOS backtest package.

- **(B) Empirical application #2, S&P grades / LDP**  
  Build a monthly snapshot from a raw ratings file, then generate grade tables and time-series plots.

- **(C) Simulations**  
  Binomial, Beta-Binomial, Temporal drift, and Prior sensitivity, including simulations, plots, and tables.

Main entrypoint:

```bash
make -f replication/Makefile paper_all
````

---

## 1) About the root Makefile

The repository also includes a root-level `Makefile` intended for development and exploratory runs (e.g. broader grids, alternative options, partial pipelines). To replicate the paper results, use `replication/Makefile` only, as it pins the training settings and provides minimal paper entrypoints.

---

## 2) System requirements

* OS: Linux or macOS recommended
* GNU Make
* Python environment managed with **Poetry**

Install:

```bash
poetry install
make -f replication/Makefile help
```

### Alternative (without Poetry)

If you prefer not to use Poetry, install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Note: the replication Makefile uses `poetry run` by default. If you run without Poetry, call the Python entry points directly or adapt the Make targets accordingly.

### Determinism / threading (recommended for replication)

The Makefile enforces single-thread numerical backends by default:

* `OMP_NUM_THREADS=1`
* `MKL_NUM_THREADS=1`
* `OPENBLAS_NUM_THREADS=1`

You may override them, but paper replication should keep defaults.

---

## 3) Data requirements and expected layout

### 3.1 Freddie Mac mortgage data (Empirical app #1)

You must obtain the dataset from the official provider and comply with the terms. The raw Freddie Mac dataset is not distributed with this repository. See `DATA_DISCLAIMER.md`.

Expected layout (high-level):

* `data/raw/mortgage_data/historical_data_YYYYQn/`
  extracted quarterly text files

Replication produces processed data under:

* `data/processed/` (generated)

See:

* `DATA_DISCLAIMER.md`
* `data/README.md`

### 3.2 Corporate ratings disclosures (Empirical app #2)

Empirical application #2 relies on rating history disclosures published under SEC Rule 17g-7(b) (XBRL). This repository does not redistribute any rating history files. In our original experiments we used a snapshot downloaded in 2022, which may no longer be available from third-party mirrors. Replication therefore requires re-downloading the underlying disclosures from official sources (or equivalent public disclosures). See `DATA_DISCLAIMER.md`.

Expected raw file (CSV example name):

* `ldp_application/data/raw/20220601_SP_Ratings_Services_Corporate.csv`

The replication Makefile first builds a monthly snapshot:

* output: `ldp_application/data/processed/sp_corporate_monthly.csv`

Optional helper tool to download and convert disclosures into sorted CSV files:

* [https://github.com/maxonlinux/ratings-history](https://github.com/maxonlinux/ratings-history)

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
   Generates label cohorts and windows.
   Output: `data/processed/default_labels/window=12m/`

2. `impute`
   Fits the imputer on Train and applies it to all splits (anti-leakage).
   Outputs:

   * `data/processed/imputed/`
   * `artifacts/imputer/`

3. `binning_fit`
   Max-Gini binning with monotonicity checks.
   Outputs:

   * `data/processed/binned/`
   * `artifacts/binning_maxgini/`

4. `model_train_final`
   Deterministic final model training with a narrow C-grid around an anchor.
   Outputs:

   * `artifacts/model_from_binned/` (model, calibration, bucket stats, etc.)
   * `data/processed/scored/` (Train and Validation scored)

5. `report`
   Generates an HTML validation report (Train and Validation).
   Output:

   * `reports/model_validation_report.html`

6. OOS diagnostics (included in `pipeline`)

   * `oos_score` to `data/processed/scored/oos_scored.parquet`
   * `oos_vintage_report` to `reports/vintage_grade_oos.html`
   * `oos_backtest_full` to `outputs/oos_backtest/` (paper snapshot package + optional PDF)

### 5.2 C-grid replication logic (important)

Paper replication uses a fast reproducible grid around `C_ANCHOR`:

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

### 6.2 Tables and plots

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

Expected outputs are written in the corresponding experiment output folders (as defined in each module).

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

* plots and tables generated by each experiment module

---

## 9) Troubleshooting

### Missing raw data

* If Freddie Mac files are missing, verify `data/raw/...` layout (see `data/README.md`).
* If the corporate ratings CSV is missing, place it under `ldp_application/data/raw/` as specified above.

### HTML reports do not open automatically

The Makefile uses `open` (macOS). On Linux, open the HTML directly in a browser.

### Non-deterministic results

* Keep `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
* Keep dependency versions pinned (`poetry.lock`)
* Avoid changing the C-grid unless explicitly intended

---

## 10) Cleaning replication outputs

```bash
make -f replication/Makefile clean_replication_outputs
```

This removes:

* `data/processed/`
* `artifacts/`
* `reports/`