# `src/` — Main Pipeline (Empirical application #1)

This folder contains the Python code for **Empirical application #1** (the main bank-grade
credit risk pipeline). The Makefile runs these scripts in a fixed order.

> Note: TTC-macro estimation is being removed from the project, so it is **not documented here**.

---

## How to run

Run everything from the **repository root** using the Makefile:

### Full pipeline (recommended)
```bash
make pipeline
````

### Step-by-step

```bash
make labels
make impute
make binning_fit
make model_train_final
make report
```

---

## Pipeline overview (Makefile block A)

### 1) Label generation (`make labels`)

**Script:** `make_labels.py`

Purpose:

* Build the default label (e.g., `default_12m`) using the rules defined in `config.yml`
* Create Train / Validation / OOS splits (depending on your config)

Output (written under `data/processed/`):

* `data/processed/default_labels/window=<LABELS_WINDOW>m/`

  * `train.parquet`, `validation.parquet`, `oos.parquet` (typical)

Main inputs:

* `config.yml`
* raw data (see `data/README.md`)

---

### 2) Imputation (anti-leakage) (`make impute`)

**Script:** `impute_and_save.py`

Purpose:

* Fit the imputer **only on the Train split**
* Transform Train / Validation / OOS consistently (no leakage)
* Fail fast if NaNs remain (when `--fail-on-nan` is enabled)

Outputs:

* Processed datasets:

  * `data/processed/imputed/train.parquet`
  * `data/processed/imputed/validation.parquet`
  * (optionally) `data/processed/imputed/oos.parquet`
* Imputer artifact:

  * `artifacts/imputer/imputer.joblib`

Key Makefile variables:

* `LABELS_WINDOW` → defines `target = default_<LABELS_WINDOW>m`

---

### 3) Binning (monotone, max-gini) (`make binning_fit`)

**Script:** `fit_binning.py`

Purpose:

* Fit monotone binning rules for numeric variables (max-gini with monotonicity checks)
* Save binned Train / Validation datasets
* Save bin definitions for later scoring

Outputs:

* Binned datasets:

  * `data/processed/binned/train.parquet`
  * `data/processed/binned/validation.parquet`
* Binning artifact:

  * `artifacts/binning_maxgini/bins.json`

Main inputs:

* `data/processed/imputed/train.parquet`
* `data/processed/imputed/validation.parquet`

---

### 4) Model training + calibration + scoring (`make model_train_*`)

**Script:** `train_model.py`

Two modes exist in the Makefile:

* `make model_train_iter` : iterative search (halving), larger exploration
* `make model_train_final`: grid search, reproducible final run

Purpose:

* Train a regularized logistic regression model
* Run time-aware cross-validation (configurable)
* Calibrate predicted probabilities (e.g., isotonic)
* Build the master scale (risk buckets / grades)
* Produce scored Train + Validation outputs

Outputs:

* Model + metadata artifacts:

  * `artifacts/model_from_binned/model_best.joblib`
  * `artifacts/model_from_binned/risk_buckets.json`
  * `artifacts/model_from_binned/bucket_stats.json` (typical)
* Scored datasets:

  * `data/processed/scored/train_scored.parquet`
  * `data/processed/scored/validation_scored.parquet`

Key options controlled by Makefile variables:

* search: `SEARCH`, `C_MIN_EXP`, `C_MAX_EXP`, `C_NUM`, `HALVING_FACTOR`
* calibration: `CALIBRATION` (e.g., `isotonic`)
* time-aware CV: `CV_SCHEME`, `CV_TIME_COL`, `CV_TIME_FREQ`
* calibration split: `CAL_SPLIT`, `CAL_SIZE`
* buckets: `N_BUCKETS` (or auto constraints via `N_BUCKETS_CANDIDATES`)
* switches: `NO_INTERACTIONS`, `TIMING`

---

### 5) Reporting (`make report`)

**Script:** `generate_report.py`

Purpose:

* Build a global HTML validation report based on scored Train + Validation
* Summarize performance, calibration, grades, and key diagnostics

Outputs:

* `reports/model_validation_report.html`

Inputs:

* `data/processed/scored/train_scored.parquet`
* `data/processed/scored/validation_scored.parquet`
* `artifacts/model_from_binned/model_best.joblib`

---

## Optional modules (manual targets)

### Score OOS (`make oos_score`)

**Script:** `apply_model.py`

Purpose:

* Apply the full pipeline artifacts (imputer + bins + model + buckets)
* Produce scored OOS dataset

Output:

* `data/processed/scored/oos_scored.parquet`

---

### Vintage/grade reports (`make val_vintage_report`, `make oos_vintage_report`)

**Script:** `generate_vintage_grade_report.py`

Purpose:

* Generate a report by vintage and grade (validation or OOS)

Outputs:

* `reports/vintage_grade_validation.html`
* `reports/vintage_grade_oos.html`

---

### Master scale recalibration (`make recalibrate_pd`, `make val_apply_ms`, `make oos_apply_ms`)

**Scripts:**

* `recalibrate_master_scale.py`
* `apply_master_scale.py`

Purpose:

* Recalibrate grade → PD mapping on a rolling window (Type-1 style)
* Apply recalibrated mapping to Validation / OOS

Outputs (typical):

* `artifacts/model_from_binned/bucket_stats_recalibrated.json`
* `data/processed/scored/validation_scored_ms.parquet`
* `data/processed/scored/oos_scored_ms.parquet`

---

### Paper-ready OOS backtest (`make oos_backtest_full`)

**Scripts:**

* `run_oos_backtest.py` (CLI runner)
* `features/oos_backtest.py` (core logic)

Purpose:

* Produce paper-ready backtesting outputs (tables + plots + LaTeX snapshot)
* Outputs are written to `outputs/oos_backtest/` by default

---

## Conventions / assumptions

* Run scripts from the repo root using Makefile targets.
* The Makefile sets `PYTHONPATH=src` so imports inside `src/` work consistently.
* All generated data goes under `data/processed/`, and artifacts under `artifacts/`.
* IDs / key columns are expected to be stable across the pipeline
  (e.g., `loan_sequence_number` for scoring).

---