# Credit Risk Modeling Pipeline (PD Model)

Bank-grade Probability of Default (PD) modeling & calibration in Python.

This project implements an end-to-end credit risk modeling pipeline (mortgage loans) with:

- generation of default labels (e.g. `default_12m`) from raw data,
- strict imputation (anti-leakage),
- monotonic binning (Max |Gini|) + WOE transformation,
- logistic regression for risk ranking,
- probability calibration (isotonic),
- construction of a Master Scale (rating grid),
- scoring for Train / Validation / OOS,
- generation of HTML reports (global + vintage / grade),
- master scale recalibration (grade -> PD) using either:
  - mean aggregation, or
  - isotonic smoothing on grade-level default rates,
- application of a recalibrated Master Scale to scored datasets (adds `pd_ms`).

The architecture is designed for:

- temporal robustness (vintage-based splits),
- data leakage prevention,
- interpretability (WOE, LR, master scale, TTC/LRA tables),
- industrialization (Makefile, frozen artifacts).

---

## 1. Project structure

```plaintext
.
├── Makefile
├── config.yml
├── pyproject.toml
├── src/
│   ├── make_labels.py
│   ├── impute_and_save.py
│   ├── fit_binning.py
│   ├── train_model.py
│   ├── apply_model.py
│   ├── recalibrate_master_scale.py
│   ├── apply_master_scale.py
│   ├── estimate_ttc_macro.py
│   ├── generate_report.py
│   ├── generate_vintage_grade_report.py
│   └── features/
│       └── binning.py
├── data/
│   ├── raw/
│   └── processed/
└── artifacts/
````

For a detailed description of each pipeline step, see:
`docs/pipeline.md` (to be created).

---

## 2. Quick start

### Prerequisites

* Python 3.9+
* Poetry
* Make

### Install dependencies

```bash
poetry install
```

### Run the full pipeline (Train + Validation + report)

```bash
make pipeline
```

This command runs:

1. label generation,
2. imputation (fit on Train / transform on Train & Validation),
3. monotonic binning (Max |Gini|),
4. model training (WOE + LR + isotonic calibration + master scale) and scoring,
5. TTC macro estimation by grade,
6. global HTML report generation.

Main outputs are written to:

* `data/processed/` (imputed, binned, scored datasets),
* `artifacts/` (imputer, bins, model, master scale, calibration tables),
* `reports/` (HTML reports).

---

## 3. Main Make targets

### Full pipeline

```bash
make pipeline
```

### Individual steps

```bash
make labels
make impute
make binning_fit
make model_train
make ttc_macro
make report
```

### OOS scoring and reports

```bash
make oos_score
make oos_vintage_report
make val_vintage_report
```

### Custom scoring

```bash
make score_custom CUSTOM_DATA=path/to/file.parquet CUSTOM_OUT=path/to/out.parquet
```

### Cleanup

```bash
make clean_all
```

---

## 4. Master Scale recalibration (grade -> PD)

The project supports a recalibration workflow where the rating grid (grades / edges) is kept fixed, while the PD attached to each grade is recomputed from scored historical data.

Two aggregation modes are supported:

* pooled: computes PD per grade as sum(defaults) / sum(observations) over the full window; this is a volume-weighted estimate,
* time_mean: computes a one-year default rate per grade and per time period (e.g. vintage quarter), then averages these default rates across time; each period contributes equally.

Two calibration methods are supported:

* mean: uses the raw grade-level PD estimate,
* isotonic: applies isotonic regression to enforce monotonic grade -> PD (volume-weighted).

### Recalibrate grade -> PD

```bash
make recalibrate_pd RECAL_METHOD=mean RECAL_AGG=time_mean RECAL_YEARS=5
```

Examples:

```bash
make recalibrate_pd RECAL_METHOD=mean RECAL_AGG=pooled RECAL_YEARS=5
make recalibrate_pd RECAL_METHOD=isotonic RECAL_AGG=time_mean RECAL_YEARS=7
```

Output:

* `artifacts/model_from_binned/bucket_stats_recalibrated.json`

---

## 5. Apply a recalibrated Master Scale to scored datasets

Once `bucket_stats_recalibrated.json` exists, you can apply the recalibrated grade -> PD mapping to scored datasets. This creates files with an additional column:

* `pd_ms`: PD from the recalibrated Master Scale (grade-level PD)

Validation:

```bash
make val_apply_ms
```

OOS:

```bash
make oos_apply_ms
```

Outputs:

* `data/processed/scored/validation_scored_ms.parquet`
* `data/processed/scored/oos_scored_ms.parquet`

You can then point reporting scripts to `pd_ms` (instead of `pd`) if you want reports driven by the recalibrated Master Scale PD.

---

## 6. Modeling principles & governance

### Anti-leakage

* Imputation is fitted on Train only.
* Binning and WOE are learned on Train and then frozen.
* No explicit time variables (e.g. `vintage`, `quarter`, `year`) are used as features.

### Monotonicity

* Monotonic binning with respect to default rate using Max |Gini|.
* Master scale with PD monotonicity checks across grades.
* Optional isotonic smoothing at the grade level during recalibration.

### Interpretability

* Main model: logistic regression on WOE features.
* Interactions are limited and explicit.
* Coefficients and performance metrics are saved and surfaced in reports.

For detailed modeling choices (WOE, isotonic calibration, master scale construction, and recalibration):
`docs/modeling_details.md` (to be created).

---

## 7. License & intended use

This project was developed in the context of bank-grade PD model calibration and validation.