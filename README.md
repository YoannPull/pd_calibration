# Credit Risk Modeling Pipeline (PD Model)

**Bank-grade Probability of Default (PD) modeling & calibration in Python**

This project implements an end-to-end **credit risk modeling pipeline** (mortgage loans) with:

- generation of default labels (`default_24m`) from raw data,
- strict imputation (anti-leakage),
- monotonic binning (Max |Gini|) + WOE transformation,
- calibrated logistic regression (isotonic),
- construction of a **Master Scale** (rating grid),
- scoring for Train / Validation / OOS,
- generation of **HTML reports** (global + vintage / grade).

The architecture is designed for:

- **temporal robustness** (vintage-based splits),
- **data leakage prevention**,
- **interpretability** (WOE, LR, master scale, TTC tables),
- **industrialization** (Makefile, frozen artifacts).

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
➡️ `docs/pipeline.md` (to be created).

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
4. model training (WOE + LR + calibration + master scale),
5. global HTML report generation.

Main outputs are written to:

* `data/processed/` (imputed, binned, scored datasets),
* `artifacts/` (imputer, bins, model, master scale),
* `reports/` (HTML reports).

---

## 3. Main Make targets

```bash
# Full pipeline
make pipeline

# Individual steps
make labels
make impute
make binning_fit
make model_train
make report

# OOS scoring + vintage/grade report
make oos_score
make oos_vintage_report

# Vintage/grade report on Validation
make val_vintage_report

# Custom scoring
make score_custom CUSTOM_DATA=path/to/file.parquet CUSTOM_OUT=path/to/out.parquet

# Full cleanup
make clean_all
```

Detailed specifications for the HTML reports are described in:
➡️ `docs/reports.md` (to be created).

---

## 4. Modeling principles & governance

### Anti-leakage

* Imputation is **fitted on Train only**.
* Binning & WOE are learned on Train and then **frozen**.
* No explicit time variables (e.g. `vintage`, `quarter`, `year`) are used as features.

### Monotonicity

* Monotonic binning w.r.t. default rate using Max |Gini|.
* Master scale with PD monotonicity checks across grades.
* Vintage/grade reports with explicit monotonicity diagnostics.

### Interpretability

* Main model: **logistic regression** on WOE features.
* Interactions are limited and explicit.
* Coefficients and performance metrics are documented in the HTML reports.

For detailed modeling choices (WOE, isotonic calibration, master scale, monotonicity checks, etc.):
➡️ `docs/modeling_details.md` (to be created).

---

## 5. License & intended use

This project was developed in the context of **bank-grade PD model calibration and validation**.

* **License**: Proprietary / Internal.
* Intended use: experimentation, backtesting, stress testing, and potential integration into production scoring chains.

