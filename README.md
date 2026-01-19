# Credit Risk Pipeline

This repository contains the **code, experiments, and reproducibility scripts associated with the SSRN paper**:

**Pull, Yoann and Hurlin, Christophe**, *A Bayesian Approach to Probability Default Model Calibration: Theoretical and Empirical Insights on the Jeffreys Test* (June 12, 2025).  
Available at SSRN: https://ssrn.com/abstract=5291474  
DOI: http://dx.doi.org/10.2139/ssrn.5291474

The project is driven by a single **Makefile** and is organized into **three blocks**:

- **(A) Empirical application #1 — Main pipeline**  
  Labels → Imputation (anti-leakage) → Binning → Model training + calibration + scoring → Report  
  (+ optional OOS scoring, vintage/grade reports, master-scale recalibration, paper-ready OOS backtest)

- **(B) Empirical application #2 — LDP / S&P grades**  
  Build yearly grade tables (CSV) + time-series plots from a corporate ratings Excel file

- **(C) Simulations**  
  Binomial / Beta-Binomial / Temporal drift / Prior sensitivity (sim + plots + tables)

---

## How to cite

If you use this code, results, or figures, please cite:

Pull, Yoann and Hurlin, Christophe, *A Bayesian Approach to Probability Default Model Calibration: Theoretical and Empirical Insights on the Jeffreys Test* (June 12, 2025). Available at SSRN: https://ssrn.com/abstract=5291474 or http://dx.doi.org/10.2139/ssrn.5291474

### BibTeX
```bibtex
@article{PullHurlin2025Jeffreys,
  title  = {A Bayesian Approach to Probability Default Model Calibration: Theoretical and Empirical Insights on the Jeffreys Test},
  author = {Pull, Yoann and Hurlin, Christophe},
  year   = {2025},
  month  = jun,
  note   = {SSRN Working Paper No. 5291474},
  url    = {https://ssrn.com/abstract=5291474},
  doi    = {10.2139/ssrn.5291474}
}
```

---

## Requirements

* Python environment managed with **Poetry**
* GNU Make

Install dependencies:

```bash
poetry install
```

Show available commands:

```bash
make help
```

---

## Repository structure

* `src/` : main pipeline code (Empirical app #1)
* `data/` : raw inputs + processed datasets (see `data/README.md`)
* `artifacts/` : trained models, binning rules, figures, tables
* `reports/` : HTML validation reports
* `ldp_application/` : LDP / S&P grade tables + plots (Empirical app #2)
* `experiments/` : simulations (Empirical app #3)
* `Makefile` : main entry point (recommended)

---

## Quick start (Empirical application #1)

Run the full main pipeline:

```bash
make pipeline
```

Or step-by-step:

```bash
make labels
make impute
make binning_fit
make model_train_final
make report
```

Outputs:

* processed datasets: `data/processed/`
* artifacts: `artifacts/`
* reports: `reports/`

---

## Empirical application #2 — LDP / S&P grades (Block B)

The LDP / S&P grade application uses corporate rating history data sourced from:
https://ratingshistory.info

This resource aggregates historical rating actions disclosed by multiple agencies under SEC Rule 17g-7(b)
and provides them as CSV files (converted from the original XBRL disclosures).

Input expected:
* `ldp_application/data/raw/data_rating_corporate.xlsx`

Run:

```bash
make sp_grade_all
```

Or separately:

```bash
make sp_grade_tables
make sp_grade_plots
```

Outputs:

* `ldp_application/outputs/sp_grade_is_oos/`
  (yearly tables, combined CSV, and `plots_timeseries/`)

---

## Simulations (Block C)

Run one experiment end-to-end:

```bash
make binom_all
make beta_binom_all
make temporal_drift_all
make prior_sens_all
```

Run all:

```bash
make sims_all
```

---

## Data sources

### Mortgage loan-level data (Empirical app #1)

The main empirical pipeline uses the **Freddie Mac Single-Family Loan-Level Dataset (Standard)**.

Official page:

```text
https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
```

In this repo we use **Standard quarterly files from 2008Q1 to 2024Q4** and store them
as extracted text files under:

* `data/raw/mortgage_data/historical_data_YYYYQn/`

See `data/README.md` for the exact layout and dictionary file.

Important: the dataset is provided by Freddie Mac under its own terms; users must obtain it
themselves and comply with the dataset’s license/terms. See `DATA_DISCLAIMER.md`.

---

## License

* Code in this repository is released under the **MIT License** (see `LICENSE`).
* This license applies to the **code only**. It does **not** grant rights to any third-party data.

---

## Clean everything

Remove all generated outputs (processed data + artifacts + reports):

```bash
make clean_all
```

---
