# Credit Risk Pipeline — Paper Replication Package

This repository contains the **code, experiments, and reproducibility material** associated with the SSRN paper:

**Pull, Yoann and Hurlin, Christophe**, *A Bayesian Approach to Probability of Default Model Calibration: Theoretical and Empirical Insights on the Jeffreys Test* (June 12, 2025).  
SSRN: https://ssrn.com/abstract=5291474  
DOI: http://dx.doi.org/10.2139/ssrn.5291474

> **Paper-grade replication entrypoints** are provided via a dedicated Makefile:
> **`replication/Makefile`** (recommended starting point).
>
> Note: the repository also contains a root `Makefile` used for development and exploratory runs.
> For paper-grade replication, please use `replication/Makefile` exclusively.

---

## How to cite

If you use this code, results, or figures, please cite:

Pull, Yoann and Hurlin, Christophe, *A Bayesian Approach to Probability of Default Model Calibration:
Theoretical and Empirical Insights on the Jeffreys Test* (June 12, 2025). Available at SSRN:
https://ssrn.com/abstract=5291474 or http://dx.doi.org/10.2139/ssrn.5291474

### BibTeX
```bibtex
@article{PullHurlin2025Jeffreys,
  title  = {A Bayesian Approach to Probability of Default Model Calibration: Theoretical and Empirical Insights on the Jeffreys Test},
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
* **GNU Make**

Install dependencies:

```bash
poetry install
```

Show replication commands:

```bash
make -f replication/Makefile help
```

---

## Quick start (paper replication)

Run the full paper bundle (A + B + C):

```bash
make -f replication/Makefile paper_all
```

Or run each block separately:

### (A) Empirical application #1 — Main mortgage pipeline

```bash
make -f replication/Makefile pipeline
```

### (B) Empirical application #2 — LDP / S&P grades

```bash
make -f replication/Makefile sp_grade_all
```

### (C) Simulations

```bash
make -f replication/Makefile sims_all
```

For a detailed step-by-step replication guide (data layout, expected outputs, troubleshooting),
see `replication/README.md`.

---

## Repository structure

* `src/` : main pipeline code (Empirical app #1)
* `replication/` : **replication-only Makefile** + documentation
* `data/` : raw inputs + processed datasets (see `data/README.md`)
* `artifacts/` : trained models, binning rules, figures, tables
* `reports/` : HTML validation reports
* `outputs/` : additional paper outputs (e.g., OOS backtest package)
* `ldp_application/` : LDP / S&P grade tables + plots (Empirical app #2)
* `experiments/` : simulations (Empirical app #3)

---

## Data sources

### Mortgage loan-level data (Empirical app #1)

The main empirical pipeline uses the **Freddie Mac Single-Family Loan-Level Dataset (Standard)**.

Official page:

```text
https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
```

In this repo we use **Standard quarterly files** and store them as extracted text files under:

* `data/raw/mortgage_data/historical_data_YYYYQn/`

See `data/README.md` for the exact layout and dictionary file.

Important: the dataset is provided by Freddie Mac under its own terms; users must obtain it
themselves and comply with the dataset’s license/terms. See `DATA_DISCLAIMER.md`.

### Corporate ratings data (Empirical app #2)

The S&P grades replication expects a corporate ratings export under:

* `ldp_application/data/raw/20220601_SP_Ratings_Services_Corporate.csv`

The replication Makefile builds a monthly snapshot and then generates grade tables + plots.

---

## License

* Code in this repository is released under the **MIT License** (see `LICENSE`).
* This license applies to the **code only**. It does **not** grant rights to any third-party data.

---

## Clean replication outputs

Remove all replication outputs (processed data + artifacts + reports):

```bash
make -f replication/Makefile clean_replication_outputs
```