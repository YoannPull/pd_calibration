# LDP Application (S&P Grade Tables + Plots)

This folder contains the **Empirical application #2** from the Makefile: an LDP-style
analysis based on **S&P rating grades**. It builds:
1) yearly grade tables (CSV), and  
2) time-series plots computed from the combined CSV.

Everything is meant to be run from the **repository root** via the Makefile.

---

## Expected structure

From the repo root:

```

ldp_application/
run_sp_grade_tables.py
run_sp_grade_plots.py
scripts/
sp_grade_tables.py
sp_grade_plots.py
data/
raw/
data_rating_corporate.xlsx
outputs/
sp_grade_is_oos/

````

---

## Input data

Place the Excel file here:

- `ldp_application/data/raw/data_rating_corporate.xlsx`

The Makefile checks that this file exists before running.

---

## How to run (recommended)

From the repository root:

### Build grade tables
```bash
make sp_grade_tables
````

### Build plots (from the combined tables CSV)

```bash
make sp_grade_plots
```

### Run both

```bash
make sp_grade_all
```

---

## Outputs

All outputs are written under:

* `ldp_application/outputs/sp_grade_is_oos/`

Typical files:

* `sp_grade_table_YYYY.csv` (one file per year)
* `sp_grade_tables_*.csv` (combined table across years)
* `plots_timeseries/` (time-series figures)

**Plotting note:** the `sp_grade_plots` target tries to use the expected combined CSV
name (based on the requested years). If it does not exist (e.g., year clipping),
it automatically falls back to the newest `sp_grade_tables_*.csv` found in the output directory.

---

## Main Makefile parameters (you can override at runtime)

You can override parameters directly in the `make` command, for example:

```bash
make sp_grade_tables SPG_OOS_START=2015 SPG_OOS_END=2020 SPG_TTC_SOURCE=sp2012
```

Common variables:

### Agency / horizon / confidence level

* `SPG_AGENCY` (default: `Standard & Poor's Ratings Services`)
* `SPG_HORIZON_MONTHS` (default: `12`)
* `SPG_CONF_LEVEL` (default: `0.95`)

### In-sample / out-of-sample windows

* `SPG_IS_START`, `SPG_IS_END`
* `SPG_OOS_START`, `SPG_OOS_END`

### TTC options

* `SPG_TTC_SOURCE` (default: `sp2012`)
* `SPG_TTC_ESTIMATOR` (default: `pooled`)
* `SPG_DROP_NO_TTC` (default: `1`)
  If `1`, drop grades without TTC reference values.
* `SPG_INCLUDE_UNOBS` (default: `0`)
  If `1`, include unobserved grades in outputs.

### Plot options

* `SPG_PMAX` (default: empty)
  If set, caps the y-axis / PD scale in plots.
* `SPG_MIN_TOTAL_N` (default: `1`)
  Minimum number of observations required to plot a grade.

---

## Scripts

Entry points called by the Makefile:

* `run_sp_grade_tables.py` : CLI runner for table generation
* `run_sp_grade_plots.py`  : CLI runner for plotting

Core logic:

* `scripts/sp_grade_tables.py`
* `scripts/sp_grade_plots.py`

The Makefile uses:

* `poetry run python` (no nested poetry inside runner scripts)
* `--no-poetry` flag when calling the runners

---

## Troubleshooting

* If you get a “Missing: data_rating_corporate.xlsx” error, verify the file path:
  `ldp_application/data/raw/data_rating_corporate.xlsx`
* If plotting fails, run tables first:

  ```bash
  make sp_grade_tables
  make sp_grade_plots
  ```
* To regenerate from scratch:

  ```bash
  rm -rf ldp_application/outputs/sp_grade_is_oos
  make sp_grade_all
  ```
