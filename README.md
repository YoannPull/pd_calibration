# PD Calibration – End-to-End Pipeline

A clean, reproducible pipeline to build default labels (Freddie Mac style), impute, bin/WOE, train a classifier with drift-aware model selection, and apply it to new data — all orchestrated with a Makefile.

---

## Overview

```

raw quartered files
│
▼
[labels]  → window=Tm/quarter=YYYYQn/data.parquet
└→ window=Tm/pooled.parquet (+ oos.parquet if configured)
│
▼
[impute]  → imputed/train.parquet + validation.parquet  (+ imputer.joblib)
│
▼
[binning] → binned/train.parquet + validation.parquet   (+ binning.joblib, IV report)
│
▼
[train]   → artifacts/model/model_best.joblib + reports
│
▼
[apply]   → scored/test_scored.parquet

```

- Labels are built per quarter; a pooled file concatenates earlier quarters (optionally defined by `pooled_until`).
- Imputation fits on pooled and validates on a held-out quarter.
- Binning learns optimal bins/WOE on train and applies to validation.
- Training selects the least-drift model (PSI/JS) under an AUC constraint.
- Apply runs the saved model on a new (already binned) dataset.

---

## Project layout (outputs)

```

data/
└── processed/
├── labels/
│   └── window=24m/
│       ├── quarter=2019Q4/data.parquet
│       ├── quarter=2020Q1/data.parquet
│       ├── pooled.parquet
│       ├── oos.parquet                 # if pooled_until provided
│       ├── _summary.csv
│       └── _manifest.json
├── imputed/
│   ├── train.parquet
│   └── validation.parquet
├── binned/
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet                     # example target for binning_apply
└── scored/
└── test_scored.parquet

artifacts/
├── imputer/
│   └── imputer.joblib
├── binning/
│   ├── binning.joblib
│   └── binning_iv.csv
└── model/
├── model_best.joblib
├── model_reports.csv
└── model_meta.json

reports/
├── iv/
│   └── binning_iv.csv
└── model/
└── model_reports.csv

````

---

## Prerequisites

- Python (3.12 recommended)
- Poetry
- Recommended libs: `pandas`, `numpy`, `scikit-learn`, `pyarrow`, `joblib`, `yaml`

Install environment:
```bash
poetry install
````

Activate:

```bash
poetry shell
```

---

## Configuration (`config.yml`)

```yaml
data:
  root: "data/raw/mortgage_data"
  quarters: ["2019Q4","2020Q1","2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2", "2021Q3", "2021Q4",
             "2022Q1","2022Q2","2022Q3","2022Q4","2023Q1","2023Q2","2023Q3","2023Q4"]
labels:
  window_months: 24
  delinquency_threshold: 3
  liquidation_codes: ["02","03","09"]
  include_ra: true
  require_full_window: false
output:
  dir: "data/processed/labels"
  make_pooled: true
  # pooled_until: "2022Q4"   # optional split boundary for pooled vs oos
  # format: parquet           # default
```

Raw data layout expected:
Each quarter under `data/raw/mortgage_data/historical_data_<YYYYQn>/` with files:

* `historical_data_<YYYYQn>.txt` (origination)
* `historical_data_time_<YYYYQn>.txt` (performance)

---

## Quickstart (Makefile)

### 1) Build labels (parallel, with pooled/oos)

```bash
# Minimal
make labels

# With options
make labels LABELS_POOLED=1 LABELS_WORKERS=6 LABELS_POOLED_UNTIL=2022Q4
```

Outputs: `data/processed/labels/window=24m/...`

### 2) Impute (train = pooled, validation = chosen quarter)

```bash
# Default VAL_QUARTER=2022Q4
make impute

# Choose a specific validation quarter
make impute VAL_QUARTER=2023Q2
```

### 3) Fit binning and apply to validation

```bash
make binning_fit
```

Apply the saved binning to any imputed dataset:

```bash
make binning_apply BINNING_APPLY_DATA=data/processed/imputed/test.parquet \
                   BINNING_APPLY_OUT=data/processed/binned/test.parquet
```

### 4) Train model (drift-aware selection)

```bash
make model_train
# Example with calibration
make model_train MODEL_CALIBRATION=isotonic
```

### 5) Apply model to binned data

```bash
make model_apply \
  MODEL_APPLY_DATA=data/processed/binned/test.parquet \
  MODEL_OUT=data/processed/scored/test_scored.parquet
```

---

## Common workflows

* Full rebuild (labels → impute → binning → train):

  ```bash
  make pipeline VAL_QUARTER=2022Q4
  ```

* Skip labels (reuse existing labels):

  ```bash
  make pipeline SKIP_LABELS=1 VAL_QUARTER=2023Q2
  ```

* Use labels from a custom location:

  ```bash
  make impute SRC_LABELS_DIR=data/processed/labels VAL_QUARTER=2023Q1
  ```

* Debug current paths/vars:

  ```bash
  make env
  make check_labels
  ```

---

## Make targets (summary)

| Target            | Description                                                                      |
| ----------------- | -------------------------------------------------------------------------------- |
| `labels`          | Build quarterly labels (+pooled, +oos if set)                                    |
| `impute`          | Fit imputer on pooled, transform train/validation                                |
| `binning_fit`     | Fit WOE/binning on train and apply to validation                                 |
| `binning_apply`   | Apply saved binning to a custom imputed dataset                                  |
| `model_train`     | Train candidates, select least-drift model (AUC threshold), optional calibration |
| `model_apply`     | Apply saved model to binned dataset                                              |
| `pipeline`        | End-to-end (labels → impute → binning_fit → model_train)                         |
| `check_labels`    | Sanity check: required label files exist                                         |
| `env`             | Print current important variables                                                |
| `clean_data`      | Remove all derived datasets                                                      |
| `clean_artifacts` | Remove all artifacts                                                             |
| `clean_reports`   | Remove reports                                                                   |
| `clean_all`       | Remove data + artifacts + reports                                                |

---

## Tips & Troubleshooting

* “Manque: …/pooled.parquet”
  Run `make labels` first, or set `SRC_LABELS_DIR` to the folder that actually contains `window=<T>m/pooled.<fmt>`.

* `VAL_QUARTER` missing
  Ensure the quarter exists under `labels/window=<T>m/quarter=<YYYYQn>/data.<fmt>`. Rebuild labels or change `VAL_QUARTER`.

* Parquet/pyarrow issues
  Ensure `pyarrow` is installed. Period dtypes are automatically converted to end-of-month timestamps on write.

* Type warnings on dates
  Upstream parsing is robust; if you pass custom data, keep `first_payment_date` monthly or convertible to monthly.

* Performance
  Use `LABELS_WORKERS=<N>` to parallelize label building (N ≈ cores − 1). Prefer Parquet over CSV for speed and memory.

---

## Design notes

* Validation uses a held-out quarter (`VAL_QUARTER`) while train is the pooled file, following a realistic temporal split.
* Model selection balances performance and stability (low PSI/JS between OOF(train) and validation predictions), with an AUC floor.
* Artifacts are versioned by stage (`imputer/`, `binning/`, `model/`) to keep training reproducible.

---

## Roadmap

* `apply_imputer.py` + Make target to impute arbitrary datasets with the saved imputer.
* Optional: targets for scoring oos directly (`binning_apply` + `model_apply` convenience wrappers).
* Reports: richer model/feature diagnostics.

---

## License

MIT (or your preferred license).

---

## Contributing

Issues and PRs welcome. Please include:

* exact command used,
* Makefile variables (output of `make env`),
* brief description and logs.

