# PD Calibration — End-to-End Pipeline

A clean, reproducible pipeline to:
- build **default labels** per quarter (Freddie-Mac–style),
- **impute** with a persisted imputer,
- **bin/WOE** features and export IV,
- **train** multiple candidates and **select** the least-drift model (PSI/JS) under an AUC floor, with optional **probability calibration**,
- **apply** the saved model to new data (with optional **risk segmentation**),
- **report** by **vintage × risk grade** with train-time bucket PDs.

Everything is orchestrated with a **Makefile** for fixed, timestamp-free outputs.

---

## Overview

```

raw quartered files
│
▼
[labels]  → data/processed/labels/window=Tm/quarter=YYYYQn/data.parquet
└→ pooled.parquet (+ oos.parquet if configured)
│
▼
[impute]  → data/processed/imputed/{train,validation}.parquet  (+ artifacts/imputer/imputer.joblib)
│
▼
[binning] → data/processed/binned/{train,validation}.parquet   (+ artifacts/binning/binning.joblib, IV report)
│
▼
[train]   → artifacts/model/model_best.joblib + model_reports.csv + model_meta.json + risk_buckets.json
│
▼
[apply]   → data/processed/scored/test_scored.parquet
│
▼
[results] → reports/results/by_vintage_grades.{csv|parquet}

```

**Key ideas**
- **Train** on pooled historical data; **validate** on a held-out quarter (or OOS).
- **Model selection** minimizes score drift (PSI/JS) between OOF(train) and validation predictions, subject to `min_auc`.
- **Segmentation**: risk buckets built **once** from **OOF(train)** of the selected (non-calibrated) model → stable deployment.
- **Results**: by-vintage grade tables with **class PD from train**, observed default rates, and bounds.

---

## Project layout (outputs)

```

data/
└── processed/
├── labels/
│   └── window=24m/
│       ├── quarter=2020Q4/data.parquet
│       ├── quarter=2021Q1/data.parquet
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
│   └── test.parquet                    # optional target for binning_apply
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
├── model_meta.json
└── risk_buckets.json

reports/
├── iv/
│   └── binning_iv.csv
├── model/
│   └── model_reports.csv
└── results/
└── by_vintage_grades.csv               # from results.py

````

---

## Prerequisites

- Python 3.11+ (3.12 recommended)
- [Poetry](https://python-poetry.org/)
- Recommended libs (handled by Poetry): `pandas`, `numpy`, `scikit-learn`, `pyarrow`, `joblib`, `yaml`

Install:

```bash
poetry install
````

Activate:

```bash
poetry shell
```

---

## Raw data expectation

Under `data/raw/mortgage_data/`, one folder per quarter:

```
data/raw/mortgage_data/
└── historical_data_<YYYYQn>/
    ├── historical_data_<YYYYQn>.txt         # origination
    └── historical_data_time_<YYYYQn>.txt    # performance
```

---

## Configuration (`config.yml`)

```yaml
data:
  root: "data/raw/mortgage_data"
  quarters: ["2019Q4","2020Q1","2020Q2","2020Q3","2020Q4",
             "2021Q1","2021Q2","2021Q3","2021Q4",
             "2022Q1","2022Q2","2022Q3","2022Q4",
             "2023Q1","2023Q2","2023Q3","2023Q4"]
labels:
  window_months: 24
  delinquency_threshold: 3
  liquidation_codes: ["02","03","09"]
  include_ra: true
  require_full_window: false
output:
  dir: "data/processed/labels"
  make_pooled: true
  # pooled_until: "2022Q4"      # optional: defines pooled vs oos split
  # format: parquet
```

---

## Makefile — main targets

```text
labels          Build quarterly labels (+pooled, +oos if set)
impute          Fit imputer on pooled, transform train/validation
binning_fit     Fit WOE/binning on train, apply to validation (+IV report)
binning_apply   Apply saved binning to a custom imputed dataset
model_train     Train candidates, select least-drift model, optional calibration, save risk_buckets.json
model_apply     Apply saved model to binned dataset (proba only)
model_apply_segment  Apply model + risk segmentation (requires buckets)
pipeline        labels → impute → binning_fit → model_train
check_labels    Sanity check for required label files
env             Print current important variables
clean_*         Remove derived datasets / artifacts / reports
```

### Common variables

* Labels:

  * `LABELS_CONFIG=config.yml`
  * `LABELS_POOLED=1`
  * `LABELS_WORKERS=<N>`
  * `LABELS_POOLED_UNTIL=YYYYQn`
  * `LABELS_WINDOW=24`, `LABELS_FORMAT=parquet`

* Imputation & splits:

  * `VAL_QUARTER=2022Q4`
  * `IMPUTE_VAL_SOURCE=quarter|oos`
  * `SRC_LABELS_DIR=data/processed/labels`

* Binning:

  * `BINNING_N_BINS=10`, `BINNING_OUTPUT=woe|bin_index|both`

* Model:

  * `MODEL_MIN_AUC=0.60`, `MODEL_DRIFT_BINS=10`
  * `MODEL_CALIBRATION=none|sigmoid|isotonic`
  * `MODEL_FEATURES="woe__credit_score,woe__original_dti"`
  * `RISK_BINS=10` (number of risk grades)

* Apply segmentation:

  * `BUCKETS_PATH=artifacts/model/risk_buckets.json`
  * `SEGMENT_COL=risk_bucket`

---

## Step-by-step (recommended)

### 1) Build labels (parallel, with pooled/OOS)

```bash
make labels
# or with options:
make labels LABELS_POOLED=1 LABELS_WORKERS=6 LABELS_POOLED_UNTIL=2022Q4
```

Outputs in `data/processed/labels/window=<T>m/` (per-quarter files + `pooled.parquet`, optional `oos.parquet`).

### 2) Impute (train = pooled, validation = quarter or OOS)

```bash
# default VAL_QUARTER=2022Q4
make impute

# choose validation quarter
make impute VAL_QUARTER=2023Q2

# or validate on OOS (requires LABELS_POOLED_UNTIL in labels step)
make impute IMPUTE_VAL_SOURCE=oos
```

### 3) Binning / WOE

```bash
make binning_fit
# apply saved binning to any imputed dataset (optional)
make binning_apply BINNING_APPLY_DATA=data/processed/imputed/test.parquet \
                   BINNING_APPLY_OUT=data/processed/binned/test.parquet
```

### 4) Train (drift-aware selection + risk buckets)

```bash
make model_train
# with calibration and custom risk grades
make model_train MODEL_CALIBRATION=isotonic RISK_BINS=10
```

Saves:

* `artifacts/model/model_best.joblib` (with `{"model", "features", "target"}`)
* `artifacts/model/model_reports.csv`
* `artifacts/model/model_meta.json`
* `artifacts/model/risk_buckets.json` (bucket edges from OOF(train); may also include per-bucket PD on train)

### 5) Apply model (score new binned data)

```bash
# probabilities only
make model_apply \
  MODEL_APPLY_DATA=data/processed/binned/test.parquet \
  MODEL_OUT=data/processed/scored/test_scored.parquet

# probabilities + segmentation (adds SEGMENT_COL)
make model_apply_segment \
  MODEL_APPLY_DATA=data/processed/binned/test.parquet \
  MODEL_OUT=data/processed/scored/test_scored.parquet
```

If your scored file also contains the target (e.g. `default_24m`), the apply script can additionally emit **OOS metrics** (AUC, Brier, LogLoss, etc.) to a JSON file (see script help).

---

## Risk segmentation

`risk_buckets.json` is produced at train time using **OOF(train)** scores from the selected model (pre-calibration), ensuring deployment stability.

Minimal schema:

```json
{
  "edges": [0.00, 0.03, 0.05, 0.08, 0.12, 1.00],
  "n_bins": 5,
  "labels": ["G01","G02","G03","G04","G05"],           // optional
  "bucket_pd": [0.01, 0.04, 0.06, 0.09, 0.20]          // optional: class PD on train (observed)
}
```

* **edges**: ascending cutpoints; bin rule is `(edges[i], edges[i+1]]`
* **bucket_pd** (optional): observed default rate in each bucket **on train** (or OOF train). When present, reporting scripts will show it as the **class PD** assigned at training time.

---

## Vintage × grade reporting

Use `results.py` to produce a by-vintage summary per risk grade. It will:

* assign buckets to each observation based on `risk_buckets.json`,
* show **grade**, **bounds**, **count**, **defaults**, **default rate**, **class PD from train** (if provided), and **mean predicted proba** in the scored data.

### Example

```bash
poetry run python src/results.py \
  --data data/processed/scored/test_scored.parquet \
  --buckets artifacts/model/risk_buckets.json \
  --target default_24m \
  --out reports/results/by_vintage_grades.csv
```

**Output columns**

* `vintage` — e.g. `2023Q2`
* `grade` — e.g. `G01`
* `bucket_index` — 0-based index
* `lower`, `upper` — bucket bounds (probability)
* `class_pd_train` — class PD as computed in training (if available in the JSON)
* `n` — number of loans in this vintage × grade
* `n_default` — observed defaults (if `--target` supplied)
* `default_rate` — observed share of defaults (if `--target` supplied)
* `mean_proba` — average predicted probability in that cell

---

## Scripts — quick reference

* **Labels**

  * `src/make_labels.py --config config.yml --outdir data/processed/labels --pooled --pooled-until 2022Q4 --workers 6`

* **Impute**

  * `src/impute_and_save.py --train-csv <...pooled...> --validation-csv <...quarter/oos...> --target default_24m --outdir data/processed/imputed --artifacts artifacts/imputer --format parquet --use-cohort --missing-flag`

* **Binning**

  * `src/fit_binning.py --train data/processed/imputed/train.parquet --validation data/processed/imputed/validation.parquet --target default_24m --outdir data/processed/binned --artifacts artifacts/binning --n-bins 10 --output woe --format parquet`

* **Training**

  * `src/train_model.py --train data/processed/binned/train.parquet --validation data/processed/binned/validation.parquet --target default_24m --artifacts artifacts/model --min-auc 0.6 --drift-bins 10 --calibration none --risk-bins 10`

* **Apply (scoring)**

  * `src/apply_model.py --data data/processed/binned/test.parquet --model artifacts/model/model_best.joblib --out data/processed/scored/test_scored.parquet [--buckets artifacts/model/risk_buckets.json --bucket-col risk_bucket --target default_24m --metrics-out reports/model/oos_metrics.json]`

* **Results**

  * `src/results.py --data data/processed/scored/test_scored.parquet --buckets artifacts/model/risk_buckets.json --target default_24m --out reports/results/by_vintage_grades.csv`

> Run `poetry run python <script> --help` for full options.

---

## Tips & Troubleshooting

* **“Manque: …/pooled.parquet”**
  Run `make labels` first, or set `SRC_LABELS_DIR` to the folder that actually contains `window=<T>m/pooled.<fmt>`.

* **Quarter not found**
  Ensure the quarter exists under `labels/window=<T>m/quarter=<YYYYQn>/data.<fmt>`. Rebuild labels or adjust `VAL_QUARTER`.

* **PyArrow / Parquet**
  Ensure `pyarrow` is installed. Period dtypes are automatically converted to end-of-month timestamps on write.

* **Performance**
  Use `LABELS_WORKERS=<N>` to parallelize label building. Prefer Parquet over CSV.

* **Drift bins choice**
  `MODEL_DRIFT_BINS` balances sensitivity of PSI/JS. Start with 10.

---

## License

MIT (adapt as needed).

## Contributing

Issues and PRs welcome. Please include:

* the exact command used,
* `make env` output,
* relevant logs, and a brief description.
::contentReference[oaicite:0]{index=0}
```
