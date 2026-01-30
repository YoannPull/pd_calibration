# ============================================================================
# Makefile – Credit Risk Pipeline (Bank-Grade) — CLEAN
# ============================================================================
# 3 BLOCKS:
#   (A) Empirical application #1 : Main pipeline (labels -> ... -> report)
#   (B) Empirical application #2 : LDP / S&P grades (tables + plots)
#   (C) Simulations              : binom / beta-binom / drift / prior sens
# ============================================================================

.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

# ============================================================================
# 0) ENVIRONMENT
# ============================================================================
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1

# Main pipeline scripts live under src/ (so we set PYTHONPATH=src)
PY ?= PYTHONPATH=src poetry run python

# LDP runner scripts are self-contained; use plain "poetry run python" here
# (and avoid nested poetry inside those runner scripts)
PYP ?= poetry run python


# ============================================================================
# 1) DIRECTORIES
# ============================================================================
DATA_DIR      := data/processed
ARTIFACTS_DIR := artifacts
REPORTS_DIR   := reports

# Raw macro series directory (e.g., FRED downloads)
MACRO_RAW_DIR := data/raw/macro


# ============================================================================
# 2) GLOBAL CONFIG (EMPIRICAL APP #1)
# ============================================================================

# 2.1 Labels
LABELS_CONFIG  ?= config.yml
LABELS_WINDOW  ?= 12
LABELS_OUTDIR   = $(DATA_DIR)/default_labels

# 2.2 Imputation
IMPUTE_TARGET_COL    = default_$(LABELS_WINDOW)m
IMPUTE_OUT_DIR       = $(DATA_DIR)/imputed
IMPUTE_ARTIFACTS_DIR = $(ARTIFACTS_DIR)/imputer

# 2.3 Binning
BINNING_OUT_DIR       = $(DATA_DIR)/binned
BINNING_ARTIFACTS_DIR = $(ARTIFACTS_DIR)/binning_maxgini
BINNING_TRAIN         = $(IMPUTE_OUT_DIR)/train.parquet
BINNING_VAL           = $(IMPUTE_OUT_DIR)/validation.parquet

# 2.4 Modeling
MODEL_ARTIFACTS_DIR = $(ARTIFACTS_DIR)/model_from_binned
SCORED_OUT_DIR      = $(DATA_DIR)/scored
MODEL_TRAIN_DATA    = $(BINNING_OUT_DIR)/train.parquet
MODEL_VAL_DATA      = $(BINNING_OUT_DIR)/validation.parquet

# 2.5 Scored outputs (produced by train_model.py)
SCORED_TRAIN_DATA   = $(SCORED_OUT_DIR)/train_scored.parquet
SCORED_VAL_DATA     = $(SCORED_OUT_DIR)/validation_scored.parquet

# 2.6 TTC Macro output (JSON)
PD_TTC_MACRO_JSON   = $(MODEL_ARTIFACTS_DIR)/pd_ttc_macro.json

# 2.7 OOS & Reporting
OOS_LABELS_PARQUET  = $(LABELS_OUTDIR)/window=$(LABELS_WINDOW)m/oos.parquet
OOS_SCORED          = $(SCORED_OUT_DIR)/oos_scored.parquet

REPORT_HTML         = $(REPORTS_DIR)/model_validation_report.html
VINTAGE_REPORT_OOS  = $(REPORTS_DIR)/vintage_grade_oos.html
VINTAGE_REPORT_VAL  = $(REPORTS_DIR)/vintage_grade_validation.html


# ============================================================================
# 3) HELP
# ============================================================================
.PHONY: help
help:
	@echo "================================================================="
	@echo "   CREDIT RISK PIPELINE -- CLEAN (3 BLOCKS)"
	@echo "================================================================="
	@echo ""
	@echo "-----------------------------------------------------------------"
	@echo "  (A) EMPIRICAL APPLICATION #1 — MAIN PIPELINE"
	@echo "-----------------------------------------------------------------"
	@echo "    make pipeline                 : labels -> impute -> binning_fit -> model_train_final -> ttc_macro -> report"
	@echo "    make labels                   : 1) Generate default labels"
	@echo "    make impute                   : 2) Impute (fit on Train / transform all)"
	@echo "    make binning_fit              : 3) Binning (Max Gini + monotonicity)"
	@echo "    make model_train_iter         : 4) Train model (ITER: halving, large search)"
	@echo "    make model_train_final        : 4) Train model (FINAL: grid, reproducible)"
	@echo "    make report                   : 6) Generate HTML report (train+val scored)"
	@echo ""
	@echo "    make recalibrate_pd           : Type-1 recalibration (grade->PD) on rolling window"
	@echo "    make val_apply_ms             : Apply recalibrated master scale on Validation (pd_ms)"
	@echo "    make oos_apply_ms             : Apply recalibrated master scale on OOS (pd_ms)"
	@echo ""
	@echo "    make oos_score                : Score OOS sample (oos.parquet)"
	@echo "    make oos_vintage_report       : Score OOS + vintage/grade report"
	@echo "    make val_vintage_report       : Vintage/grade report on Validation"
	@echo "    make score_custom             : Score a custom dataset"
	@echo ""
	@echo "    make oos_backtest_full        : Paper-ready OOS backtest (tables+plots+LaTeX snapshot)"
	@echo "                                 : (use vars: OOSBT_OOS=..., OOSBT_BUCKET_STATS=..., OOSBT_PDK_TARGET=...)"
	@echo ""
	@echo "-----------------------------------------------------------------"
	@echo "  (B) EMPIRICAL APPLICATION #2 — LDP / S&P GRADES (tables + plots)"
	@echo "-----------------------------------------------------------------"
	@echo "    make sp_grade_tables          : Build S&P grade tables (CSV per year + combined CSV)"
	@echo "    make sp_grade_plots           : Build plots from combined CSV (timeseries)"
	@echo "    make sp_grade_all             : tables + plots"
	@echo ""
	@echo "-----------------------------------------------------------------"
	@echo "  (C) SIMULATIONS"
	@echo "-----------------------------------------------------------------"
	@echo "    make binom_all                : sim+plots+tables (binomial)"
	@echo "    make beta_binom_all           : sim+plots+tables (beta-binomial)"
	@echo "    make temporal_drift_all       : sim+plots+tables (temporal drift)"
	@echo "    make prior_sens_all           : sim+plots+tables (prior sensitivity)"
	@echo "    make sims_all                 : run everything"
	@echo ""
	@echo "-----------------------------------------------------------------"
	@echo "  CLEAN"
	@echo "-----------------------------------------------------------------"
	@echo "    make clean_all                : Remove all processed outputs / artifacts / reports"
	@echo "================================================================="


# ============================================================================
# (A) EMPIRICAL APPLICATION #1 — MAIN PIPELINE (1 -> 6)
# ============================================================================

# ----------------------------------------------------------------------------
# A.1) LABELS
# ----------------------------------------------------------------------------
.PHONY: labels
labels:
	@echo "\n[1/6] GENERATING LABELS..."
	$(PY) src/make_labels.py --config $(LABELS_CONFIG) --workers 12

# ----------------------------------------------------------------------------
# A.2) IMPUTATION (Anti-leakage)
# ----------------------------------------------------------------------------
.PHONY: impute
impute:
	@echo "\n[2/6] IMPUTATION (fit on Train / transform all)..."
	$(PY) src/impute_and_save.py \
		--labels-window-dir $(LABELS_OUTDIR)/window=$(LABELS_WINDOW)m \
		--target $(IMPUTE_TARGET_COL) \
		--outdir $(IMPUTE_OUT_DIR) \
		--artifacts $(IMPUTE_ARTIFACTS_DIR) \
		--use-splits \
		--fail-on-nan

# ----------------------------------------------------------------------------
# A.3) BINNING (Monotone)
# ----------------------------------------------------------------------------
.PHONY: binning_fit
binning_fit:
	@echo "\n[3/6] BINNING (Max Gini + monotonicity check)..."
	$(PY) src/fit_binning.py \
		--train $(BINNING_TRAIN) \
		--validation $(BINNING_VAL) \
		--target $(IMPUTE_TARGET_COL) \
		--outdir $(BINNING_OUT_DIR) \
		--artifacts $(BINNING_ARTIFACTS_DIR) \
		--max-bins-num 10 \
		--min-bin-size-num 300 \
		--n-quantiles-num 50

# ----------------------------------------------------------------------------
# A.4) MODEL TRAINING (Master Scale + scoring Train/Validation)
# ----------------------------------------------------------------------------
# Hyperparameter search
SEARCH          ?= halving
# C_MIN_EXP       ?= -12
# C_MAX_EXP       ?= 6
# C_NUM           ?= 120
C_MIN_EXP       ?= 1.8
C_MAX_EXP       ?= 2.9
C_NUM           ?= 25
HALVING_FACTOR  ?= 3
SEARCH_VERBOSE  ?= 0

# Logistic regression training
LR_SOLVER       ?= lbfgs
LR_MAX_ITER     ?= 4000
COEF_STATS      ?= none

# Time-aware CV / calibration split
CV_SCHEME       ?= time
CAL_SPLIT       ?= time_last
CAL_SIZE        ?= 0.40
CV_TIME_COL     ?= vintage
CV_TIME_FREQ    ?= Q

# Calibration method
CALIBRATION     ?= isotonic

# Rolling windows
GRID_YEARS      ?= 12
TTC_YEARS       ?= 12
GRID_TIME_COL   ?= vintage
GRID_TIME_FREQ  ?= Q

# Master scale (risk buckets)
N_BUCKETS       ?= 10
N_BUCKETS_CANDIDATES ?=
MIN_BUCKET_COUNT ?= 300
MIN_BUCKET_BAD   ?= 5

# Switches
NO_INTERACTIONS ?= 0
TIMING          ?= 1

# Flags derived from switches
ifeq ($(NO_INTERACTIONS),1)
  NO_INTERACTIONS_FLAG := --no-interactions
else
  NO_INTERACTIONS_FLAG :=
endif

ifeq ($(TIMING),1)
  TIMING_FLAG := --timing
else
  TIMING_FLAG :=
endif

# Optional “auto bucket selection” constraints
ifneq ($(strip $(N_BUCKETS_CANDIDATES)),)
  N_BUCKETS_CANDIDATES_FLAG := --n-buckets-candidates "$(N_BUCKETS_CANDIDATES)" \
                               --min-bucket-count $(MIN_BUCKET_COUNT) \
                               --min-bucket-bad $(MIN_BUCKET_BAD)
else
  N_BUCKETS_CANDIDATES_FLAG :=
endif

.PHONY: model_train_iter
model_train_iter:
	@echo "\n[4/6] MODEL TRAIN (ITER) — halving search, large space..."
	$(PY) src/train_model.py \
		--train $(MODEL_TRAIN_DATA) \
		--validation $(MODEL_VAL_DATA) \
		--target $(IMPUTE_TARGET_COL) \
		--artifacts $(MODEL_ARTIFACTS_DIR) \
		--cv-folds 5 \
		--corr-threshold 0.85 \
		--calibration $(CALIBRATION) \
		--search halving \
		--c-min-exp $(C_MIN_EXP) \
		--c-max-exp $(C_MAX_EXP) \
		--c-num $(C_NUM) \
		--halving-factor $(HALVING_FACTOR) \
		--search-verbose $(SEARCH_VERBOSE) \
		--lr-solver $(LR_SOLVER) \
		--lr-max-iter $(LR_MAX_ITER) \
		--coef-stats $(COEF_STATS) \
		--cv-scheme $(CV_SCHEME) \
		--calibration-split $(CAL_SPLIT) \
		--calibration-size $(CAL_SIZE) \
		--cv-time-col $(CV_TIME_COL) \
		--cv-time-freq $(CV_TIME_FREQ) \
		--n-buckets $(N_BUCKETS) \
		$(N_BUCKETS_CANDIDATES_FLAG) \
		--grid-window-years $(GRID_YEARS) \
		--grid-time-col $(GRID_TIME_COL) \
		--grid-time-freq $(GRID_TIME_FREQ) \
		--ttc-window-years $(TTC_YEARS) \
		--scored-outdir $(SCORED_OUT_DIR) \
		--ttc-mode train \
		$(NO_INTERACTIONS_FLAG) \
		$(TIMING_FLAG)

.PHONY: model_train_final
model_train_final:
	@echo "\n[4/6] MODEL TRAIN (FINAL) — grid search, reproducible..."
	$(PY) src/train_model.py \
		--train $(MODEL_TRAIN_DATA) \
		--validation $(MODEL_VAL_DATA) \
		--target $(IMPUTE_TARGET_COL) \
		--artifacts $(MODEL_ARTIFACTS_DIR) \
		--cv-folds 5 \
		--corr-threshold 0.85 \
		--calibration $(CALIBRATION) \
		--search grid \
		--c-min-exp $(C_MIN_EXP) \
		--c-max-exp $(C_MAX_EXP) \
		--c-num $(C_NUM) \
		--lr-solver $(LR_SOLVER) \
		--lr-max-iter $(LR_MAX_ITER) \
		--coef-stats $(COEF_STATS) \
		--cv-scheme $(CV_SCHEME) \
		--calibration-split $(CAL_SPLIT) \
		--calibration-size $(CAL_SIZE) \
		--cv-time-col $(CV_TIME_COL) \
		--cv-time-freq $(CV_TIME_FREQ) \
		--n-buckets $(N_BUCKETS) \
		$(N_BUCKETS_CANDIDATES_FLAG) \
		--grid-window-years $(GRID_YEARS) \
		--grid-time-col $(GRID_TIME_COL) \
		--grid-time-freq $(GRID_TIME_FREQ) \
		--ttc-window-years $(TTC_YEARS) \
		--scored-outdir $(SCORED_OUT_DIR) \
		--ttc-mode train \
		$(NO_INTERACTIONS_FLAG) \
		$(TIMING_FLAG)

# ----------------------------------------------------------------------------
# A.5) TTC MACRO PD ESTIMATION
# ----------------------------------------------------------------------------
.PHONY: ttc_macro
ttc_macro:
	@echo "\n[5/6] ESTIMATING TTC MACRO PD PER GRADE (pd_ttc_macro)..."
	@test -f "$(SCORED_TRAIN_DATA)" || (echo "[ERR] Missing: $(SCORED_TRAIN_DATA)"; exit 1)
	@test -f "$(SCORED_VAL_DATA)"   || (echo "[ERR] Missing: $(SCORED_VAL_DATA)"; exit 1)
	$(PY) src/estimate_ttc_macro.py \
		--train-scored $(SCORED_TRAIN_DATA) \
		--val-scored $(SCORED_VAL_DATA) \
		--macro-dir $(MACRO_RAW_DIR) \
		--target $(IMPUTE_TARGET_COL) \
		--time-col vintage \
		--out-json $(PD_TTC_MACRO_JSON)
	@echo "✔ TTC macro PD written to: $(PD_TTC_MACRO_JSON)"

# ----------------------------------------------------------------------------
# A.6) REPORTING
# ----------------------------------------------------------------------------
.PHONY: report
report:
	@echo "\n[6/6] GENERATING GLOBAL REPORT (Train + Validation scored)..."
	$(PY) src/generate_report.py \
		--train $(SCORED_TRAIN_DATA) \
		--validation $(SCORED_VAL_DATA) \
		--out $(REPORT_HTML) \
		--target $(IMPUTE_TARGET_COL) \
		--score score_ttc --pd pd --grade grade \
		--model $(MODEL_ARTIFACTS_DIR)/model_best.joblib
	@echo "✔ Report generated: $(REPORT_HTML)"
	@open $(REPORT_HTML) 2>/dev/null || true

# ----------------------------------------------------------------------------
# A.X) FULL PIPELINE (shortcut)
# ----------------------------------------------------------------------------
.PHONY: pipeline
pipeline: labels impute binning_fit model_train_final ttc_macro report
	@echo "\n-------------------------------------------------------"
	@echo "✔ VALIDATION PIPELINE COMPLETED."
	@echo "-------------------------------------------------------"


# ============================================================================
# (A bis) RECALIBRATION / MASTER SCALE (APP #1)
# ============================================================================
RECAL_METHOD ?= mean
RECAL_AGG    ?= time_mean
RECAL_YEARS  ?= 5

BUCKET_STATS_RECAL := $(MODEL_ARTIFACTS_DIR)/bucket_stats_recalibrated.json

.PHONY: recalibrate_pd
recalibrate_pd:
	@echo "\n[REC] Recalibrating grade->PD (method=$(RECAL_METHOD), agg=$(RECAL_AGG), years=$(RECAL_YEARS))..."
	@test -f "$(SCORED_TRAIN_DATA)" || (echo "[ERR] Missing: $(SCORED_TRAIN_DATA)"; exit 1)
	@test -f "$(SCORED_VAL_DATA)"   || (echo "[ERR] Missing: $(SCORED_VAL_DATA)"; exit 1)
	$(PY) src/recalibrate_master_scale.py \
		--scored $(SCORED_TRAIN_DATA) \
		--scored $(SCORED_VAL_DATA) \
		--target $(IMPUTE_TARGET_COL) \
		--grade-col grade \
		--time-col vintage \
		--time-freq Q \
		--method $(RECAL_METHOD) \
		--aggregation $(RECAL_AGG) \
		--window-years $(RECAL_YEARS) \
		--out-json $(BUCKET_STATS_RECAL)
	@echo "✔ Recalibrated bucket stats written to: $(BUCKET_STATS_RECAL)"

SCORED_VAL_MS := $(SCORED_OUT_DIR)/validation_scored_ms.parquet
OOS_SCORED_MS := $(SCORED_OUT_DIR)/oos_scored_ms.parquet

.PHONY: val_apply_ms
val_apply_ms:
	@echo "\n[MS] Applying recalibrated master scale to Validation..."
	@test -f "$(SCORED_VAL_DATA)"      || (echo "[ERR] Missing: $(SCORED_VAL_DATA)"; exit 1)
	@test -f "$(BUCKET_STATS_RECAL)"   || (echo "[ERR] Missing: $(BUCKET_STATS_RECAL)"; exit 1)
	$(PY) src/apply_master_scale.py \
		--in-scored $(SCORED_VAL_DATA) \
		--out $(SCORED_VAL_MS) \
		--bucket-stats $(BUCKET_STATS_RECAL) \
		--grade-col grade \
		--pd-col-out pd_ms
	@echo "✔ Validation enriched: $(SCORED_VAL_MS)"

.PHONY: oos_apply_ms
oos_apply_ms: oos_score
	@echo "\n[MS] Applying recalibrated master scale to OOS..."
	@test -f "$(OOS_SCORED)"          || (echo "[ERR] Missing: $(OOS_SCORED)"; exit 1)
	@test -f "$(BUCKET_STATS_RECAL)" || (echo "[ERR] Missing: $(BUCKET_STATS_RECAL)"; exit 1)
	$(PY) src/apply_master_scale.py \
		--in-scored $(OOS_SCORED) \
		--out $(OOS_SCORED_MS) \
		--bucket-stats $(BUCKET_STATS_RECAL) \
		--grade-col grade \
		--pd-col-out pd_ms
	@echo "✔ OOS enriched: $(OOS_SCORED_MS)"


# ============================================================================
# (A ter) MANUAL MODULES (APP #1)
# ============================================================================
.PHONY: oos_score
oos_score:
	@echo "\n[MANUAL] SCORING OOS SAMPLE..."
	@test -f "$(OOS_LABELS_PARQUET)" || (echo "[ERR] Missing: $(OOS_LABELS_PARQUET)"; exit 1)
	$(PY) src/apply_model.py \
		--data $(OOS_LABELS_PARQUET) \
		--out $(OOS_SCORED) \
		--imputer $(IMPUTE_ARTIFACTS_DIR)/imputer.joblib \
		--bins $(BINNING_ARTIFACTS_DIR)/bins.json \
		--model $(MODEL_ARTIFACTS_DIR)/model_best.joblib \
		--buckets $(MODEL_ARTIFACTS_DIR)/risk_buckets.json \
		--target $(IMPUTE_TARGET_COL) \
		--id-col loan_sequence_number
	@echo "✔ OOS scoring done: $(OOS_SCORED)"

.PHONY: oos_vintage_report
oos_vintage_report: oos_score
	@echo "\n[MANUAL] VINTAGE / GRADE REPORT (OOS)..."
	$(PY) src/generate_vintage_grade_report.py \
		--data $(OOS_SCORED) \
		--out $(VINTAGE_REPORT_OOS) \
		--vintage-col vintage \
		--grade-col grade \
		--pd-col pd \
		--target $(IMPUTE_TARGET_COL) \
		--sample-name "OOS" \
		--bucket-stats $(MODEL_ARTIFACTS_DIR)/bucket_stats.json
	@echo "✔ OOS report: $(VINTAGE_REPORT_OOS)"
	@open $(VINTAGE_REPORT_OOS) 2>/dev/null || true

.PHONY: val_vintage_report
val_vintage_report:
	@echo "\n[MANUAL] VINTAGE / GRADE REPORT (Validation)..."
	@test -f "$(SCORED_VAL_DATA)" || (echo "[ERR] Missing: $(SCORED_VAL_DATA)"; exit 1)
	$(PY) src/generate_vintage_grade_report.py \
		--data $(SCORED_VAL_DATA) \
		--out $(VINTAGE_REPORT_VAL) \
		--vintage-col vintage \
		--grade-col grade \
		--pd-col pd \
		--target $(IMPUTE_TARGET_COL) \
		--sample-name "Validation" \
		--bucket-stats $(MODEL_ARTIFACTS_DIR)/bucket_stats.json
	@echo "✔ Validation report: $(VINTAGE_REPORT_VAL)"
	@open $(VINTAGE_REPORT_VAL) 2>/dev/null || true

CUSTOM_DATA   ?= $(OOS_LABELS_PARQUET)
CUSTOM_OUT    ?= $(SCORED_OUT_DIR)/custom_scored.parquet
CUSTOM_TARGET ?= $(IMPUTE_TARGET_COL)

.PHONY: score_custom
score_custom:
	@echo "\n[MANUAL] Scoring custom dataset: $(CUSTOM_DATA)"
	@test -f "$(CUSTOM_DATA)" || (echo "[ERR] Missing: $(CUSTOM_DATA)"; exit 1)
	$(PY) src/apply_model.py \
		--data $(CUSTOM_DATA) \
		--out $(CUSTOM_OUT) \
		--imputer $(IMPUTE_ARTIFACTS_DIR)/imputer.joblib \
		--bins $(BINNING_ARTIFACTS_DIR)/bins.json \
		--model $(MODEL_ARTIFACTS_DIR)/model_best.joblib \
		--buckets $(MODEL_ARTIFACTS_DIR)/risk_buckets.json \
		--target $(CUSTOM_TARGET) \
		--id-col loan_sequence_number
	@echo "✔ Custom scoring output: $(CUSTOM_OUT)"


# ============================================================================
# (A quater) OOS BACKTEST — FULL PAPER-READY (tables + plots + LaTeX snapshot)
# ============================================================================
OOSBT_OUTDIR        ?= outputs/oos_backtest
OOSBT_OOS           ?= $(OOS_SCORED)
OOSBT_BUCKET_STATS  ?= $(MODEL_ARTIFACTS_DIR)/bucket_stats.json
OOSBT_BUCKET_SECTION ?= train

OOSBT_PDK_TARGET    ?= pd_ttc
OOSBT_CONF_LEVEL    ?= 0.95
OOSBT_TL_MAIN_MODE  ?= decision
OOSBT_SAVE_PDF      ?= 1
OOSBT_FOCUS_YEAR    ?=

OOSBT_PAPER_SNAPSHOT ?= 2023Q4
OOSBT_PAPER_UNITS    ?= bps
OOSBT_PAPER_ALPHA    ?= 0.05

.PHONY: oos_backtest_full
oos_backtest_full: oos_score
	@echo "\n[OOS-BACKTEST] FULL PAPER-READY BACKTEST..."
	@test -f "$(OOSBT_OOS)"          || (echo "[ERR] Missing: $(OOSBT_OOS)"; exit 1)
	@test -f "$(OOSBT_BUCKET_STATS)" || (echo "[ERR] Missing: $(OOSBT_BUCKET_STATS)"; exit 1)
	@if [ "$(OOSBT_SAVE_PDF)" = "1" ]; then PDF_FLAG="--save-pdf"; else PDF_FLAG="--no-pdf"; fi; \
	FOCUS_FLAG=""; \
	if [ -n "$(strip $(OOSBT_FOCUS_YEAR))" ]; then FOCUS_FLAG="--focus-year $(OOSBT_FOCUS_YEAR)"; fi; \
	$(PY) src/run_oos_backtest.py \
		--oos "$(OOSBT_OOS)" \
		--bucket-stats "$(OOSBT_BUCKET_STATS)" \
		--bucket-section "$(OOSBT_BUCKET_SECTION)" \
		--out "$(OOSBT_OUTDIR)" \
		--pdk-target "$(OOSBT_PDK_TARGET)" \
		--conf-level $(OOSBT_CONF_LEVEL) \
		--tl-main-mode "$(OOSBT_TL_MAIN_MODE)" \
		--paper-snapshot "$(OOSBT_PAPER_SNAPSHOT)" \
		--paper-units "$(OOSBT_PAPER_UNITS)" \
		--paper-alpha $(OOSBT_PAPER_ALPHA) \
		$$PDF_FLAG \
		$$FOCUS_FLAG
	@echo "✔ OOS backtest written to: $(OOSBT_OUTDIR)"


# ============================================================================
# (B) EMPIRICAL APPLICATION #2 — LDP / S&P GRADES (tables + plots)
#   + NEW STEP: build the monthly snapshot dataset (CSV-only)
# ============================================================================

SPG_ROOT   ?= ldp_application

# -----------------------------
# Build monthly snapshot (CSV only)
# -----------------------------
SP_RAW_CSV ?= $(SPG_ROOT)/data/raw/20220601_SP_Ratings_Services_Corporate.csv
SP_SNAP_CSV ?= $(SPG_ROOT)/data/processed/sp_corporate_monthly.csv
SP_SNAP_SCRIPT ?= $(SPG_ROOT)/process_sp_base.py

SP_SNAP_START ?= 2010-01-01
SP_SNAP_END   ?= 2021-07-01

.PHONY: sp_snapshot

sp_snapshot:
	@echo "\n[SP-SNAPSHOT] Build monthly snapshot (CSV-only, corporate.xlsx-like schema)..."
	@test -f "$(SP_RAW_CSV)" || (echo "[ERR] Missing raw CSV: $(SP_RAW_CSV)"; exit 1)
	@test -f "$(SP_SNAP_SCRIPT)" || (echo "[ERR] Missing script: $(SP_SNAP_SCRIPT)"; exit 1)
	$(PYP) "$(SP_SNAP_SCRIPT)" \
		--raw "$(SP_RAW_CSV)" \
		--out_csv "$(SP_SNAP_CSV)" \
		--start "$(SP_SNAP_START)" \
		--end   "$(SP_SNAP_END)"
	@echo "✔ Snapshot written:"
	@echo "  - $(SP_SNAP_CSV)"


# -----------------------------
# S&P grades (tables + plots)
# -----------------------------
# Use CSV snapshot as the default input for the grade pipeline
SPG_FILE   ?= $(SP_SNAP_CSV)
SPG_OUTDIR ?= $(SPG_ROOT)/outputs/sp_grade_is_oos

SPG_AGENCY ?= Standard & Poor's Ratings Services
SPG_HORIZON_MONTHS ?= 12
SPG_CONF_LEVEL ?= 0.95

SPG_IS_START  ?= 2010
SPG_IS_END    ?= 2018
SPG_OOS_START ?= 2010
SPG_OOS_END   ?= 2020

SPG_TTC_SOURCE    ?= sp2012
SPG_TTC_ESTIMATOR ?= pooled
SPG_DROP_NO_TTC   ?= 1

# New: year handling in snapshot
#   - If your snapshot contains the custom July–June "year" column (recommended): 1
#   - If you want calendar year from rating_action_date.year: 0
SPG_PREFER_YEAR_COLUMN ?= 1

SPG_PMAX        ?=
SPG_MIN_TOTAL_N ?= 1

SPG_TABLES_CSV ?= $(SPG_OUTDIR)/sp_grade_tables_$(SPG_OOS_START)_$(SPG_OOS_END).csv

.PHONY: sp_grade_tables sp_grade_plots sp_grade_all

# Ensure tables always run on the freshest snapshot
sp_grade_tables: sp_snapshot
	@echo "\n[SP-GRADE] Tables (TTC_SOURCE=$(SPG_TTC_SOURCE))..."
	@test -f "$(SPG_FILE)" || (echo "[ERR] Missing: $(SPG_FILE)"; exit 1)
	@if [ "$(SPG_DROP_NO_TTC)" = "1" ]; then DROP_FLAG="--drop-grades-without-ttc"; else DROP_FLAG=""; fi; \
	if [ "$(SPG_PREFER_YEAR_COLUMN)" = "1" ]; then YEAR_FLAG="--prefer-year-column"; else YEAR_FLAG="--no-prefer-year-column"; fi; \
	$(PYP) $(SPG_ROOT)/run_sp_grade_tables.py --no-poetry \
		--file "$(SPG_FILE)" \
		--outdir "$(SPG_OUTDIR)" \
		--agency "$(SPG_AGENCY)" \
		--horizon-months $(SPG_HORIZON_MONTHS) \
		--confidence-level $(SPG_CONF_LEVEL) \
		--is-start-year $(SPG_IS_START) \
		--is-end-year $(SPG_IS_END) \
		--oos-start-year $(SPG_OOS_START) \
		--oos-end-year $(SPG_OOS_END) \
		--ttc-source $(SPG_TTC_SOURCE) \
		--ttc-estimator $(SPG_TTC_ESTIMATOR) \
		$$YEAR_FLAG $$DROP_FLAG
	@echo "✔ Tables written to: $(SPG_OUTDIR)"
	@echo "  - sp_grade_table_YYYY.csv (one per year)"
	@echo "  - sp_grade_tables_*.csv (combined)"

sp_grade_plots:
	@echo "\n[SP-GRADE] Plots..."
	@TABLES="$(SPG_TABLES_CSV)"; \
	if [ ! -f "$$TABLES" ]; then \
		TABLES=$$(ls -1t "$(SPG_OUTDIR)"/sp_grade_tables_*.csv 2>/dev/null | head -n 1 || true); \
	fi; \
	test -f "$$TABLES" || (echo "[ERR] Missing tables CSV in $(SPG_OUTDIR). Run 'make sp_grade_tables' first."; exit 1); \
	echo "Using tables CSV: $$TABLES"; \
	PMAX_FLAG=""; \
	if [ -n "$(strip $(SPG_PMAX))" ]; then PMAX_FLAG="--p-max $(SPG_PMAX)"; fi; \
	$(PYP) $(SPG_ROOT)/run_sp_grade_plots.py --no-poetry \
		--tables-csv "$$TABLES" \
		--outdir "$(SPG_OUTDIR)/plots_timeseries" \
		--confidence-level $(SPG_CONF_LEVEL) \
		--min-total-n $(SPG_MIN_TOTAL_N) \
		$$PMAX_FLAG
	@echo "✔ Plots written to: $(SPG_OUTDIR)/plots_timeseries"

sp_grade_all: sp_grade_tables sp_grade_plots



# ============================================================================
# (C) SIMULATIONS
# ============================================================================
.PHONY: binom_sim binom_plots binom_tables binom_all
binom_sim:
	poetry run python -m experiments.binom_coverage.sim_binom
binom_plots:
	poetry run python -m experiments.binom_coverage.plot_binom
binom_tables:
	poetry run python -m experiments.binom_coverage.make_table_binom
binom_all: binom_sim binom_plots binom_tables

.PHONY: beta_binom_sim beta_binom_plots beta_binom_tables beta_binom_all
beta_binom_sim:
	poetry run python -m experiments.beta_binom_jeffreys.sim_beta_binom
beta_binom_plots:
	poetry run python -m experiments.beta_binom_jeffreys.plot_beta_binom
beta_binom_tables:
	poetry run python -m experiments.beta_binom_jeffreys.make_table_beta_binom
beta_binom_all: beta_binom_sim beta_binom_plots beta_binom_tables
 
.PHONY: temporal_drift_sim temporal_drift_plots temporal_drift_tables temporal_drift_all
temporal_drift_sim:
	poetry run python -m experiments.temporal_drift.sim_temporal_drift
temporal_drift_plots:
	poetry run python -m experiments.temporal_drift.plot_temporal_drift
temporal_drift_tables:
	poetry run python -m experiments.temporal_drift.make_table_temporal_drift
temporal_drift_all: temporal_drift_sim temporal_drift_plots temporal_drift_tables

.PHONY: prior_sens_sim prior_sens_plots prior_sens_tables prior_sens_all
prior_sens_sim:
	poetry run python -m experiments.prior_sensibility.sim_prior_sensibility
prior_sens_plots:
	poetry run python -m experiments.prior_sensibility.plot_prior_sensibility
prior_sens_tables:
	poetry run python -m experiments.prior_sensibility.make_table_prior_sensibility
prior_sens_all: prior_sens_sim prior_sens_plots prior_sens_tables

.PHONY: sims_all
sims_all: beta_binom_all temporal_drift_all binom_all prior_sens_all


.PHONY: plot_all_sims
plot_all_sims: prior_sens_plots beta_binom_plots temporal_drift_plots binom_plots

# ============================================================================
# CLEANUP
# ============================================================================
.PHONY: clean_all
clean_all:
	rm -rf $(DATA_DIR) $(ARTIFACTS_DIR) $(REPORTS_DIR)
	@echo "Everything cleaned."



## TEMP ##
.PHONY: diag
diag: 
	poetry run python diagnose_vars_stages.py \
  --pooled data/processed/default_labels/window=12m/pooled.parquet \
  --imputed data/processed/imputed/train.parquet \
  --binned data/processed/binned/train.parquet \
  --bin-suffix __BIN \
  --vars amortization_type,window,relief_refinance_indicator,interest_only_indicator,super_conforming_flag \
  --outdir outputs \
  --subdir var_diagnostics_flags
