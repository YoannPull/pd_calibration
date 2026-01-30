# ============================================================================
# replication/Makefile — Credit Risk Pipeline (REPLICATION-ONLY)
# ============================================================================
# Goal:
#   - Minimal, paper-grade replication entrypoints.
#   - Deterministic "FINAL" model training with a narrow C-grid around an anchor.
#   - Includes all paper components:
#       (A) Main mortgage pipeline (labels -> ... -> report + OOS diagnostics)
#       (B) Empirical app #2: LDP / S&P grades (tables + plots)
#       (C) Simulations (binom / beta-binom / drift / prior sens)
#
# Usage (from repo root):
#   make -f replication/Makefile help
#   make -f replication/Makefile paper_all
#
# C-grid:
#   Default anchor = best_C from my last GRID run (468.633503).
#   Override examples:
#       make -f replication/Makefile model_train_final C_ANCHOR=234.286842
#       make -f replication/Makefile model_train_final C_EXP_SPAN=0.25 C_NUM=21
# ============================================================================

.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

# ----------------------------------------------------------------------------
# 0) ENVIRONMENT
# ----------------------------------------------------------------------------
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1

# Main pipeline scripts live under src/
PY  ?= PYTHONPATH=src poetry run python
# Neutral python for make-time computations
PY0 ?= poetry run python
# For LDP runner scripts (self-contained)
PYP ?= poetry run python

# ----------------------------------------------------------------------------
# 1) DIRECTORIES
# ----------------------------------------------------------------------------
DATA_DIR      := data/processed
ARTIFACTS_DIR := artifacts
REPORTS_DIR   := reports
MACRO_RAW_DIR := data/raw/macro

# ----------------------------------------------------------------------------
# 2) GLOBAL CONFIG (A) — MAIN PIPELINE
# ----------------------------------------------------------------------------
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
SCORED_TRAIN_DATA   = $(SCORED_OUT_DIR)/train_scored.parquet
SCORED_VAL_DATA     = $(SCORED_OUT_DIR)/validation_scored.parquet

# 2.5 OOS & Reporting
OOS_LABELS_PARQUET  = $(LABELS_OUTDIR)/window=$(LABELS_WINDOW)m/oos.parquet
OOS_SCORED          = $(SCORED_OUT_DIR)/oos_scored.parquet

REPORT_HTML         = $(REPORTS_DIR)/model_validation_report.html
VINTAGE_REPORT_OOS  = $(REPORTS_DIR)/vintage_grade_oos.html
VINTAGE_REPORT_VAL  = $(REPORTS_DIR)/vintage_grade_validation.html

# ----------------------------------------------------------------------------
# 3) FIXED TRAINING SETTINGS (paper replication)
# ----------------------------------------------------------------------------
LR_SOLVER       ?= lbfgs
LR_MAX_ITER     ?= 4000
COEF_STATS      ?= none

CV_SCHEME       ?= time
CAL_SPLIT       ?= time_last
CAL_SIZE        ?= 0.40
CV_TIME_COL     ?= vintage
CV_TIME_FREQ    ?= Q

CALIBRATION     ?= isotonic

GRID_YEARS      ?= 12
TTC_YEARS       ?= 12
GRID_TIME_COL   ?= vintage
GRID_TIME_FREQ  ?= Q
N_BUCKETS       ?= 10

NO_INTERACTIONS ?= 0
TIMING          ?= 1

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

# ----------------------------------------------------------------------------
# 4) REPRODUCIBLE, FAST GRID FOR C (around an anchor)
# ----------------------------------------------------------------------------
C_ANCHOR   ?= 468.633503
C_EXP_SPAN ?= 0
C_NUM      ?= 1

C_MIN_EXP := $(shell $(PY0) -c "import math; C=float('$(C_ANCHOR)'); print(f'{math.log10(C):.6f}')")
C_MAX_EXP := $(shell $(PY0) -c "import math; C=float('$(C_ANCHOR)'); print(f'{math.log10(C):.6f}')")


# ----------------------------------------------------------------------------
# 5) OUTPUT APPLICATION A PARAMETERS (OOS BACKTEST)
# ----------------------------------------------------------------------------
OOSBT_OUTDIR         ?= outputs/oos_backtest
OOSBT_OOS            ?= $(OOS_SCORED)
OOSBT_BUCKET_STATS   ?= $(MODEL_ARTIFACTS_DIR)/bucket_stats.json
OOSBT_BUCKET_SECTION ?= train

OOSBT_PDK_TARGET     ?= pd_ttc
OOSBT_CONF_LEVEL     ?= 0.95
OOSBT_TL_MAIN_MODE   ?= decision
OOSBT_SAVE_PDF       ?= 1
OOSBT_FOCUS_YEAR     ?=

OOSBT_PAPER_SNAPSHOT ?= 2023Q4
OOSBT_PAPER_UNITS    ?= bps
OOSBT_PAPER_ALPHA    ?= 0.05

# ----------------------------------------------------------------------------
# 6) HELP
# ----------------------------------------------------------------------------
.PHONY: help
help:
	@echo "=============================================================="
	@echo "  replication/Makefile — REPLICATION-ONLY (paper entrypoints)"
	@echo "=============================================================="
	@echo ""
	@echo "Paper bundles:"
	@echo "  make -f replication/Makefile paper_all"
	@echo "      (A) pipeline  +  (B) sp_grade_all  +  (C) sims_all"
	@echo ""
	@echo "(A) Main pipeline:"
	@echo "  make -f replication/Makefile pipeline"
	@echo "  make -f replication/Makefile labels | impute | binning_fit | model_train_final | report"
	@echo "  make -f replication/Makefile oos_score | oos_vintage_report | oos_backtest_full"
	@echo ""
	@echo "(B) S&P grades / LDP app:"
	@echo "  make -f replication/Makefile sp_grade_all"
	@echo "  make -f replication/Makefile sp_snapshot | sp_grade_tables | sp_grade_plots"
	@echo ""
	@echo "(C) Simulations:"
	@echo "  make -f replication/Makefile sims_all"
	@echo "  make -f replication/Makefile binom_all | beta_binom_all | temporal_drift_all | prior_sens_all"
	@echo ""
	@echo "C-grid (fast + reproducible):"
	@echo "  C_ANCHOR=$(C_ANCHOR) | C_EXP_SPAN=$(C_EXP_SPAN) | C_NUM=$(C_NUM)"
	@echo "  Computed exp bounds: [$(C_MIN_EXP), $(C_MAX_EXP)]"
	@echo "  Override: make -f replication/Makefile model_train_final C_ANCHOR=234.286842 C_EXP_SPAN=0.25 C_NUM=21"
	@echo "=============================================================="
 
# ============================================================================
# (A) EMPIRICAL APPLICATION #1 — MAIN PIPELINE
# ============================================================================

.PHONY: labels
labels:
	@echo "\n[1/6] GENERATING LABELS..."
	$(PY) src/make_labels.py --config $(LABELS_CONFIG) --workers 12

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

.PHONY: model_train_final
model_train_final:
	@echo "\n[4/6] MODEL TRAIN (FINAL) — reproducible grid around C_ANCHOR..."
	@echo "      C_ANCHOR   = $(C_ANCHOR)"
	@echo "      C_EXP_SPAN = $(C_EXP_SPAN)"
	@echo "      C_NUM      = $(C_NUM)"
	@echo "      C_MIN_EXP  = $(C_MIN_EXP)"
	@echo "      C_MAX_EXP  = $(C_MAX_EXP)"
	@test -n "$(strip $(C_MIN_EXP))" || (echo "[ERR] C_MIN_EXP is empty (check PY0/poetry availability)"; exit 1)
	@test -n "$(strip $(C_MAX_EXP))" || (echo "[ERR] C_MAX_EXP is empty (check PY0/poetry availability)"; exit 1)

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
		--grid-window-years $(GRID_YEARS) \
		--grid-time-col $(GRID_TIME_COL) \
		--grid-time-freq $(GRID_TIME_FREQ) \
		--ttc-window-years $(TTC_YEARS) \
		--scored-outdir $(SCORED_OUT_DIR) \
		--ttc-mode train \
		$(NO_INTERACTIONS_FLAG) \
		$(TIMING_FLAG)

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






.PHONY: pipeline
pipeline: labels impute binning_fit model_train_final report oos_vintage_report oos_backtest_full
	@echo "\n-------------------------------------------------------"
	@echo "✔ (A) MAIN PIPELINE COMPLETED."
	@echo "-------------------------------------------------------"

# ============================================================================
# (B) EMPIRICAL APPLICATION #2 — LDP / S&P GRADES (tables + plots)
#   + build the monthly snapshot dataset (CSV-only)
# ============================================================================

SPG_ROOT   ?= ldp_application

# Build monthly snapshot (CSV only)
SP_RAW_CSV     ?= $(SPG_ROOT)/data/raw/20220601_SP_Ratings_Services_Corporate.csv
SP_SNAP_CSV    ?= $(SPG_ROOT)/data/processed/sp_corporate_monthly.csv
SP_SNAP_SCRIPT ?= $(SPG_ROOT)/process_sp_base.py

SP_SNAP_START ?= 2010-01-01
SP_SNAP_END   ?= 2021-07-01

.PHONY: sp_snapshot
sp_snapshot:
	@echo "\n[SP-SNAPSHOT] Build monthly snapshot (CSV-only)..."
	@test -f "$(SP_RAW_CSV)" || (echo "[ERR] Missing raw CSV: $(SP_RAW_CSV)"; exit 1)
	@test -f "$(SP_SNAP_SCRIPT)" || (echo "[ERR] Missing script: $(SP_SNAP_SCRIPT)"; exit 1)
	$(PYP) "$(SP_SNAP_SCRIPT)" \
		--raw "$(SP_RAW_CSV)" \
		--out_csv "$(SP_SNAP_CSV)" \
		--start "$(SP_SNAP_START)" \
		--end   "$(SP_SNAP_END)"
	@echo "✔ Snapshot written: $(SP_SNAP_CSV)"

# S&P grades (tables + plots)
SPG_FILE   ?= $(SP_SNAP_CSV)
SPG_OUTDIR ?= $(SPG_ROOT)/outputs/sp_grade_is_oos

SPG_AGENCY         ?= Standard & Poor's Ratings Services
SPG_HORIZON_MONTHS ?= 12
SPG_CONF_LEVEL     ?= 0.95

SPG_IS_START  ?= 2010
SPG_IS_END    ?= 2018
SPG_OOS_START ?= 2010
SPG_OOS_END   ?= 2020

SPG_TTC_SOURCE    ?= sp2012
SPG_TTC_ESTIMATOR ?= pooled
SPG_DROP_NO_TTC   ?= 1

SPG_PREFER_YEAR_COLUMN ?= 1

SPG_PMAX        ?=
SPG_MIN_TOTAL_N ?= 1

SPG_TABLES_CSV ?= $(SPG_OUTDIR)/sp_grade_tables_$(SPG_OOS_START)_$(SPG_OOS_END).csv

.PHONY: sp_grade_tables sp_grade_plots sp_grade_all

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
	@echo "\n-------------------------------------------------------"
	@echo "✔ (B) S&P / LDP PIPELINE COMPLETED."
	@echo "-------------------------------------------------------"

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
	@echo "\n-------------------------------------------------------"
	@echo "✔ (C) SIMULATIONS COMPLETED."
	@echo "-------------------------------------------------------"

.PHONY: plot_all_sims
plot_all_sims: prior_sens_plots beta_binom_plots temporal_drift_plots binom_plots

# ============================================================================
# PAPER BUNDLE (A + B + C)
# ============================================================================
.PHONY: paper_all
paper_all: pipeline sp_grade_all sims_all
	@echo "\n======================================================="
	@echo "✔ PAPER REPLICATION COMPLETED: (A) + (B) + (C)"
	@echo "======================================================="

# ----------------------------------------------------------------------------
# CLEAN (replication outputs only)
# ----------------------------------------------------------------------------
.PHONY: clean_replication_outputs
clean_replication_outputs:
	rm -rf $(DATA_DIR) $(ARTIFACTS_DIR) $(REPORTS_DIR) $(OOSBT_OUTDIR)
	@echo "Replication outputs cleaned."
