# ============================================================================
# Makefile – Pipeline Risque de Crédit (Bank-Grade)
# ============================================================================
# Objectif :
#   - Pipeline principal : Labels -> Impute -> Binning -> Model -> TTC Macro -> Report
#   - Modules manuels : scoring OOS, reports vintage/grade, scoring custom
#   - Recalibration (type 1) : update table grade -> PD (mean ou isotonic)
#   - Application Master Scale : ajoute une PD "pd_ms" aux jeux scorés
# ============================================================================

.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help


# ============================================================================
# 0) ENVIRONNEMENT
# ============================================================================
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
PY ?= PYTHONPATH=src poetry run python


# ============================================================================
# 1) RÉPERTOIRES
# ============================================================================
DATA_DIR      := data/processed
ARTIFACTS_DIR := artifacts
REPORTS_DIR   := reports

# Séries macro brutes (FRED)
MACRO_RAW_DIR := data/raw/macro


# ============================================================================
# 2) CONFIGURATION GLOBALE
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

# 2.5 Scored (sorties de train_model.py)
SCORED_TRAIN_DATA   = $(SCORED_OUT_DIR)/train_scored.parquet
SCORED_VAL_DATA     = $(SCORED_OUT_DIR)/validation_scored.parquet

# 2.6 TTC Macro
PD_TTC_MACRO_JSON   = $(MODEL_ARTIFACTS_DIR)/pd_ttc_macro.json

# 2.7 OOS & Reporting
OOS_LABELS_PARQUET  = $(LABELS_OUTDIR)/window=$(LABELS_WINDOW)m/oos.parquet
OOS_SCORED          = $(SCORED_OUT_DIR)/oos_scored.parquet

REPORT_HTML         = $(REPORTS_DIR)/model_validation_report.html
VINTAGE_REPORT_OOS  = $(REPORTS_DIR)/vintage_grade_oos.html
VINTAGE_REPORT_VAL  = $(REPORTS_DIR)/vintage_grade_validation.html


# ============================================================================
# 3) AIDE
# ============================================================================
.PHONY: help
help:
	@echo "================================================================="
	@echo "   PIPELINE RISQUE DE CRÉDIT -- FOCUS CALIBRATION"
	@echo "================================================================="
	@echo "  make pipeline                 : (1) labels -> (2) impute -> (3) binning -> (4) model -> (5) ttc_macro -> (6) report"
	@echo "-----------------------------------------------------------------"
	@echo "  [PIPELINE]"
	@echo "    make labels                 : 1) Génère les labels"
	@echo "    make impute                 : 2) Impute (Fit Train / Transform All)"
	@echo "    make binning_fit            : 3) Discrétise (Max Gini + Monotonicité)"
	@echo "    make model_train            : 4) Entraîne (LR + Calibration + Grille + scored)"
	@echo "    make ttc_macro              : 5) Estime la PD TTC macro par grade"
	@echo "    make report                 : 6) Génère le rapport HTML (Train+Val scorés)"
	@echo "-----------------------------------------------------------------"
	@echo "  [RECALIBRATION MASTER SCALE]"
	@echo "    make recalibrate_pd         : Recalibration type (1) (grade->PD) sur fenêtre glissante"
	@echo "    make val_apply_ms           : Applique la master scale recalibrée sur Validation (ajoute pd_ms)"
	@echo "    make oos_apply_ms           : Applique la master scale recalibrée sur OOS (ajoute pd_ms)"
	@echo "-----------------------------------------------------------------"
	@echo "  [MODULES MANUELS]"
	@echo "    make oos_score              : Scoring OOS (fichier oos.parquet)"
	@echo "    make oos_vintage_report     : Scoring OOS + report vintage/grade"
	@echo "    make val_vintage_report     : Report vintage/grade sur Validation"
	@echo "    make score_custom           : Scorer un fichier spécifique"
	@echo "    make clean_all              : Nettoyage complet"
	@echo "================================================================="


# ============================================================================
# 4) PIPELINE PRINCIPAL (1 -> 6)
# ============================================================================

# ----------------------------------------------------------------------------
# 4.1) LABELS
# ----------------------------------------------------------------------------
.PHONY: labels
labels:
	@echo "\n[1/6] GÉNÉRATION DES LABELS..."
	$(PY) src/make_labels.py --config $(LABELS_CONFIG) --workers 12

# ----------------------------------------------------------------------------
# 4.2) IMPUTATION (Anti-Leakage)
# ----------------------------------------------------------------------------
.PHONY: impute
impute:
	@echo "\n[2/6] IMPUTATION (Fit on Train / Transform All)..."
	$(PY) src/impute_and_save.py \
		--labels-window-dir $(LABELS_OUTDIR)/window=$(LABELS_WINDOW)m \
		--target $(IMPUTE_TARGET_COL) \
		--outdir $(IMPUTE_OUT_DIR) \
		--artifacts $(IMPUTE_ARTIFACTS_DIR) \
		--use-splits \
		--fail-on-nan

# ----------------------------------------------------------------------------
# 4.3) BINNING (Monotone)
# ----------------------------------------------------------------------------
.PHONY: binning_fit
binning_fit:
	@echo "\n[3/6] BINNING (Max Gini + Monotonicity Check)..."
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
# 4.4) MODEL TRAINING (Master Scale + SCORING TRAIN/VAL)
# ----------------------------------------------------------------------------
# Tu peux surcharger ces paramètres à l'appel:
#   make model_train C_NUM=80 C_MIN_EXP=-8 C_MAX_EXP=4 SEARCH=halving GRID_YEARS=10 TTC_YEARS=10 COEF_STATS=none
#
SEARCH        ?= halving     # grid | halving
C_MIN_EXP     ?= -8
C_MAX_EXP     ?= 4
C_NUM         ?= 60
HALVING_FACTOR?= 3
LR_SOLVER     ?= lbfgs       # lbfgs | saga | newton-cg
LR_MAX_ITER   ?= 3000
COEF_STATS    ?= none # statsmodels | none

GRID_YEARS    ?= 10
TTC_YEARS     ?= 10
GRID_TIME_COL ?= vintage
GRID_TIME_FREQ?= Q           # Q | M

N_BUCKETS     ?= 10
# optionnel : choisir automatiquement n_buckets (ex: 7,10,12,15)
# N_BUCKETS_CANDIDATES ?= 7,10,12,15
MIN_BUCKET_COUNT ?= 300
MIN_BUCKET_BAD   ?= 5

.PHONY: model_train
model_train:
	@echo "\n[4/6] ENTRAÎNEMENT DU MODÈLE & CRÉATION GRILLE + SCORING TRAIN/VAL..."
	$(PY) src/train_model.py \
		--train $(MODEL_TRAIN_DATA) \
		--validation $(MODEL_VAL_DATA) \
		--target $(IMPUTE_TARGET_COL) \
		--artifacts $(MODEL_ARTIFACTS_DIR) \
		--cv-folds 5 \
		--corr-threshold 0.85 \
		--calibration isotonic \
		--search $(SEARCH) \
		--c-min-exp $(C_MIN_EXP) \
		--c-max-exp $(C_MAX_EXP) \
		--c-num $(C_NUM) \
		--halving-factor $(HALVING_FACTOR) \
		--lr-solver $(LR_SOLVER) \
		--lr-max-iter $(LR_MAX_ITER) \
		--coef-stats $(COEF_STATS) \
		--n-buckets $(N_BUCKETS) \
		--grid-window-years $(GRID_YEARS) \
		--grid-time-col $(GRID_TIME_COL) \
		--grid-time-freq $(GRID_TIME_FREQ) \
		--ttc-window-years $(TTC_YEARS) \
		--scored-outdir $(SCORED_OUT_DIR) \
		--ttc-mode train

# ----------------------------------------------------------------------------
# 4.5) PD TTC MACRO (grade × temps + macro)
# ----------------------------------------------------------------------------
.PHONY: ttc_macro
ttc_macro:
	@echo "\n[5/6] ESTIMATION PD TTC MACRO PAR GRADE (pd_ttc_macro)..."
	@test -f "$(SCORED_TRAIN_DATA)" || (echo "[ERR] Fichier train_scored manquant : $(SCORED_TRAIN_DATA)"; exit 1)
	@test -f "$(SCORED_VAL_DATA)"   || (echo "[ERR] Fichier validation_scored manquant : $(SCORED_VAL_DATA)"; exit 1)
	$(PY) src/estimate_ttc_macro.py \
		--train-scored $(SCORED_TRAIN_DATA) \
		--val-scored $(SCORED_VAL_DATA) \
		--macro-dir $(MACRO_RAW_DIR) \
		--target $(IMPUTE_TARGET_COL) \
		--time-col vintage \
		--out-json $(PD_TTC_MACRO_JSON)
	@echo "✔ PD TTC macro par grade sauvegardée dans : $(PD_TTC_MACRO_JSON)"

# ----------------------------------------------------------------------------
# 4.6) REPORTING (TRAIN + VALIDATION déjà scorés)
# ----------------------------------------------------------------------------
.PHONY: report
report:
	@echo "\n[6/6] GENERATION DU RAPPORT GLOBAL (Train + Validation scorés)..."
	$(PY) src/generate_report.py \
		--train $(SCORED_TRAIN_DATA) \
		--validation $(SCORED_VAL_DATA) \
		--out $(REPORT_HTML) \
		--target $(IMPUTE_TARGET_COL) \
		--score score_ttc --pd pd --grade grade \
		--model $(MODEL_ARTIFACTS_DIR)/model_best.joblib
	@echo "✔ Rapport généré : $(REPORT_HTML)"
	@open $(REPORT_HTML) 2>/dev/null || true

# ----------------------------------------------------------------------------
# 4.X) PIPELINE GLOBAL
# ----------------------------------------------------------------------------
.PHONY: pipeline
pipeline: labels impute binning_fit model_train ttc_macro report
	@echo "\n-------------------------------------------------------"
	@echo "✔ PIPELINE DE VALIDATION TERMINÉ."
	@echo "-------------------------------------------------------"


# ============================================================================
# 5) RECALIBRATION MASTER SCALE (Type 1 : update grade -> PD)
# ============================================================================

# Paramètres recalibration
RECAL_METHOD ?= mean        # mean | isotonic
RECAL_AGG    ?= time_mean   # pooled | time_mean
RECAL_YEARS  ?= 5

BUCKET_STATS_RECAL := $(MODEL_ARTIFACTS_DIR)/bucket_stats_recalibrated.json

.PHONY: recalibrate_pd
recalibrate_pd:
	@echo "\n[REC] Recalibration PD par grade (method=$(RECAL_METHOD), agg=$(RECAL_AGG), years=$(RECAL_YEARS))..."
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
	@echo "✔ Bucket stats recalibrés : $(BUCKET_STATS_RECAL)"


# ============================================================================
# 6) APPLICATION MASTER SCALE (ajoute pd_ms aux fichiers scorés)
# ============================================================================

SCORED_VAL_MS := $(SCORED_OUT_DIR)/validation_scored_ms.parquet
OOS_SCORED_MS := $(SCORED_OUT_DIR)/oos_scored_ms.parquet

.PHONY: val_apply_ms
val_apply_ms:
	@echo "\n[MS] Application master scale recalibrée sur Validation..."
	@test -f "$(SCORED_VAL_DATA)"      || (echo "[ERR] Missing: $(SCORED_VAL_DATA)"; exit 1)
	@test -f "$(BUCKET_STATS_RECAL)"   || (echo "[ERR] Missing: $(BUCKET_STATS_RECAL)"; exit 1)
	$(PY) src/apply_master_scale.py \
		--in-scored $(SCORED_VAL_DATA) \
		--out $(SCORED_VAL_MS) \
		--bucket-stats $(BUCKET_STATS_RECAL) \
		--grade-col grade \
		--pd-col-out pd_ms
	@echo "✔ Validation enrichie : $(SCORED_VAL_MS)"

.PHONY: oos_apply_ms
oos_apply_ms: oos_score
	@echo "\n[MS] Application master scale recalibrée sur OOS..."
	@test -f "$(OOS_SCORED)"          || (echo "[ERR] Missing: $(OOS_SCORED)"; exit 1)
	@test -f "$(BUCKET_STATS_RECAL)" || (echo "[ERR] Missing: $(BUCKET_STATS_RECAL)"; exit 1)
	$(PY) src/apply_master_scale.py \
		--in-scored $(OOS_SCORED) \
		--out $(OOS_SCORED_MS) \
		--bucket-stats $(BUCKET_STATS_RECAL) \
		--grade-col grade \
		--pd-col-out pd_ms
	@echo "✔ OOS enrichi : $(OOS_SCORED_MS)"


# ============================================================================
# 7) MODULES STANDALONE (HORS PIPELINE AUTOMATIQUE)
# ============================================================================

# ----------------------------------------------------------------------------
# 7.1) OOS SCORING (Full pipeline OOS)
# ----------------------------------------------------------------------------
.PHONY: oos_score
oos_score:
	@echo "\n[MANUEL] SCORING ECHANTILLON OOS (Out-of-Sample, full pipeline)..."
	@test -f "$(OOS_LABELS_PARQUET)" || (echo "[ERR] Fichier OOS manquant : $(OOS_LABELS_PARQUET)"; exit 1)
	$(PY) src/apply_model.py \
		--data $(OOS_LABELS_PARQUET) \
		--out $(OOS_SCORED) \
		--imputer $(IMPUTE_ARTIFACTS_DIR)/imputer.joblib \
		--bins $(BINNING_ARTIFACTS_DIR)/bins.json \
		--model $(MODEL_ARTIFACTS_DIR)/model_best.joblib \
		--buckets $(MODEL_ARTIFACTS_DIR)/risk_buckets.json \
		--target $(IMPUTE_TARGET_COL) \
		--id-col loan_sequence_number
	@echo "✔ Scoring terminé : $(OOS_SCORED)"

# ----------------------------------------------------------------------------
# 7.2) OOS: Vintage/Grade report
# ----------------------------------------------------------------------------
.PHONY: oos_vintage_report
oos_vintage_report: oos_score
	@echo "\n[MANUEL] VINTAGE / GRADE REPORT SUR OOS..."
	$(PY) src/generate_vintage_grade_report.py \
		--data $(OOS_SCORED) \
		--out $(VINTAGE_REPORT_OOS) \
		--vintage-col vintage \
		--grade-col grade \
		--pd-col pd \
		--target $(IMPUTE_TARGET_COL) \
		--sample-name "OOS" \
		--bucket-stats $(MODEL_ARTIFACTS_DIR)/bucket_stats.json
	@echo "✔ Rapport OOS Vintage / Grade généré : $(VINTAGE_REPORT_OOS)"
	@open $(VINTAGE_REPORT_OOS) 2>/dev/null || true

# ----------------------------------------------------------------------------
# 7.3) Validation: Vintage/Grade report
# ----------------------------------------------------------------------------
.PHONY: val_vintage_report
val_vintage_report:
	@echo "\n[MANUEL] VINTAGE / GRADE REPORT SUR VALIDATION..."
	@test -f "$(SCORED_VAL_DATA)" || (echo "[ERR] Fichier validation_scored manquant : $(SCORED_VAL_DATA)"; exit 1)
	$(PY) src/generate_vintage_grade_report.py \
		--data $(SCORED_VAL_DATA) \
		--out $(VINTAGE_REPORT_VAL) \
		--vintage-col vintage \
		--grade-col grade \
		--pd-col pd \
		--target $(IMPUTE_TARGET_COL) \
		--sample-name "Validation" \
		--bucket-stats $(MODEL_ARTIFACTS_DIR)/bucket_stats.json
	@echo "✔ Rapport Validation Vintage / Grade généré : $(VINTAGE_REPORT_VAL)"
	@open $(VINTAGE_REPORT_VAL) 2>/dev/null || true


# ----------------------------------------------------------------------------
# 7.4) SCORING CUSTOM
# ----------------------------------------------------------------------------
CUSTOM_DATA   ?= $(OOS_LABELS_PARQUET)
CUSTOM_OUT    ?= $(SCORED_OUT_DIR)/custom_scored.parquet
CUSTOM_TARGET ?= $(IMPUTE_TARGET_COL)

.PHONY: score_custom
score_custom:
	@echo "\n[MANUEL] Scoring fichier : $(CUSTOM_DATA)"
	@test -f "$(CUSTOM_DATA)" || (echo "[ERR] Fichier introuvable : $(CUSTOM_DATA)"; exit 1)
	$(PY) src/apply_model.py \
		--data $(CUSTOM_DATA) \
		--out $(CUSTOM_OUT) \
		--imputer $(IMPUTE_ARTIFACTS_DIR)/imputer.joblib \
		--bins $(BINNING_ARTIFACTS_DIR)/bins.json \
		--model $(MODEL_ARTIFACTS_DIR)/model_best.joblib \
		--buckets $(MODEL_ARTIFACTS_DIR)/risk_buckets.json \
		--target $(CUSTOM_TARGET) \
		--id-col loan_sequence_number
	@echo "✔ Résultat : $(CUSTOM_OUT)"


# ----------------------------------------------------------------------------
# 7.5) NETTOYAGE
# ----------------------------------------------------------------------------
.PHONY: clean_all
clean_all:
	rm -rf $(DATA_DIR) $(ARTIFACTS_DIR) $(REPORTS_DIR)
	@echo "✔ Tout est nettoyé."


# ============================================================================
# 8) SIMULATIONS
# ============================================================================

# --- Binomial ---
.PHONY: binom_sim binom_plots binom_tables binom_all
binom_sim:
	poetry run python -m experiments.binom_coverage.sim_binom
binom_plots:
	poetry run python -m experiments.binom_coverage.plot_binom
binom_tables:
	poetry run python -m experiments.binom_coverage.make_table_binom
binom_all: binom_sim binom_plots binom_tables

# --- Beta-binomial ---
.PHONY: beta_binom_sim beta_binom_plots beta_binom_tables beta_binom_all
beta_binom_sim:
	poetry run python -m experiments.beta_binom_jeffreys.sim_beta_binom
beta_binom_plots:
	poetry run python -m experiments.beta_binom_jeffreys.plot_beta_binom
beta_binom_tables:
	poetry run python -m experiments.beta_binom_jeffreys.make_table_beta_binom
beta_binom_all: beta_binom_sim beta_binom_plots beta_binom_tables

# --- Temporal drift ---
.PHONY: temporal_drift_sim temporal_drift_plots temporal_drift_tables temporal_drift_all
temporal_drift_sim:
	poetry run python -m experiments.temporal_drift.sim_temporal_drift
temporal_drift_plots:
	poetry run python -m experiments.temporal_drift.plot_temporal_drift
temporal_drift_tables:
	poetry run python -m experiments.temporal_drift.make_table_temporal_drift
temporal_drift_all: temporal_drift_sim temporal_drift_plots temporal_drift_tables

# --- Tout lancer ---
.PHONY: sims_all
sims_all: beta_binom_all temporal_drift_all binom_all
