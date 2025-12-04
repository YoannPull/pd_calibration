# ============================================================================
# Makefile – Pipeline Risque de Crédit (Bank-Grade)
# ============================================================================

.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

# --- 1. Environnement ---
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
PY ?= PYTHONPATH=src poetry run python

# --- 2. Répertoires ---
DATA_DIR      := data/processed
ARTIFACTS_DIR := artifacts
REPORTS_DIR   := reports

# Répertoire des séries macro brutes (FRED)
MACRO_RAW_DIR := data/raw/macro

# --- 3. Configuration Globale ---
# Labels
LABELS_CONFIG  ?= config.yml
LABELS_WINDOW  ?= 12
LABELS_OUTDIR   = $(DATA_DIR)/default_labels

# Imputation
IMPUTE_TARGET_COL    = default_$(LABELS_WINDOW)m
IMPUTE_OUT_DIR       = $(DATA_DIR)/imputed
IMPUTE_ARTIFACTS_DIR = $(ARTIFACTS_DIR)/imputer

# Binning
BINNING_OUT_DIR       = $(DATA_DIR)/binned
BINNING_ARTIFACTS_DIR = $(ARTIFACTS_DIR)/binning_maxgini
BINNING_TRAIN         = $(IMPUTE_OUT_DIR)/train.parquet
BINNING_VAL           = $(IMPUTE_OUT_DIR)/validation.parquet

# Modeling
MODEL_ARTIFACTS_DIR = $(ARTIFACTS_DIR)/model_from_binned
SCORED_OUT_DIR      = $(DATA_DIR)/scored
MODEL_TRAIN_DATA    = $(BINNING_OUT_DIR)/train.parquet
MODEL_VAL_DATA      = $(BINNING_OUT_DIR)/validation.parquet

# Scored (produits par train_model.py)
SCORED_TRAIN_DATA   = $(SCORED_OUT_DIR)/train_scored.parquet
SCORED_VAL_DATA     = $(SCORED_OUT_DIR)/validation_scored.parquet

# PD TTC macro (sortie de estimate_ttc_macro.py)
PD_TTC_MACRO_JSON   = $(MODEL_ARTIFACTS_DIR)/pd_ttc_macro.json

# OOS & Reporting
OOS_LABELS_PARQUET  = $(LABELS_OUTDIR)/window=$(LABELS_WINDOW)m/oos.parquet
OOS_SCORED          = $(SCORED_OUT_DIR)/oos_scored.parquet
REPORT_TRAIN_HTML   = $(REPORTS_DIR)/train_report.html
REPORT_VAL_HTML     = $(REPORTS_DIR)/validation_report.html
VINTAGE_REPORT_OOS  = $(REPORTS_DIR)/vintage_grade_oos.html
VINTAGE_REPORT_VAL  = $(REPORTS_DIR)/vintage_grade_validation.html

# ============================================================================
# CIBLES (TARGETS)
# ============================================================================

.PHONY: help
help:
	@echo "================================================================="
	@echo "   PIPELINE RISQUE DE CRÉDIT -- FOCUS CALIBRATION"
	@echo "================================================================="
	@echo "  make pipeline            : Labels -> Impute -> Binning -> Model -> TTC_macro -> Report"
	@echo "-----------------------------------------------------------------"
	@echo "  make labels              : 1. Génère les labels"
	@echo "  make impute              : 2. Impute (Fit Train / Transform All)"
	@echo "  make binning_fit         : 3. Discrétise (Max Gini + Monotonicité)"
	@echo "  make model_train         : 4. Entraîne (LR + Calibration + Grille, + scored)"
	@echo "  make ttc_macro           : 5. Estime la PD TTC macro par grade (pd_ttc_macro)"
	@echo "  make report              : 6. Génère le rapport HTML (Train & Val scored)"
	@echo "-----------------------------------------------------------------"
	@echo "  make oos_score           : [MANUEL] Scoring OOS (fichier oos.parquet)"
	@echo "  make oos_vintage_report  : [MANUEL] Scoring OOS + Vintage/Grade report"
	@echo "  make val_vintage_report  : [MANUEL] Vintage/Grade report sur Validation"
	@echo "  make score_custom        : [MANUEL] Scorer un fichier spécifique"
	@echo "  make clean_all           : Nettoyage complet"
	@echo "================================================================="

# ----------------------------------------------------------------------------
# 1. LABELS
# ----------------------------------------------------------------------------
.PHONY: labels
labels:
	@echo "\n[1/6] GÉNÉRATION DES LABELS..."
	$(PY) src/make_labels.py --config $(LABELS_CONFIG) --workers 12

# ----------------------------------------------------------------------------
# 2. IMPUTATION (Anti-Leakage)
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
# 3. BINNING (Monotone)
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
# 4. MODEL TRAINING (Master Scale + SCORING TRAIN/VAL)
# ----------------------------------------------------------------------------
# --ttc-mode train_val ou train

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
		--n-buckets 10 \
		--scored-outdir $(SCORED_OUT_DIR) \
		--ttc-mode train

# ----------------------------------------------------------------------------
# 5. PD TTC MACRO (grade × temps + macro)
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
# 6. REPORTING (TRAIN + VALIDATION déjà scorés)
# ----------------------------------------------------------------------------
.PHONY: report
report:
	@echo "\n[6/6] GENERATION DU RAPPORT GLOBAL (Train + Validation scored)..."

	@echo "-> Création du RAPPORT UNIQUE (Train + Validation)..."
	$(PY) src/generate_report.py \
		--train $(SCORED_TRAIN_DATA) \
		--validation $(SCORED_VAL_DATA) \
		--out $(REPORTS_DIR)/model_validation_report.html \
		--target $(IMPUTE_TARGET_COL) \
		--score score_ttc --pd pd --grade grade \
		--model $(MODEL_ARTIFACTS_DIR)/model_best.joblib

	@echo "✔ Rapport généré : $(REPORTS_DIR)/model_validation_report.html"
	@open $(REPORTS_DIR)/model_validation_report.html 2>/dev/null || true

# ----------------------------------------------------------------------------
# PIPELINE GLOBAL
# ----------------------------------------------------------------------------
.PHONY: pipeline
pipeline: labels impute binning_fit model_train ttc_macro report
	@echo "\n-------------------------------------------------------"
	@echo "✔ PIPELINE DE VALIDATION TERMINÉ."
	@echo "-------------------------------------------------------"

# ============================================================================
# MODULES STANDALONE (HORS PIPELINE AUTOMATIQUE)
# ============================================================================

# --- OOS SCORING (FULL PIPELINE SUR OOS) ---
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

# --- OOS SCORING + VINTAGE/GRADE REPORT ---
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
		--bucket-stats artifacts/model_from_binned/bucket_stats.json
	@echo "✔ Rapport OOS Vintage / Grade généré : $(VINTAGE_REPORT_OOS)"
	@open $(VINTAGE_REPORT_OOS) 2>/dev/null || true

# --- VALIDATION VINTAGE/GRADE REPORT ---
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
		--bucket-stats artifacts/model_from_binned/bucket_stats.json
	@echo "✔ Rapport Validation Vintage / Grade généré : $(VINTAGE_REPORT_VAL)"
	@open $(VINTAGE_REPORT_VAL) 2>/dev/null || true


# --- SCORING CUSTOM ---
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

# --- NETTOYAGE ---
.PHONY: clean_all
clean_all:
	rm -rf $(DATA_DIR) $(ARTIFACTS_DIR) $(REPORTS_DIR)
	@echo "✔ Tout est nettoyé."
