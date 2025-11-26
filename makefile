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

# --- 3. Configuration Globale ---
# Labels
LABELS_CONFIG  ?= config.yml
LABELS_WINDOW  ?= 24
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

# OOS & Reporting
OOS_LABELS_PARQUET = $(LABELS_OUTDIR)/window=$(LABELS_WINDOW)m/oos.parquet
SCORED_TRAIN_DATA  = $(SCORED_OUT_DIR)/train_scored.parquet
SCORED_VAL_DATA    = $(SCORED_OUT_DIR)/validation_scored.parquet
REPORT_TRAIN_HTML  = $(REPORTS_DIR)/train_report.html
REPORT_VAL_HTML    = $(REPORTS_DIR)/validation_report.html

# ============================================================================
# CIBLES (TARGETS)
# ============================================================================

.PHONY: help
help:
	@echo "================================================================="
	@echo "   PIPELINE RISQUE DE CRÉDIT -- FOCUS CALIBRATION"
	@echo "================================================================="
	@echo "  make pipeline      : Exécute le flux principal (Labels -> Report)"
	@echo "-----------------------------------------------------------------"
	@echo "  make labels        : 1. Génère les labels"
	@echo "  make impute        : 2. Impute (Fit Train / Trans. Val)"
	@echo "  make binning_fit   : 3. Discrétise (Max Gini + Monotonicité)"
	@echo "  make model_train   : 4. Entraîne (LR + Calibration + Grille)"
	@echo "  make report        : 5. Génère les rapports HTML (Train & Val)"
	@echo "-----------------------------------------------------------------"
	@echo "  make oos_score     : [MANUEL] Applique le modèle sur OOS"
	@echo "  make score_custom  : [MANUEL] Scorer un fichier spécifique"
	@echo "  make clean_all     : Nettoyage complet"
	@echo "================================================================="

# ----------------------------------------------------------------------------
# 1. LABELS
# ----------------------------------------------------------------------------
.PHONY: labels
labels:
	@echo "\n[1/5] GÉNÉRATION DES LABELS..."
	$(PY) src/make_labels.py --config $(LABELS_CONFIG) --workers 6

# ----------------------------------------------------------------------------
# 2. IMPUTATION (Anti-Leakage)
# ----------------------------------------------------------------------------
.PHONY: impute
impute:
	@echo "\n[2/5] IMPUTATION (Fit on Train / Transform All)..."
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
	@echo "\n[3/5] BINNING (Max Gini + Monotonicity Check)..."
	$(PY) src/fit_binning.py \
		--train $(BINNING_TRAIN) \
		--validation $(BINNING_VAL) \
		--target $(IMPUTE_TARGET_COL) \
		--outdir $(BINNING_OUT_DIR) \
		--artifacts $(BINNING_ARTIFACTS_DIR) \
		--max-bins-num 6 \
		--min-bin-size-num 200 \
		--n-quantiles-num 50

# ----------------------------------------------------------------------------
# 4. MODEL TRAINING (Master Scale)
# ----------------------------------------------------------------------------
.PHONY: model_train
model_train:
	@echo "\n[4/5] ENTRAÎNEMENT DU MODÈLE & CRÉATION GRILLE..."
	$(PY) src/train_model.py \
		--train $(MODEL_TRAIN_DATA) \
		--validation $(MODEL_VAL_DATA) \
		--target $(IMPUTE_TARGET_COL) \
		--artifacts $(MODEL_ARTIFACTS_DIR) \
		--cv-folds 5 \
		--corr-threshold 0.85 \
		--calibration isotonic \
		--n-buckets 10

# ----------------------------------------------------------------------------
# 5. REPORTING (TRAIN + VALIDATION dans un SEUL HTML)
# ----------------------------------------------------------------------------
.PHONY: report
report:
	@echo "\n[5/5] GENERATION DU RAPPORT GLOBAL..."

	@echo "-> Scoring TRAIN..."
	$(PY) src/apply_model.py \
		--data $(MODEL_TRAIN_DATA) \
		--out $(SCORED_TRAIN_DATA) \
		--imputer $(IMPUTE_ARTIFACTS_DIR)/imputer.joblib \
		--bins $(BINNING_ARTIFACTS_DIR)/bins.json \
		--model $(MODEL_ARTIFACTS_DIR)/model_best.joblib \
		--buckets $(MODEL_ARTIFACTS_DIR)/risk_buckets.json \
		--target $(IMPUTE_TARGET_COL) \
		--id-col loan_sequence_number

	@echo "-> Scoring VALIDATION..."
	$(PY) src/apply_model.py \
		--data $(MODEL_VAL_DATA) \
		--out $(SCORED_VAL_DATA) \
		--imputer $(IMPUTE_ARTIFACTS_DIR)/imputer.joblib \
		--bins $(BINNING_ARTIFACTS_DIR)/bins.json \
		--model $(MODEL_ARTIFACTS_DIR)/model_best.joblib \
		--buckets $(MODEL_ARTIFACTS_DIR)/risk_buckets.json \
		--target $(IMPUTE_TARGET_COL) \
		--id-col loan_sequence_number

	@echo "-> Création du RAPPORT UNIQUE (Train + Validation)..."
	$(PY) src/generate_report.py \
		--train $(SCORED_TRAIN_DATA) \
		--validation $(SCORED_VAL_DATA) \
		--out $(REPORTS_DIR)/model_validation_report.html \
		--target $(IMPUTE_TARGET_COL) \
		--score score --pd pd --grade grade \
		--model $(MODEL_ARTIFACTS_DIR)/model_best.joblib

	@echo "✔ Rapport généré : $(REPORTS_DIR)/model_validation_report.html"
	@open $(REPORTS_DIR)/model_validation_report.html 2>/dev/null || true

# ----------------------------------------------------------------------------
# PIPELINE GLOBAL
# ----------------------------------------------------------------------------
.PHONY: pipeline
pipeline: labels impute binning_fit model_train report
	@echo "\n-------------------------------------------------------"
	@echo "✔ PIPELINE DE VALIDATION TERMINÉ."
	@echo "-------------------------------------------------------"

# ============================================================================
# MODULES STANDALONE (HORS PIPELINE AUTOMATIQUE)
# ============================================================================

# --- OOS SCORING ---
.PHONY: oos_score
oos_score:
	@echo "\n[MANUEL] SCORING ECHANTILLON OOS (Out-of-Sample)..."
	@test -f "$(OOS_LABELS_PARQUET)" || (echo "[ERR] Fichier OOS manquant : $(OOS_LABELS_PARQUET)"; exit 1)
	$(PY) src/apply_model.py \
		--data $(OOS_LABELS_PARQUET) \
		--out $(SCORED_OUT_DIR)/oos_scored.parquet \
		--imputer $(IMPUTE_ARTIFACTS_DIR)/imputer.joblib \
		--bins $(BINNING_ARTIFACTS_DIR)/bins.json \
		--model $(MODEL_ARTIFACTS_DIR)/model_best.joblib \
		--buckets $(MODEL_ARTIFACTS_DIR)/risk_buckets.json \
		--target $(IMPUTE_TARGET_COL) \
		--id-col loan_sequence_number
	@echo "✔ Scoring terminé : $(SCORED_OUT_DIR)/oos_scored.parquet"

# --- SCORING CUSTOM ---
CUSTOM_DATA ?= $(OOS_LABELS_PARQUET)
CUSTOM_OUT  ?= $(SCORED_OUT_DIR)/custom_scored.parquet
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