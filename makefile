# ============================================================================
# Makefile – Pipeline : labels → impute → binning → train → apply
# (sans runs horodatés : sorties fixes et stables)
#
# Usage :
#   make help
#   make labels
#   make impute VAL_QUARTER=2022Q4
#   make binning_fit
#   make model_train
#   make pipeline VAL_QUARTER=2022Q4        # labels → impute → binning_fit → model_train
#   make pipeline SKIP_LABELS=1             # impute → binning_fit → model_train (réutilise labels existants)
# ============================================================================

.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

# Commande Python unifiée
PY ?= PYTHONPATH=src poetry run python

# Répertoires fixes (pas de timestamp)
DATA_BASE_DIR      := data/processed
ARTIFACTS_BASE_DIR := artifacts
REPORTS_BASE_DIR   := reports

# Dossiers de données par étape
LABELS_OUTDIR   ?= $(DATA_BASE_DIR)/labels
IMPUTE_OUT_DIR  ?= $(DATA_BASE_DIR)/imputed
BINNING_OUT_DIR ?= $(DATA_BASE_DIR)/binned
SCORED_OUT_DIR  ?= $(DATA_BASE_DIR)/scored

# Dossiers d'artefacts par étape
IMPUTE_ARTIFACTS_DIR  ?= $(ARTIFACTS_BASE_DIR)/imputer
BINNING_ARTIFACTS_DIR ?= $(ARTIFACTS_BASE_DIR)/binning
MODEL_ARTIFACTS_DIR   ?= $(ARTIFACTS_BASE_DIR)/model

# ---------------------------------------------------------------------------
# PARAMÈTRES AVEC VALEURS PAR DÉFAUT (définir AVANT le sanitize)
# ---------------------------------------------------------------------------
# LABELS
LABELS_CONFIG ?= config.yml
LABELS_FORMAT ?= parquet          # parquet | csv
LABELS_POOLED ?= 1                # 1 = sauvegarde aussi le pooled
LABELS_WINDOW ?= 24               # doit matcher config.yml: labels.window_months
# Parallélisation & frontière du pooled (si supportées par src/make_labels.py)
LABELS_WORKERS      ?=            # ex: 6
LABELS_POOLED_UNTIL ?=            # ex: 2022Q4

# D’où lire les labels ? (par défaut: ceux que tu viens de générer)
SRC_LABELS_DIR ?= $(LABELS_OUTDIR)

# train = pooled ; validation = un quarter tenu-out
VAL_QUARTER ?= 2022Q4

# --- Sanitize (enlève espaces/retours à la ligne accidentels) ---
LABELS_WINDOW      := $(strip $(LABELS_WINDOW))
VAL_QUARTER        := $(strip $(VAL_QUARTER))
SRC_LABELS_DIR     := $(strip $(SRC_LABELS_DIR))
LABELS_FORMAT      := $(strip $(LABELS_FORMAT))
LABELS_WORKERS     := $(strip $(LABELS_WORKERS))
LABELS_POOLED_UNTIL:= $(strip $(LABELS_POOLED_UNTIL))


# ----------------------------------------------------------------------------
# HELP
# ----------------------------------------------------------------------------
## help : affiche cette aide
.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mCibles disponibles\033[0m\n\n"} /^[a-zA-Z0-9_%-]+:.*##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^# Exemple/ {print} ' $(MAKEFILE_LIST)
	@printf "\n\033[1mExemples\033[0m\n"
	@echo "  make pipeline VAL_QUARTER=2022Q4"
	@echo "  make pipeline SKIP_LABELS=1        # saute la création des labels"
	@echo "  make labels LABELS_POOLED=1 LABELS_POOLED_UNTIL=2022Q4 LABELS_WORKERS=6"
	@echo "  make clean_all"


# ============================================================================
# 1) LABELS (src/make_labels.py)
# ============================================================================

LABELS_CMD ?= $(PY) src/make_labels.py
LABELS_POOLED_OPT       := $(if $(filter $(LABELS_POOLED),1),--pooled,)
LABELS_WORKERS_OPT      := $(if $(LABELS_WORKERS),--workers $(LABELS_WORKERS),)
LABELS_POOLED_UNTIL_OPT := $(if $(LABELS_POOLED_UNTIL),--pooled-until $(LABELS_POOLED_UNTIL),)

## labels : génère les labels (multi-quarters, optionnellement parallélisé) dans $(LABELS_OUTDIR)
.PHONY: labels
labels:
	$(LABELS_CMD) \
		--config $(LABELS_CONFIG) \
		--outdir $(LABELS_OUTDIR) \
		--format $(LABELS_FORMAT) \
		$(LABELS_POOLED_OPT) \
		$(LABELS_WORKERS_OPT) \
		$(LABELS_POOLED_UNTIL_OPT)

# Sorties attendues (ex. T=24) :
#   $(LABELS_OUTDIR)/window=$(LABELS_WINDOW)m/quarter=$(YYYYQn)/data.$(LABELS_FORMAT)
#   $(LABELS_OUTDIR)/window=$(LABELS_WINDOW)m/pooled.$(LABELS_FORMAT)


# ============================================================================
# 2) IMPUTATION (src/impute_and_save.py)
# ============================================================================

# Fichiers d'entrée (dérivés APRÈS sanitize)
IMPUTE_TRAIN_CSV ?= $(SRC_LABELS_DIR)/window=$(LABELS_WINDOW)m/pooled.$(LABELS_FORMAT)
IMPUTE_VAL_CSV   ?= $(SRC_LABELS_DIR)/window=$(LABELS_WINDOW)m/quarter=$(VAL_QUARTER)/data.$(LABELS_FORMAT)

IMPUTE_TARGET_COL   ?= default_$(LABELS_WINDOW)m
IMPUTE_FORMAT       ?= parquet
IMPUTE_COHORT       ?= 1       # 1 = imputation par cohortes
IMPUTE_MISSING_FLAG ?= 1       # 1 = ajoute les colonnes was_missing_*

IMPUTE_CMD ?= $(PY) src/impute_and_save.py
IMPUTE_COHORT_OPT  := $(if $(filter $(IMPUTE_COHORT),1),--use-cohort,)
IMPUTE_MISSING_OPT := $(if $(filter $(IMPUTE_MISSING_FLAG),1),--missing-flag,)

## check_labels : vérifie que les fichiers labels source existent
.PHONY: check_labels
check_labels:
	@test -f "$(IMPUTE_TRAIN_CSV)" || (echo "[ERR] Manque: $(IMPUTE_TRAIN_CSV)"; exit 1)
	@test -f "$(IMPUTE_VAL_CSV)"   || (echo "[ERR] Manque: $(IMPUTE_VAL_CSV)"; exit 1)

## impute : impute train/validation → $(IMPUTE_OUT_DIR) + artifacts → $(IMPUTE_ARTIFACTS_DIR)
.PHONY: impute
impute: check_labels
	$(IMPUTE_CMD) \
		--train-csv $(IMPUTE_TRAIN_CSV) \
		--validation-csv $(IMPUTE_VAL_CSV) \
		--target $(IMPUTE_TARGET_COL) \
		--outdir $(IMPUTE_OUT_DIR) \
		--artifacts $(IMPUTE_ARTIFACTS_DIR) \
		--format $(IMPUTE_FORMAT) \
		$(IMPUTE_COHORT_OPT) $(IMPUTE_MISSING_OPT)


# ============================================================================
# 3) BINNING (src/fit_binning.py / src/apply_binning.py)
# ============================================================================

BINNING_TRAIN         ?= $(IMPUTE_OUT_DIR)/train.parquet
BINNING_VAL           ?= $(IMPUTE_OUT_DIR)/validation.parquet
BINNING_TARGET        ?= $(IMPUTE_TARGET_COL)
BINNING_FORMAT        ?= parquet          # parquet | csv
BINNING_N_BINS        ?= 10
BINNING_OUTPUT        ?= woe              # woe | bin_index | both
BINNING_INCLUDE_CAT   ?= 0                # 1 = inclure catégorielles
BINNING_DROP_ORIGINAL ?= 0                # 1 = ne garder que les features encodées
BINNING_VARIABLES     ?=                  # "var1,var2,..."

BINNING_INCLUDE_CAT_OPT := $(if $(filter $(BINNING_INCLUDE_CAT),1),--include-categorical,)
BINNING_DROP_ORIG_OPT   := $(if $(filter $(BINNING_DROP_ORIGINAL),1),--drop-original,)
BINNING_VARIABLES_OPT   := $(if $(strip $(BINNING_VARIABLES)),--variables "$(BINNING_VARIABLES)",)

BINNING_FIT_CMD   ?= $(PY) src/fit_binning.py
BINNING_APPLY_CMD ?= $(PY) src/apply_binning.py

## binning_fit : fit du binning (WOE/IV) sur train, apply sur val → $(BINNING_OUT_DIR) + artefacts
.PHONY: binning_fit
binning_fit:
	$(BINNING_FIT_CMD) \
		--train $(BINNING_TRAIN) \
		--validation $(BINNING_VAL) \
		--target $(BINNING_TARGET) \
		--outdir $(BINNING_OUT_DIR) \
		--artifacts $(BINNING_ARTIFACTS_DIR) \
		--n-bins $(BINNING_N_BINS) \
		--output $(BINNING_OUTPUT) \
		--format $(BINNING_FORMAT) \
		$(BINNING_INCLUDE_CAT_OPT) $(BINNING_DROP_ORIG_OPT) $(BINNING_VARIABLES_OPT)
	@mkdir -p "$(REPORTS_BASE_DIR)/iv"
	@if [ -f "$(BINNING_ARTIFACTS_DIR)/binning_iv.csv" ]; then cp "$(BINNING_ARTIFACTS_DIR)/binning_iv.csv" "$(REPORTS_BASE_DIR)/iv/binning_iv.csv"; fi

# Appliquer un binning déjà appris à un nouveau dataset imputé
BINNING_APPLY_DATA ?= $(IMPUTE_OUT_DIR)/test.parquet
BINNING_MODEL      ?= $(BINNING_ARTIFACTS_DIR)/binning.joblib
BINNING_APPLY_OUT  ?= $(BINNING_OUT_DIR)/test.parquet

## binning_apply : applique le binning sauvegardé (binning.joblib) à un nouveau dataset → $(BINNING_APPLY_OUT)
.PHONY: binning_apply
binning_apply:
	$(BINNING_APPLY_CMD) \
		--data $(BINNING_APPLY_DATA) \
		--binner $(BINNING_MODEL) \
		--out $(BINNING_APPLY_OUT)

## binning : enchaîne impute puis binning_fit
.PHONY: binning
binning: impute binning_fit


# ============================================================================
# 4) MODEL – TRAIN / APPLY
# ============================================================================

MODEL_TRAIN_DATA  ?= $(BINNING_OUT_DIR)/train.parquet
MODEL_VAL_DATA    ?= $(BINNING_OUT_DIR)/validation.parquet
MODEL_TARGET      ?= $(IMPUTE_TARGET_COL)
MODEL_MIN_AUC     ?= 0.60
MODEL_DRIFT_BINS  ?= 10
MODEL_CALIBRATION ?= none           # none|sigmoid|isotonic
MODEL_FEATURES    ?=                # ex: "woe__credit_score,woe__original_dti"

MODEL_TRAIN_CMD ?= $(PY) src/train_model.py
MODEL_FEATURES_OPT := $(if $(strip $(MODEL_FEATURES)),--features "$(MODEL_FEATURES)",)

## model_train : entraîne + sélectionne le modèle le moins drifté, sauvegarde dans $(MODEL_ARTIFACTS_DIR)
.PHONY: model_train
model_train:
	$(MODEL_TRAIN_CMD) \
		--train $(MODEL_TRAIN_DATA) \
		--validation $(MODEL_VAL_DATA) \
		--target $(MODEL_TARGET) \
		--artifacts $(MODEL_ARTIFACTS_DIR) \
		--min-auc $(MODEL_MIN_AUC) \
		--drift-bins $(MODEL_DRIFT_BINS) \
		--calibration $(MODEL_CALIBRATION) \
		$(MODEL_FEATURES_OPT)
	@mkdir -p "$(REPORTS_BASE_DIR)/model"
	@if [ -f "$(MODEL_ARTIFACTS_DIR)/model_reports.csv" ]; then cp "$(MODEL_ARTIFACTS_DIR)/model_reports.csv" "$(REPORTS_BASE_DIR)/model/model_reports.csv"; fi

MODEL_APPLY_DATA ?= $(BINNING_OUT_DIR)/test.parquet
MODEL_PATH       ?= $(MODEL_ARTIFACTS_DIR)/model_best.joblib
MODEL_OUT        ?= $(SCORED_OUT_DIR)/test_scored.parquet
MODEL_APPLY_CMD  ?= $(PY) src/apply_model.py

## model_apply : applique le modèle sauvegardé (model_best.joblib) et écrit une colonne proba → $(MODEL_OUT)
.PHONY: model_apply
model_apply:
	$(MODEL_APPLY_CMD) \
		--data $(MODEL_APPLY_DATA) \
		--model $(MODEL_PATH) \
		--out $(MODEL_OUT)


# ============================================================================
# 5) PIPELINE(S)
# ============================================================================

## pipeline_from_labels : impute → binning_fit → model_train en réutilisant des labels existants
.PHONY: pipeline_from_labels
pipeline_from_labels: check_labels impute binning_fit model_train

# SKIP_LABELS=1 => ne (re)fait pas les labels
SKIP_LABELS ?= 0

ifeq ($(SKIP_LABELS),1)
PIPELINE_STEPS := pipeline_from_labels
else
PIPELINE_STEPS := labels impute binning_fit model_train
endif

## pipeline : exécute labels → impute → binning_fit → model_train (ou sans labels si SKIP_LABELS=1)
.PHONY: pipeline
pipeline: $(PIPELINE_STEPS)


# ============================================================================
# OUTILS / MAINTENANCE
# ============================================================================

## env : affiche les principaux paramètres/chemins
.PHONY: env
env:
	@echo "LABELS_OUTDIR=$(LABELS_OUTDIR)"
	@echo "SRC_LABELS_DIR=$(SRC_LABELS_DIR)"
	@echo "IMPUTE_OUT_DIR=$(IMPUTE_OUT_DIR)"
	@echo "BINNING_OUT_DIR=$(BINNING_OUT_DIR)"
	@echo "SCORED_OUT_DIR=$(SCORED_OUT_DIR)"
	@echo "IMPUTE_TRAIN_CSV=$(IMPUTE_TRAIN_CSV)"
	@echo "IMPUTE_VAL_CSV=$(IMPUTE_VAL_CSV)"
	@echo "IMPUTE_ARTIFACTS_DIR=$(IMPUTE_ARTIFACTS_DIR)"
	@echo "BINNING_ARTIFACTS_DIR=$(BINNING_ARTIFACTS_DIR)"
	@echo "MODEL_ARTIFACTS_DIR=$(MODEL_ARTIFACTS_DIR)"
	@echo "LABELS_WORKERS=$(LABELS_WORKERS)"
	@echo "LABELS_POOLED_UNTIL=$(LABELS_POOLED_UNTIL)"

## clean_data : supprime les jeux dérivés (labels, imputed, binned, scored)
.PHONY: clean_data
clean_data:
	@rm -rf "$(LABELS_OUTDIR)" "$(IMPUTE_OUT_DIR)" "$(BINNING_OUT_DIR)" "$(SCORED_OUT_DIR)" || true
	@echo "→ Données dérivées supprimées."

## clean_artifacts : supprime les artefacts (imputer, binning, model)
.PHONY: clean_artifacts
clean_artifacts:
	@rm -rf "$(IMPUTE_ARTIFACTS_DIR)" "$(BINNING_ARTIFACTS_DIR)" "$(MODEL_ARTIFACTS_DIR)" || true
	@echo "→ Artefacts supprimés."

## clean_reports : supprime les rapports
.PHONY: clean_reports
clean_reports:
	@rm -rf "$(REPORTS_BASE_DIR)/iv" "$(REPORTS_BASE_DIR)/model" || true
	@echo "→ Rapports supprimés."

## clean_all : supprime toutes les sorties (données, artefacts, rapports)
.PHONY: clean_all
clean_all: clean_data clean_artifacts clean_reports

## print-% : debug d'une variable Make (ex: make print-LABELS_OUTDIR)
print-%:
	@echo '$*=$($*)'
