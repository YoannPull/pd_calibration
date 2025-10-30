# ============================================================================
# Makefile – Pipeline end-to-end : splits → impute → binning → train → apply
# Dépend de Poetry + Python. La cible par défaut affiche l'aide.
# Usage rapide :
#   make help
#   make splits
#   make impute
#   make binning_fit
#   make binning_apply
#   make model_train
#   make model_apply
#   make pipeline          # impute → binning_fit → model_train
# ============================================================================

.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

# Commande Python unifiée (évite de la répéter partout)
PY ?= PYTHONPATH=src poetry run python

# Répertoire commun pour sauvegarder les artefacts (imputer, binning, modèle)
ARTIFACTS_DIR ?= artifacts


# ----------------------------------------------------------------------------
# HELP
# ----------------------------------------------------------------------------
## help : affiche cette aide
.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mCibles disponibles\033[0m\n\n"} /^[a-zA-Z0-9_%-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^# Exemple/ {print} ' $(MAKEFILE_LIST)
	@printf "\n\033[1mExemples\033[0m\n"
	@echo "  make binning BINNING_INCLUDE_CAT=1 BINNING_N_BINS=8 BINNING_OUTPUT=woe"
	@echo "  make model_train MODEL_CALIBRATION=isotonic MODEL_MIN_AUC=0.65"
	@echo "  make pipeline"


# ============================================================================
# 1) SPLITS (génération des jeux avec imputation intégrée au script splits)
# ============================================================================

SPLITS_WINDOW        ?= 24
SPLITS_IN_DIR        ?= data/processed
SPLITS_SUBDIR        ?= default_labels
SPLITS_OUT_DIR       ?= data/processed/default_labels_imputed
SPLITS_FORMAT        ?= parquet
SPLITS_IMPUT_COHORT  ?= 1
SPLITS_TRAIN_RANGE   ?= 2018Q1:2021Q4
SPLITS_VAL_RANGE     ?= 2022Q1:2022Q4
SPLITS_TEST_RANGE    ?= 2023Q1:

SPLITS_CMD ?= $(PY) src/make_splits_impute.py
SPLITS_COHORT_OPT := $(if $(filter $(SPLITS_IMPUT_COHORT),1),--imput_cohort,)

## splits : crée les splits (train/val/test) avec imputation intégrée au script
.PHONY: splits
splits:
	$(SPLITS_CMD) \
		--window $(SPLITS_WINDOW) \
		--in_dir $(SPLITS_IN_DIR) --subdir $(SPLITS_SUBDIR) \
		--out_dir $(SPLITS_OUT_DIR) \
		--format $(SPLITS_FORMAT) $(SPLITS_COHORT_OPT) \
		--train_range "$(SPLITS_TRAIN_RANGE)" \
		--val_range   "$(SPLITS_VAL_RANGE)" \
		--test_range  "$(SPLITS_TEST_RANGE)"


# ============================================================================
# 2) IMPUTATION (impute_and_save.py) – séparé des variables SPLITS
# ============================================================================

IMPUTE_TRAIN_CSV     ?= data/processed/merged/non_imputed/train.parquet
IMPUTE_VAL_CSV       ?= data/processed/merged/non_imputed/validation.parquet
IMPUTE_TARGET_COL    ?= default_24m
IMPUTE_OUT_DIR       ?= data/processed/merged/imputed
IMPUTE_FORMAT        ?= parquet
IMPUTE_ARTIFACTS_DIR ?= $(ARTIFACTS_DIR)
IMPUTE_COHORT        ?= 1       # 1 = imputation par cohortes
IMPUTE_MISSING_FLAG  ?= 1       # 1 = ajoute les colonnes was_missing_*

IMPUTE_CMD ?= $(PY) src/impute_and_save.py
IMPUTE_COHORT_OPT  := $(if $(filter $(IMPUTE_COHORT),1),--use-cohort,)
IMPUTE_MISSING_OPT := $(if $(filter $(IMPUTE_MISSING_FLAG),1),--missing-flag,)

## impute : impute train/validation et sauvegarde (parquet/csv) + imputer.joblib
.PHONY: impute
impute:
	$(IMPUTE_CMD) \
		--train-csv $(IMPUTE_TRAIN_CSV) \
		--validation-csv $(IMPUTE_VAL_CSV) \
		--target $(IMPUTE_TARGET_COL) \
		--outdir $(IMPUTE_OUT_DIR) \
		--artifacts $(IMPUTE_ARTIFACTS_DIR) \
		--format $(IMPUTE_FORMAT) \
		$(IMPUTE_COHORT_OPT) $(IMPUTE_MISSING_OPT)


# ============================================================================
# 3) BINNING (fit_binning.py / apply_binning.py)
# ============================================================================

# Entrées par défaut : réutilise les sorties de l’imputation
BINNING_TRAIN           ?= $(IMPUTE_OUT_DIR)/train.parquet
BINNING_VAL             ?= $(IMPUTE_OUT_DIR)/validation.parquet
BINNING_TARGET          ?= $(IMPUTE_TARGET_COL)
BINNING_OUT_DIR         ?= data/processed/merged/binned
BINNING_ARTIFACTS_DIR   ?= $(ARTIFACTS_DIR)
BINNING_FORMAT          ?= parquet          # parquet | csv
BINNING_N_BINS          ?= 10
BINNING_OUTPUT          ?= woe              # woe | bin_index | both
BINNING_INCLUDE_CAT     ?= 0                # 1 = inclure catégorielles
BINNING_DROP_ORIGINAL   ?= 0                # 1 = ne garder que les features encodées
BINNING_VARIABLES       ?=                  # "var1,var2,..."

BINNING_INCLUDE_CAT_OPT := $(if $(filter $(BINNING_INCLUDE_CAT),1),--include-categorical,)
BINNING_DROP_ORIG_OPT   := $(if $(filter $(BINNING_DROP_ORIGINAL),1),--drop-original,)
BINNING_VARIABLES_OPT   := $(if $(strip $(BINNING_VARIABLES)),--variables "$(BINNING_VARIABLES)",)

BINNING_FIT_CMD   ?= $(PY) src/fit_binning.py
BINNING_APPLY_CMD ?= $(PY) src/apply_binning.py

## binning_fit : fit du binning (WOE/IV) sur train, apply sur val, sauvegarde artifacts
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

# Appliquer un binning déjà appris à un nouveau dataset imputé
BINNING_APPLY_DATA ?= $(IMPUTE_OUT_DIR)/test.parquet
BINNING_MODEL      ?= $(BINNING_ARTIFACTS_DIR)/binning.joblib
BINNING_APPLY_OUT  ?= $(BINNING_OUT_DIR)/test.parquet

## binning_apply : applique le binning sauvegardé (binning.joblib) à un nouveau dataset
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

MODEL_TRAIN_DATA    ?= $(BINNING_OUT_DIR)/train.parquet
MODEL_VAL_DATA      ?= $(BINNING_OUT_DIR)/validation.parquet
MODEL_TARGET        ?= $(IMPUTE_TARGET_COL)
MODEL_ARTIFACTS_DIR ?= $(ARTIFACTS_DIR)
MODEL_MIN_AUC       ?= 0.60
MODEL_DRIFT_BINS    ?= 10
MODEL_CALIBRATION   ?= none           # none|sigmoid|isotonic
MODEL_FEATURES      ?=                # ex: "woe__credit_score,woe__original_dti"

MODEL_TRAIN_CMD ?= $(PY) src/train_model.py
MODEL_FEATURES_OPT := $(if $(strip $(MODEL_FEATURES)),--features "$(MODEL_FEATURES)",)

## model_train : entraîne plusieurs modèles, sélectionne le moins de drift, sauvegarde model_best.joblib
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

MODEL_APPLY_DATA ?= $(BINNING_OUT_DIR)/test.parquet
MODEL_PATH       ?= $(MODEL_ARTIFACTS_DIR)/model_best.joblib
MODEL_OUT        ?= $(BINNING_OUT_DIR)/test_scored.parquet
MODEL_APPLY_CMD  ?= $(PY) src/apply_model.py

## model_apply : applique le modèle sauvegardé (model_best.joblib) et écrit une colonne proba
.PHONY: model_apply
model_apply:
	$(MODEL_APPLY_CMD) \
		--data $(MODEL_APPLY_DATA) \
		--model $(MODEL_PATH) \
		--out $(MODEL_OUT)


# ============================================================================
# 5) PIPELINE RACCOURCI
# ============================================================================

## pipeline : exécute impute → binning_fit → model_train (chemin classique)
.PHONY: pipeline
pipeline: impute binning_fit model_train


# ============================================================================
# OUTILS / DEBUG
# ============================================================================

## env : affiche les principaux paramètres actuels
.PHONY: env
env:
	@echo "ARTIFACTS_DIR=$(ARTIFACTS_DIR)"
	@echo "IMPUTE_OUT_DIR=$(IMPUTE_OUT_DIR)"
	@echo "BINNING_OUT_DIR=$(BINNING_OUT_DIR)"
	@echo "MODEL_TRAIN_DATA=$(MODEL_TRAIN_DATA)"
	@echo "MODEL_VAL_DATA=$(MODEL_VAL_DATA)"

## print-% : debug d'une variable Make (ex: make print-IMPUTE_OUT_DIR)
print-%:
	@echo '$*=$($*)'

# Exemple :
# make binning BINNING_INCLUDE_CAT=1 BINNING_N_BINS=8 BINNING_OUTPUT=woe
# make binning_apply BINNING_APPLY_DATA=data/processed/merged/imputed/new_batch.parquet BINNING_APPLY_OUT=data/processed/merged/binned/new_batch.parquet
# make model_train MODEL_CALIBRATION=isotonic MODEL_MIN_AUC=0.65
# make model_apply MODEL_APPLY_DATA=data/processed/merged/binned/new_batch.parquet
