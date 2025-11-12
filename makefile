# ============================================================================
# Makefile – Pipeline : labels → impute → binning → train (from binned) → apply → results
# + oos_score (imputer → binning → model sur OOS)
# (sorties fixes et stables, sans horodatage)
#
# Usage :
#   make help
#   make labels
#   make impute                       # (par défaut: lit window/_splits.json)
#   make impute IMPUTE_FROM_SPLITS=0 VAL_QUARTER=2022Q4   # mode rétro (quarter explicite)
#   make binning_fit
#   make model_train                  # entraîne depuis WOE/__BIN avec calibration isotonic
#   make model_apply                  # applique proba uniquement
#   make model_apply_segment          # applique proba + segmentation (si apply_model.py le supporte)
#   make oos_score                    # applique imputer → binning → modèle sur OOS (avec artefacts du training)
#   make results                      # génère un rapport vintage × grades
#   make pipeline
#   make pipeline SKIP_LABELS=1
# ============================================================================

.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

# (Optionnel) Limiter les threads BLAS
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
export NUMEXPR_NUM_THREADS ?= 1

# Commande Python unifiée
PY ?= PYTHONPATH=src poetry run python

# Répertoires (fixes)
DATA_BASE_DIR      := data/processed
ARTIFACTS_BASE_DIR := artifacts
REPORTS_BASE_DIR   := reports

# Dossiers de données par étape
LABELS_OUTDIR   ?= $(DATA_BASE_DIR)/default_labels
IMPUTE_OUT_DIR  ?= $(DATA_BASE_DIR)/imputed
BINNING_OUT_DIR ?= $(DATA_BASE_DIR)/binned
SCORED_OUT_DIR  ?= $(DATA_BASE_DIR)/scored

# Dossiers d'artefacts par étape
IMPUTE_ARTIFACTS_DIR  ?= $(ARTIFACTS_BASE_DIR)/imputer
BINNING_ARTIFACTS_DIR ?= $(ARTIFACTS_BASE_DIR)/binning_maxgini
MODEL_ARTIFACTS_DIR   ?= $(ARTIFACTS_BASE_DIR)/model_from_binned

# ---------------------------------------------------------------------------
# PARAMÈTRES (définir AVANT sanitize)
# ---------------------------------------------------------------------------
# LABELS (tout est désormais lu dans le YAML ; seules options utiles ici : chemin du YAML + workers)
LABELS_CONFIG ?= config.yml
LABELS_WORKERS ?=          # ex: 6

# Ces variables restent utilisées par les étapes suivantes
LABELS_WINDOW ?= 24
LABELS_FORMAT ?= parquet
SRC_LABELS_DIR ?= $(LABELS_OUTDIR)

# === IMPUTATION ===
# Nouveau: utiliser par défaut les splits (window/_splits.json)
IMPUTE_FROM_SPLITS ?= 1        # 1 = utilise --labels-window-dir --use-splits ; 0 = mode rétro
VAL_QUARTER        ?= 2022Q4   # utilisé seulement si IMPUTE_FROM_SPLITS=0
IMPUTE_VAL_SOURCE  ?= quarter  # quarter | oos (rétro-compat si IMPUTE_FROM_SPLITS=0)

# --------- Options BINNING (fit/apply) — version max |Gini|
BINNING_TRAIN          ?= $(IMPUTE_OUT_DIR)/train.parquet
BINNING_VAL            ?= $(IMPUTE_OUT_DIR)/validation.parquet
BINNING_TARGET         ?= default_$(LABELS_WINDOW)m
BINNING_FORMAT         ?= parquet
BIN_INCLUDE_MISSING    ?= 1
BIN_MISSING_LABEL      ?= __MISSING__
BIN_COL_SUFFIX         ?= __BIN
MAX_BINS_CATEG         ?= 6
MIN_BIN_SIZE_CATEG     ?= 200
MAX_BINS_NUM           ?= 6
MIN_BIN_SIZE_NUM       ?= 200
N_QUANTILES_NUM        ?= 50
MIN_GINI_KEEP          ?=
BIN_DROP_MISSING_FLAGS ?= 1   # <— robuste : ne repose pas sur flags manquants
BIN_NO_DENYLIST        ?= 0
N_JOBS_CATEG           ?= -1
N_JOBS_NUM             ?= -1

# --------- Options TRAIN-FROM-BINNED
WOE_PREFIXES           ?= woe__
BIN_SUFFIX             ?= __BIN
CV_FOLDS               ?= 5
CORR_THRESHOLD         ?= 0.85
NO_ISOTONIC            ?= 0          # 1 -> désactive l’isotonic, utilise sigmoid
DROP_PROXY_CUTOFF      ?=
CONDITIONAL_PROXIES    ?=
NO_PRIOR_SHIFT_ADJUST  ?= 0
# Ablation
ABLATION_MAX_STEPS     ?= 10
ABLATION_MAX_AUC_LOSS  ?= 0.02
# Timing
MODEL_TIMING           ?= 0          # 1 -> --timing (sauve timings.json + recap final)
MODEL_TIMING_LIVE      ?= 0          # 1 -> --timing-live (logs live par section)

# Chemin des buckets pour l’apply segmenté
BUCKETS_PATH ?= $(MODEL_ARTIFACTS_DIR)/risk_buckets.json
SEGMENT_COL  ?= risk_bucket

# ------------------- results.py (rapport vintage × grade) -------------------
RESULTS_CMD         ?= $(PY) src/results.py
RESULTS_DATA        ?= $(SCORED_OUT_DIR)/test_scored.parquet
RESULTS_BUCKETS     ?= $(BUCKETS_PATH)
RESULTS_TARGET      ?=
RESULTS_PROBA_COL   ?= proba
RESULTS_VINTAGE_COL ?= vintage
RESULTS_OUT         ?= $(REPORTS_BASE_DIR)/by_vintage_grades.csv

# --- Sanitize ---
LABELS_WINDOW        := $(strip $(LABELS_WINDOW))
SRC_LABELS_DIR       := $(strip $(SRC_LABELS_DIR))
LABELS_FORMAT        := $(strip $(LABELS_FORMAT))
LABELS_WORKERS       := $(strip $(LABELS_WORKERS))
IMPUTE_FROM_SPLITS   := $(strip $(IMPUTE_FROM_SPLITS))
IMPUTE_VAL_SOURCE    := $(strip $(IMPUTE_VAL_SOURCE))
VAL_QUARTER          := $(strip $(VAL_QUARTER))

BINNING_TARGET       := $(strip $(BINNING_TARGET))

WOE_PREFIXES         := $(strip $(WOE_PREFIXES))
BIN_SUFFIX           := $(strip $(BIN_SUFFIX))
CV_FOLDS             := $(strip $(CV_FOLDS))
CORR_THRESHOLD       := $(strip $(CORR_THRESHOLD))
NO_ISOTONIC          := $(strip $(NO_ISOTONIC))
DROP_PROXY_CUTOFF    := $(strip $(DROP_PROXY_CUTOFF))
CONDITIONAL_PROXIES  := $(strip $(CONDITIONAL_PROXIES))
NO_PRIOR_SHIFT_ADJUST:= $(strip $(NO_PRIOR_SHIFT_ADJUST))
ABLATION_MAX_STEPS   := $(strip $(ABLATION_MAX_STEPS))
ABLATION_MAX_AUC_LOSS:= $(strip $(ABLATION_MAX_AUC_LOSS))
MODEL_TIMING         := $(strip $(MODEL_TIMING))
MODEL_TIMING_LIVE    := $(strip $(MODEL_TIMING_LIVE))

RESULTS_DATA         := $(strip $(RESULTS_DATA))
RESULTS_BUCKETS      := $(strip $(RESULTS_BUCKETS))
RESULTS_TARGET       := $(strip $(RESULTS_TARGET))
RESULTS_PROBA_COL    := $(strip $(RESULTS_PROBA_COL))
RESULTS_VINTAGE_COL  := $(strip $(RESULTS_VINTAGE_COL))
RESULTS_OUT          := $(strip $(RESULTS_OUT))

# Dérivés
LABELS_WINDOW_DIR    := $(SRC_LABELS_DIR)/window=$(LABELS_WINDOW)m

# Flags dérivés binning (max |Gini|)
BIN_INCLUDE_MISSING_OPT := $(if $(filter $(BIN_INCLUDE_MISSING),1),--include-missing,)
MIN_GINI_KEEP_OPT       := $(if $(MIN_GINI_KEEP),--min-gini-keep $(MIN_GINI_KEEP),)
BIN_DROP_MISSING_FLAGS_OPT := $(if $(filter $(BIN_DROP_MISSING_FLAGS),1),--drop-missing-flags,)
BIN_NO_DENYLIST_OPT        := $(if $(filter $(BIN_NO_DENYLIST),1),--no-denylist,)
N_JOBS_CATEG_OPT           := $(if $(N_JOBS_CATEG),--n-jobs-categ $(N_JOBS_CATEG),)
N_JOBS_NUM_OPT             := $(if $(N_JOBS_NUM),--n-jobs-num $(N_JOBS_NUM),)

# Flags dérivés train_from_binned
NO_ISOTONIC_OPT          := $(if $(filter $(NO_ISOTONIC),1),--no-isotonic,)
DROP_PROXY_OPT           := $(if $(DROP_PROXY_CUTOFF),--drop-proxy-cutoff $(DROP_PROXY_CUTOFF),)
CONDITIONAL_PROXIES_OPT  := $(if $(CONDITIONAL_PROXIES),--conditional-proxies "$(CONDITIONAL_PROXIES)",)
NO_PRIOR_SHIFT_OPT       := $(if $(filter $(NO_PRIOR_SHIFT_ADJUST),1),--no-prior-shift-adjust,)
ABLATION_MAX_STEPS_OPT   := $(if $(ABLATION_MAX_STEPS),--ablation-max-steps $(ABLATION_MAX_STEPS),)
ABLATION_MAX_AUC_LOSS_OPT:= $(if $(ABLATION_MAX_AUC_LOSS),--ablation-max-auc-loss $(ABLATION_MAX_AUC_LOSS),)
MODEL_TIMING_OPT         := $(if $(filter $(MODEL_TIMING),1),--timing,)
MODEL_TIMING_LIVE_OPT    := $(if $(filter $(MODEL_TIMING_LIVE),1),--timing-live,)

# ============================================================================
# HELP
# ============================================================================
## help : affiche cette aide
.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mCibles disponibles\033[0m\n\n"} /^[a-zA-Z0-9_%-]+:.*##/ { printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\n\033[1mExemples\033[0m\n"
	@echo "  make labels LABELS_WORKERS=6"
	@echo "  make impute                                  # utilise window/_splits.json"
	@echo "  make impute_from_splits                      # utilise _splits.json (validation_quarters)"
	@echo "  make impute_val VAL_QUARTER=2024Q3           # mode rétro: validation=quarter explicite"
	@echo "  make binning_fit BIN_INCLUDE_MISSING=1 MAX_BINS_NUM=8 N_QUANTILES_NUM=80 N_JOBS_CATEG=-1 N_JOBS_NUM=-1"
	@echo "  make binning_fit BIN_DROP_MISSING_FLAGS=1"
	@echo "  make binning_fit BIN_NO_DENYLIST=1   # pour désactiver la denylist stricte"
	@echo "  make model_train WOE_PREFIXES=\"woe__,_WOE\""
	@echo "  make model_train WOE_PREFIXES=\"\" DROP_PROXY_CUTOFF=0.25 CONDITIONAL_PROXIES=\"original_interest_rate\""
	@echo "  make model_train MODEL_TIMING=1 MODEL_TIMING_LIVE=1  # timings live + fichier"
	@echo "  make oos_score                    # inférence complète sur OOS avec artefacts du training"
	@echo "  make pipeline SKIP_LABELS=1"

# ============================================================================
# 1) LABELS
# ============================================================================
LABELS_CMD ?= $(PY) src/make_labels.py
LABELS_WORKERS_OPT := $(if $(LABELS_WORKERS),--workers $(LABELS_WORKERS),)

## labels : génère les labels selon le YAML (splits explicites) → $(LABELS_OUTDIR)
.PHONY: labels
labels:
	$(LABELS_CMD) --config $(LABELS_CONFIG) $(LABELS_WORKERS_OPT)

# ============================================================================
# 2) IMPUTATION
# ============================================================================
IMPUTE_CMD ?= $(PY) src/impute_and_save.py

# Chemins rétro-compat si IMPUTE_FROM_SPLITS=0
IMPUTE_TRAIN_CSV       ?= $(SRC_LABELS_DIR)/window=$(LABELS_WINDOW)m/pooled.$(LABELS_FORMAT)
IMPUTE_VAL_CSV_QUARTER := $(SRC_LABELS_DIR)/window=$(LABELS_WINDOW)m/quarter=$(VAL_QUARTER)/data.$(LABELS_FORMAT)
IMPUTE_VAL_CSV_OOS     := $(SRC_LABELS_DIR)/window=$(LABELS_WINDOW)m/oos.$(LABELS_FORMAT)
ifeq ($(IMPUTE_VAL_SOURCE),oos)
  IMPUTE_VAL_CSV := $(IMPUTE_VAL_CSV_OOS)
else
  IMPUTE_VAL_CSV := $(IMPUTE_VAL_CSV_QUARTER)
endif

IMPUTE_TARGET_COL   ?= default_$(LABELS_WINDOW)m
IMPUTE_FORMAT       ?= parquet
IMPUTE_COHORT       ?= 1

# Par défaut: pas de flags was_missing_* en imputation
IMPUTE_MISSING_FLAG ?= 0

IMPUTE_COHORT_OPT  := $(if $(filter $(IMPUTE_COHORT),1),--use-cohort,)
IMPUTE_MISSING_OPT := $(if $(filter $(IMPUTE_MISSING_FLAG),1),--missing-flag,)

## check_labels : vérifie que les fichiers labels source existent
.PHONY: check_labels
check_labels:
ifeq ($(IMPUTE_FROM_SPLITS),1)
	@test -d "$(LABELS_WINDOW_DIR)" || (echo "[ERR] Manque dossier: $(LABELS_WINDOW_DIR)"; exit 1)
	@test -f "$(LABELS_WINDOW_DIR)/pooled.$(LABELS_FORMAT)" || (echo "[ERR] Manque pooled: $(LABELS_WINDOW_DIR)/pooled.$(LABELS_FORMAT)"; exit 1)
	@test -f "$(LABELS_WINDOW_DIR)/_splits.json" || (echo "[ERR] Manque splits: $(LABELS_WINDOW_DIR)/_splits.json"; exit 1)
else
	@test -f "$(IMPUTE_TRAIN_CSV)" || (echo "[ERR] Manque: $(IMPUTE_TRAIN_CSV)"; exit 1)
	@test -f "$(IMPUTE_VAL_CSV)"   || (echo "[ERR] Manque: $(IMPUTE_VAL_CSV)"; exit 1)
endif

## impute : impute train/validation → $(IMPUTE_OUT_DIR) + artifacts → $(IMPUTE_ARTIFACTS_DIR)
.PHONY: impute
impute: check_labels
ifeq ($(IMPUTE_FROM_SPLITS),1)
	$(IMPUTE_CMD) \
		--labels-window-dir $(LABELS_WINDOW_DIR) \
		--use-splits \
		--target $(IMPUTE_TARGET_COL) \
		--outdir $(IMPUTE_OUT_DIR) \
		--artifacts $(IMPUTE_ARTIFACTS_DIR) \
		--format $(IMPUTE_FORMAT) \
		$(IMPUTE_COHORT_OPT) $(IMPUTE_MISSING_OPT)
else
	$(IMPUTE_CMD) \
		--train-csv $(IMPUTE_TRAIN_CSV) \
		--validation-csv $(IMPUTE_VAL_CSV) \
		--target $(IMPUTE_TARGET_COL) \
		--outdir $(IMPUTE_OUT_DIR) \
		--artifacts $(IMPUTE_ARTIFACTS_DIR) \
		--format $(IMPUTE_FORMAT) \
		$(IMPUTE_COHORT_OPT) $(IMPUTE_MISSING_OPT)
endif

# Raccourcis pratiques
## impute_from_splits : imputation en lisant window/_splits.json (validation_quarters)
.PHONY: impute_from_splits
impute_from_splits:
	$(MAKE) impute IMPUTE_FROM_SPLITS=1

## impute_val : imputation rétro avec un quarter explicite (ex: VAL_QUARTER=2024Q3)
.PHONY: impute_val
impute_val:
	$(MAKE) impute IMPUTE_FROM_SPLITS=0 IMPUTE_VAL_SOURCE=quarter VAL_QUARTER=$(VAL_QUARTER)

# ============================================================================
# 3) BINNING (fit/apply) – max |Gini|
# ============================================================================
BINNING_FIT_CMD   ?= $(PY) src/fit_binning.py
BINNING_APPLY_CMD ?= $(PY) src/apply_binning.py

## binning_fit : fit max |Gini| sur train, apply sur val → $(BINNING_OUT_DIR) + bins.json
.PHONY: binning_fit
binning_fit:
	$(BINNING_FIT_CMD) \
		--train $(BINNING_TRAIN) \
		--validation $(BINNING_VAL) \
		--target $(BINNING_TARGET) \
		--outdir $(BINNING_OUT_DIR) \
		--artifacts $(BINNING_ARTIFACTS_DIR) \
		--format $(BINNING_FORMAT) \
		--bin-col-suffix "$(BIN_COL_SUFFIX)" \
		--max-bins-categ $(MAX_BINS_CATEG) \
		--min-bin-size-categ $(MIN_BIN_SIZE_CATEG) \
		--max-bins-num $(MAX_BINS_NUM) \
		--min-bin-size-num $(MIN_BIN_SIZE_NUM) \
		--n-quantiles-num $(N_QUANTILES_NUM) \
		--missing-label "$(BIN_MISSING_LABEL)" \
		$(BIN_INCLUDE_MISSING_OPT) \
		$(MIN_GINI_KEEP_OPT) \
		$(BIN_DROP_MISSING_FLAGS_OPT) \
		$(BIN_NO_DENYLIST_OPT) \
		$(N_JOBS_CATEG_OPT) \
		$(N_JOBS_NUM_OPT)
	@mkdir -p "$(REPORTS_BASE_DIR)/iv" ; true

# Appliquer un binning déjà appris à un nouveau dataset imputé
BINNING_APPLY_DATA ?= $(IMPUTE_OUT_DIR)/test.parquet
BINNING_MODEL      ?= $(BINNING_ARTIFACTS_DIR)/bins.json
BINNING_APPLY_OUT  ?= $(BINNING_OUT_DIR)/test.parquet

## binning_apply : applique les bins (bins.json) à un nouveau dataset → $(BINNING_APPLY_OUT)
.PHONY: binning_apply
binning_apply:
	$(BINNING_APPLY_CMD) \
		--data $(BINNING_APPLY_DATA) \
		--bins $(BINNING_MODEL) \
		--out $(BINNING_APPLY_OUT) \
		--bin-col-suffix "$(BIN_COL_SUFFIX)"

## binning : enchaîne impute puis binning_fit
.PHONY: binning
binning: impute binning_fit

# ============================================================================
# 4) MODEL – TRAIN / APPLY
# ============================================================================
MODEL_TRAIN_CMD ?= $(PY) src/train_model.py
MODEL_APPLY_CMD ?= $(PY) src/apply_model.py

MODEL_TRAIN_DATA ?= $(BINNING_OUT_DIR)/train.parquet
MODEL_VAL_DATA   ?= $(BINNING_OUT_DIR)/validation.parquet
MODEL_TARGET     ?= $(IMPUTE_TARGET_COL)

## model_train : entraîne depuis WOE/__BIN + calibration isotonic → $(MODEL_ARTIFACTS_DIR)
.PHONY: model_train
model_train:
	$(MODEL_TRAIN_CMD) \
		--train $(MODEL_TRAIN_DATA) \
		--validation $(MODEL_VAL_DATA) \
		--target $(MODEL_TARGET) \
		--artifacts $(MODEL_ARTIFACTS_DIR) \
		--bin-suffix "$(BIN_SUFFIX)" \
		--woe-prefixes "$(WOE_PREFIXES)" \
		--cv-folds $(CV_FOLDS) \
		--corr-threshold $(CORR_THRESHOLD) \
		$(NO_ISOTONIC_OPT) \
		$(DROP_PROXY_OPT) \
		$(CONDITIONAL_PROXIES_OPT) \
		$(NO_PRIOR_SHIFT_OPT) \
		$(ABLATION_MAX_STEPS_OPT) \
		$(ABLATION_MAX_AUC_LOSS_OPT) \
		$(MODEL_TIMING_OPT) \
		$(MODEL_TIMING_LIVE_OPT)
	@mkdir -p "$(REPORTS_BASE_DIR)/model"
	@if [ -f "$(MODEL_ARTIFACTS_DIR)/reports.csv" ]; then cp "$(MODEL_ARTIFACTS_DIR)/reports.csv" "$(REPORTS_BASE_DIR)/model/reports.csv"; fi
	@if [ -f "$(MODEL_ARTIFACTS_DIR)/importance.csv" ]; then cp "$(MODEL_ARTIFACTS_DIR)/model/importance.csv" "$(REPORTS_BASE_DIR)/model/importance.csv"; fi || true

MODEL_APPLY_DATA ?= $(BINNING_OUT_DIR)/test.parquet
MODEL_PATH       ?= $(MODEL_ARTIFACTS_DIR)/model_best.joblib
MODEL_OUT        ?= $(SCORED_OUT_DIR)/test_scored.parquet

## model_apply : applique le modèle sauvegardé (proba seule) → $(MODEL_OUT)
.PHONY: model_apply
model_apply:
	$(MODEL_APPLY_CMD) \
		--data $(MODEL_APPLY_DATA) \
		--model $(MODEL_PATH) \
		--out $(MODEL_OUT) \
		--bin-suffix "$(BIN_SUFFIX)"

## model_apply_segment : applique le modèle + segmentation (si supporté)
.PHONY: model_apply_segment
model_apply_segment:
	$(MODEL_APPLY_CMD) \
		--data $(MODEL_APPLY_DATA) \
		--model $(MODEL_PATH) \
		--out $(MODEL_OUT) \
		--buckets $(BUCKETS_PATH) \
		--bucket-col $(SEGMENT_COL) \
		--bin-suffix "$(BIN_SUFFIX)"

# ============================================================================
# 5) RESULTS (rapport vintage × grade)
# ============================================================================
## results : génère un rapport par vintage/grade depuis un fichier scoré
.PHONY: results
results:
	$(RESULTS_CMD) \
		--data $(RESULTS_DATA) \
		--buckets $(RESULTS_BUCKETS) \
		--proba-col $(RESULTS_PROBA_COL) \
		--vintage-col $(RESULTS_VINTAGE_COL) \
		$(if $(RESULTS_TARGET),--target $(RESULTS_TARGET),) \
		--out $(RESULTS_OUT)

# ============================================================================
# 6) PIPELINE(S)
# ============================================================================
## pipeline_from_labels : impute → binning_fit → model_train
.PHONY: pipeline_from_labels
pipeline_from_labels: check_labels impute binning_fit model_train

SKIP_LABELS ?= 0
ifeq ($(SKIP_LABELS),1)
  PIPELINE_STEPS := pipeline_from_labels
else
  PIPELINE_STEPS := labels impute binning_fit model_train
endif

## pipeline : labels → impute → binning_fit → model_train (ou sans labels si SKIP_LABELS=1)
.PHONY: pipeline
pipeline: $(PIPELINE_STEPS)

# ============================================================================
# 7) OOS SCORE (imputer → binning → model, avec les artefacts du training)
# ============================================================================
# Chemins OOS (surchargables au besoin)
OOS_LABELS_PARQUET ?= $(SRC_LABELS_DIR)/window=$(LABELS_WINDOW)m/oos.$(LABELS_FORMAT)
OOS_IMPUTED        ?= $(IMPUTE_OUT_DIR)/oos.parquet
OOS_BINNED         ?= $(BINNING_OUT_DIR)/oos.parquet
OOS_SCORED         ?= $(SCORED_OUT_DIR)/oos_scored.parquet
OOS_METRICS_OUT    ?= $(REPORTS_BASE_DIR)/model/oos_metrics.json

# Artefacts du training (déjà produits par impute / binning_fit / model_train)
IMPUTER_PATH ?= $(IMPUTE_ARTIFACTS_DIR)/imputer.joblib
IMPUTER_META ?= $(IMPUTE_ARTIFACTS_DIR)/imputer_meta.json
BINS_JSON    ?= $(BINNING_ARTIFACTS_DIR)/bins.json
MODEL_PATH   ?= $(MODEL_ARTIFACTS_DIR)/model_best.joblib

# OOS: par défaut on NE drop PAS les extras (pour ne rien perdre) ; on droppe les flags NA si présents
OOS_IMPUTE_DROP_EXTRAS ?= 0
OOS_IMPUTE_DROP_FLAGS  ?= 1

OOS_IMPUTE_DROP_FLAGS_OPT := $(if $(filter $(OOS_IMPUTE_DROP_FLAGS),1),--drop-missing-flags,)

## check_trained : vérifie que le modèle et ses artefacts existent
.PHONY: check_trained
check_trained:
	@test -f "$(IMPUTER_PATH)" || (echo "[ERR] Imputer manquant: $(IMPUTER_PATH) (run: make impute)"; exit 1)
	@test -f "$(IMPUTER_META)" || (echo "[ERR] imputer_meta manquant: $(IMPUTER_META) (run: make impute)"; exit 1)
	@test -f "$(BINS_JSON)"    || (echo "[ERR] bins.json manquant: $(BINS_JSON) (run: make binning_fit)"; exit 1)
	@test -f "$(MODEL_PATH)"   || (echo "[ERR] Modèle manquant: $(MODEL_PATH) (run: make model_train)"; exit 1)
	@test -f "$(BUCKETS_PATH)" || (echo "[ERR] Buckets manquants: $(BUCKETS_PATH) (run: make model_train)"; exit 1)

## oos_score : applique imputer + binning + modèle sur OOS, avec buckets & métriques
.PHONY: oos_score
oos_score: check_trained
	@test -f "$(OOS_LABELS_PARQUET)" || (echo "[ERR] Manque OOS labels: $(OOS_LABELS_PARQUET)"; exit 1)
	$(PY) src/apply_imputer.py \
		--imputer $(IMPUTER_PATH) \
		--data $(OOS_LABELS_PARQUET) \
		--out $(OOS_IMPUTED) \
		--meta $(IMPUTER_META) \
		$(OOS_IMPUTE_DROP_FLAGS_OPT)
	$(PY) src/apply_binning.py \
		--data $(OOS_IMPUTED) \
		--bins $(BINS_JSON) \
		--out $(OOS_BINNED) \
		--bin-col-suffix "$(BIN_COL_SUFFIX)"
	$(MODEL_APPLY_CMD) \
		--data $(OOS_BINNED) \
		--model $(MODEL_PATH) \
		--out $(OOS_SCORED) \
		--buckets $(BUCKETS_PATH) \
		--bucket-col $(SEGMENT_COL) \
		--bin-suffix "$(BIN_SUFFIX)" \
		$(if $(OOS_METRICS_OUT),--metrics-out $(OOS_METRICS_OUT),)
	@echo "✔ OOS scored → $(OOS_SCORED)"
	@echo "  (metrics: $(OOS_METRICS_OUT))"

# ============================================================================
# OUTILS / MAINTENANCE
# ============================================================================
## env : affiche les principaux paramètres/chemins
.PHONY: env
env:
	@echo "LABELS_OUTDIR=$(LABELS_OUTDIR)"
	@echo "IMPUTE_OUT_DIR=$(IMPUTE_OUT_DIR)"
	@echo "BINNING_OUT_DIR=$(BINNING_OUT_DIR)"
	@echo "MODEL_ARTIFACTS_DIR=$(MODEL_ARTIFACTS_DIR)"
	@echo "MODEL_TRAIN_DATA=$(MODEL_TRAIN_DATA)"
	@echo "MODEL_VAL_DATA=$(MODEL_VAL_DATA)"
	@echo "MODEL_TARGET=$(MODEL_TARGET)"
	@echo "WOE_PREFIXES=$(WOE_PREFIXES)"
	@echo "BIN_SUFFIX=$(BIN_SUFFIX)"
	@echo "CV_FOLDS=$(CV_FOLDS)"
	@echo "CORR_THRESHOLD=$(CORR_THRESHOLD)"
	@echo "DROP_PROXY_CUTOFF=$(DROP_PROXY_CUTOFF)"
	@echo "CONDITIONAL_PROXIES=$(CONDITIONAL_PROXIES)"
	@echo "NO_ISOTONIC=$(NO_ISOTONIC) | NO_PRIOR_SHIFT_ADJUST=$(NO_PRIOR_SHIFT_ADJUST)"
	@echo "BIN_INCLUDE_MISSING=$(BIN_INCLUDE_MISSING) | BIN_COL_SUFFIX=$(BIN_COL_SUFFIX) | MIN_GINI_KEEP=$(MIN_GINI_KEEP)"
	@echo "BIN_DROP_MISSING_FLAGS=$(BIN_DROP_MISSING_FLAGS) | BIN_NO_DENYLIST=$(BIN_NO_DENYLIST)"
	@echo "N_JOBS_CATEG=$(N_JOBS_CATEG) | N_JOBS_NUM=$(N_JOBS_NUM)"
	@echo "ABLATION_MAX_STEPS=$(ABLATION_MAX_STEPS) | ABLATION_MAX_AUC_LOSS=$(ABLATION_MAX_AUC_LOSS)"
	@echo "MODEL_TIMING=$(MODEL_TIMING) | MODEL_TIMING_LIVE=$(MODEL_TIMING_LIVE)"
	@echo "IMPUTE_FROM_SPLITS=$(IMPUTE_FROM_SPLITS)"
	@echo "RESULTS_DATA=$(RESULTS_DATA)"
	@echo "RESULTS_BUCKETS=$(RESULTS_BUCKETS)"
	@echo "RESULTS_OUT=$(RESULTS_OUT)"
	@echo "IMPUTER_PATH=$(IMPUTER_PATH)"
	@echo "BINS_JSON=$(BINS_JSON)"
	@echo "MODEL_PATH=$(MODEL_PATH)"
	@echo "BUCKETS_PATH=$(BUCKETS_PATH)"
	@echo "OOS_LABELS_PARQUET=$(OOS_LABELS_PARQUET)"
	@echo "OOS_SCORED=$(OOS_SCORED)"

## clean_data : supprime les jeux dérivés
.PHONY: clean_data
clean_data:
	@rm -rf "$(LABELS_OUTDIR)" "$(IMPUTE_OUT_DIR)" "$(BINNING_OUT_DIR)" "$(SCORED_OUT_DIR)" || true
	@echo "→ Données dérivées supprimées."

## clean_artifacts : supprime les artefacts
.PHONY: clean_artifacts
clean_artifacts:
	@rm -rf "$(IMPUTE_ARTIFACTS_DIR)" "$(BINNING_ARTIFACTS_DIR)" "$(MODEL_ARTIFACTS_DIR)" || true
	@echo "→ Artefacts supprimés."

## clean_reports : supprime les rapports
.PHONY: clean_reports
clean_reports:
	@rm -rf "$(REPORTS_BASE_DIR)/iv" "$(REPORTS_BASE_DIR)/model" "$(REPORTS_BASE_DIR)/by_vintage_grades.csv" || true
	@echo "→ Rapports supprimés."

## clean_all : supprime toutes les sorties
.PHONY: clean_all
clean_all: clean_data clean_artifacts clean_reports

## print-% : debug d'une variable (ex: make print-MODEL_TARGET)
print-%:
	@echo '$*=$($*)'
