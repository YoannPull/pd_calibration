WINDOW ?= 24
IN_DIR ?= data/processed
SUBDIR ?= default_labels
OUT_DIR ?= data/processed/default_labels_imputed
FORMAT ?= parquet
IMPUT_COHORT ?= 1
TRAIN_RANGE ?= 2018Q1:2021Q4
VAL_RANGE ?= 2022Q1:2022Q4
TEST_RANGE ?= 2023Q1:

# Robuste: exécute le script directement via son chemin source
CMD ?= PYTHONPATH=src poetry run python src/make_splits_impute.py

.PHONY: splits
splits:
	$(CMD) \
		--window $(WINDOW) \
		--in_dir $(IN_DIR) --subdir $(SUBDIR) \
		--out_dir $(OUT_DIR) \
		--format $(FORMAT) $(if $(filter $(IMPUT_COHORT),1),--imput_cohort,) \
		--train_range "$(TRAIN_RANGE)" \
		--val_range   "$(VAL_RANGE)" \
		--test_range  "$(TEST_RANGE)"
