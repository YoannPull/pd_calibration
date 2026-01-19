#!/usr/bin/env python3
# src/fit_binning.py
# -*- coding: utf-8 -*-

"""
Fit max-|Gini| bins on TRAIN and apply them to VALIDATION.

This script learns a supervised binning scheme on the training sample using the
robust `features.binning` module (including a monotonicity constraint), then
applies the *same* learned bins to the validation dataset to ensure consistent
feature preprocessing.

Outputs
-------
- Binned datasets:
    - <outdir>/train.(parquet|csv)
    - <outdir>/validation.(parquet|csv)
- Learned binning artifact:
    - <artifacts>/bins.json

Notes
-----
- The strict denylist (dates, vintage-like fields, etc.) is enabled by default to
  prevent target leakage / temporal proxies from being binned as predictive features.
- Some metadata columns (e.g., vintage, loan_sequence_number) can be passed through
  to the outputs for analysis and reporting.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from the binning module
# ---------------------------------------------------------------------------

# Clean import (assumes execution via `poetry run ...` or with PYTHONPATH set)
try:
    from features.binning import (
        run_binning_maxgini_on_df,
        transform_with_learned_bins,
        save_bins_json,
        DENYLIST_STRICT_DEFAULT,
        EXCLUDE_IDS_DEFAULT,
    )
except ImportError:
    # Fallback if executed as a standalone script without the expected environment
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from features.binning import (
        run_binning_maxgini_on_df,
        transform_with_learned_bins,
        save_bins_json,
        DENYLIST_STRICT_DEFAULT,
        EXCLUDE_IDS_DEFAULT,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Define and parse CLI arguments."""
    p = argparse.ArgumentParser(description="Fit Max-|Gini| Binning (Monotonic) & Apply")

    # I/O
    p.add_argument("--train", required=True, help="Path to train dataset (parquet/csv)")
    p.add_argument("--validation", required=True, help="Path to validation dataset (parquet/csv)")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument("--outdir", default="data/processed/binned", help="Output directory for datasets")
    p.add_argument("--artifacts", default="artifacts/binning_maxgini", help="Output directory for bins.json")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Output format for datasets")

    # Binning behavior
    p.add_argument("--bin-col-suffix", default="__BIN", help="Suffix for created binned columns")
    p.add_argument("--include-missing", action="store_true", help="Create a dedicated bin for missing values")
    p.add_argument("--missing-label", default="__MISSING__", help="Label used for the missing bin")

    # Hyperparameters (bin counts and minimum bin sizes)
    p.add_argument("--max-bins-categ", type=int, default=6, help="Max bins for categorical variables")
    p.add_argument("--min-bin-size-categ", type=int, default=200, help="Minimum bin size for categorical variables")

    p.add_argument("--max-bins-num", type=int, default=6, help="Max bins for numerical variables")
    p.add_argument("--min-bin-size-num", type=int, default=200, help="Minimum bin size for numerical variables")
    p.add_argument("--n-quantiles-num", type=int, default=50, help="Granularity for initial numeric splits")

    p.add_argument(
        "--min-gini-keep",
        type=float,
        default=None,
        help="Optional: drop variables with |Gini| below this threshold",
    )

    # Safety / cleanup flags
    p.add_argument(
        "--no-denylist",
        action="store_true",
        help="Disable the strict denylist (dates, vintage-like fields, etc.)",
    )
    p.add_argument(
        "--drop-missing-flags",
        action="store_true",
        help="Drop missingness indicator columns such as 'was_missing_*'",
    )

    # Note: parallelism (n_jobs) was removed from the backend for robustness/simplicity.
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_any(path: str) -> pd.DataFrame:
    """
    Load a dataset from disk.

    Supported formats:
    - Parquet (.parquet, .pq)
    - CSV (fallback)
    """
    p = Path(path)
    if not p.exists():
        logger.error("File not found: %s", p)
        sys.exit(1)

    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_any(df: pd.DataFrame, path: Path) -> None:
    """Save a dataset to disk (parquet or csv based on the file extension)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    outdir = Path(args.outdir)
    artifacts = Path(args.artifacts)
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    # 1) Load datasets
    logger.info("Loading datasets...")
    tr = load_any(args.train)
    va = load_any(args.validation)

    if args.target not in tr.columns:
        logger.error("Target '%s' missing from train dataset.", args.target)
        sys.exit(1)

    # 2) Fit binning on TRAIN
    logger.info("Starting binning on TRAIN (%d rows)...", len(tr))
    logger.info("Target: %s | Monotonicity constraint: ACTIVE", args.target)

    learned, tr_enriched, tr_binned = run_binning_maxgini_on_df(
        df=tr,
        target_col=args.target,
        include_missing=args.include_missing,
        missing_label=args.missing_label,
        max_bins_categ=args.max_bins_categ,
        min_bin_size_categ=args.min_bin_size_categ,
        max_bins_num=args.max_bins_num,
        min_bin_size_num=args.min_bin_size_num,
        n_quantiles_num=args.n_quantiles_num,
        bin_col_suffix=args.bin_col_suffix,
        min_gini_keep=args.min_gini_keep,
        denylist_strict=([] if args.no_denylist else list(DENYLIST_STRICT_DEFAULT)),
        drop_missing_flags=bool(args.drop_missing_flags),
        exclude_ids=EXCLUDE_IDS_DEFAULT,
    )

    # 3) Apply learned bins to VALIDATION
    logger.info("Applying learned bins to VALIDATION (%d rows)...", len(va))
    va_binned = transform_with_learned_bins(va, learned)

    # 3b) Pass-through metadata columns (not features, but useful for analysis/reporting)
    META_COLS = ["vintage", "loan_sequence_number"]
    for col in META_COLS:
        if col in tr.columns:
            tr_binned[col] = tr[col]
        if col in va.columns:
            va_binned[col] = va[col]

    # 4) Save outputs
    logger.info("Saving binned datasets...")
    if args.format == "parquet":
        save_any(tr_binned, outdir / "train.parquet")
        save_any(va_binned, outdir / "validation.parquet")
    else:
        save_any(tr_binned, outdir / "train.csv")
        save_any(va_binned, outdir / "validation.csv")

    # Save learned bins as JSON artifact
    json_path = artifacts / "bins.json"
    save_bins_json(learned, json_path)

    logger.info("✔ Bins saved: %s", json_path)
    logger.info("✔ Data saved: %s", outdir)


if __name__ == "__main__":
    main()
