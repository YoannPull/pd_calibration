#!/usr/bin/env python3
# src/impute_and_save.py
# -*- coding: utf-8 -*-

"""
impute_and_save.py
==================

Impute train/validation datasets using `DataImputer`, then save:
- Parquet files (native dtype preservation), OR
- CSV files + pickled dtype metadata (parse_dates, categorical dtypes, other dtypes)

Key fixes / improvements
------------------------
1) Data leakage prevention:
   When using split manifests (`window/_splits.json`), the script explicitly removes
   validation quarters from the pooled training file before fitting the imputer.

2) Memory optimization:
   When possible, it attempts to read `pooled.parquet` with a push-down filter
   (via pandas/pyarrow) to avoid loading validation rows in memory.

Typical usage
-------------
A) Explicit mode (provide paths)
    python src/impute_and_save.py --train-csv train.parquet --validation-csv val.parquet --outdir ...

B) Split-driven mode (recommended)
    python src/impute_and_save.py --use-splits --labels-window-dir data/processed/default_labels/window=24m \
        --outdir data/processed/imputed --artifacts artifacts/imputer

Outputs
-------
- Imputed datasets in <outdir>:
    - train.parquet / validation.parquet   (or train_imputed.csv / validation_imputed.csv)
- Imputer artifact in <artifacts>:
    - imputer.joblib
    - imputer_meta.json

Notes
-----
- The imputer is fitted on TRAIN only (after filtering out validation quarters).
- Optionally, the target column can be removed before imputation and re-attached afterward.
- If `--fail-on-nan` is enabled, the script raises an error if NaNs remain after imputation.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from joblib import dump

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

# Ensure `src/` is importable when the script is executed directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Import the project imputer implementation.
# If import fails, check PYTHONPATH or adjust the import paths below.
try:
    from features.impute import DataImputer
except ImportError:
    # Local dev fallback if the package is not installed/available as expected.
    sys.path.append(".")
    from src.features.impute import DataImputer


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Impute & save datasets + persist imputer")

    # Explicit mode (backward compatible): provide train/validation paths.
    p.add_argument("--train-csv", help="Training file (CSV/Parquet)")
    p.add_argument("--validation-csv", help="Validation file (CSV/Parquet)")

    # Split-driven mode: use a manifest produced upstream (make_labels.py).
    p.add_argument(
        "--labels-window-dir",
        help="Path to the window=XXm directory (e.g., data/processed/default_labels/window=24m)",
    )
    p.add_argument(
        "--use-splits",
        action="store_true",
        help="Use window/_splits.json to construct train/validation",
    )

    # Optional settings
    p.add_argument("--target", default=None, help="Target column name (optional)")
    p.add_argument("--outdir", default="data/processed/imputed", help="Output directory for imputed datasets")
    p.add_argument(
        "--artifacts",
        default="artifacts/imputer",
        help="Directory used to persist the imputer and metadata",
    )
    p.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Save imputed datasets as Parquet (recommended) or CSV",
    )
    p.add_argument("--use-cohort", action="store_true", help="Enable cohort-based imputation (if supported)")
    p.add_argument("--missing-flag", action="store_true", help="Add missingness indicator features (was_missing_*)")
    p.add_argument("--fail-on-nan", action="store_true", help="Fail if NaNs remain after imputation")

    return p.parse_args()


# =============================================================================
# I/O helpers
# =============================================================================

def load_any(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from Parquet or CSV (format inferred from extension)."""
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def concat_parquet(paths: List[Path]) -> pd.DataFrame:
    """
    Concatenate multiple parquet files, aligning columns when schemas differ.

    This is used in split-driven mode where each quarter is stored separately.
    """
    if not paths:
        return pd.DataFrame()

    dfs = []
    for p in paths:
        if p.exists():
            dfs.append(pd.read_parquet(p))

    if not dfs:
        return pd.DataFrame()

    # Align schemas across quarterly files.
    all_cols = sorted(set().union(*[df.columns for df in dfs]))
    dfs = [df.reindex(columns=all_cols) for df in dfs]
    return pd.concat(dfs, ignore_index=True)


def save_parquet(df_train_imp: pd.DataFrame, df_val_imp: pd.DataFrame, outdir: Path):
    """Save imputed datasets to parquet in a consistent location and naming scheme."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Remove existing outputs to avoid confusion with stale files.
    (outdir / "train.parquet").unlink(missing_ok=True)
    (outdir / "validation.parquet").unlink(missing_ok=True)

    df_train_imp.to_parquet(outdir / "train.parquet", index=False)
    df_val_imp.to_parquet(outdir / "validation.parquet", index=False)
    print(f"✔ Wrote parquet datasets to {outdir}")


def save_csv_with_dtypes(df_train_imp: pd.DataFrame, df_val_imp: pd.DataFrame, outdir: Path):
    """
    Save imputed datasets to CSV and persist dtype metadata.

    CSV does not preserve dtypes well. We store:
    - parse_dates: datetime-like columns
    - cat_dtypes: categorical dtypes
    - other_dtypes: remaining dtypes (including nullable Int64 when needed)
    """
    from pandas.api.types import CategoricalDtype  # local import for pickling stability

    outdir.mkdir(parents=True, exist_ok=True)

    dtypes = df_train_imp.dtypes.to_dict()
    parse_dates = [c for c, dt in dtypes.items() if str(dt).startswith("datetime64")]
    cat_dtypes = {c: dt for c, dt in dtypes.items() if isinstance(dt, CategoricalDtype)}
    other_dtypes = {
        c: ("Int64" if str(dt).startswith("int") and df_train_imp[c].isna().any() else dt)
        for c, dt in dtypes.items()
        if c not in parse_dates and c not in cat_dtypes
    }

    with open(outdir / "parse_dates.pkl", "wb") as f:
        pickle.dump(parse_dates, f)
    with open(outdir / "cat_dtypes.pkl", "wb") as f:
        pickle.dump(cat_dtypes, f)
    with open(outdir / "other_dtypes.pkl", "wb") as f:
        pickle.dump(other_dtypes, f)

    df_train_imp.to_csv(outdir / "train_imputed.csv", index=False)
    df_val_imp.to_csv(outdir / "validation_imputed.csv", index=False)
    print(f"✔ Wrote CSV + dtype pickles to {outdir}")


# =============================================================================
# Split manifest loader
# =============================================================================

def _pull(d: dict, key: str, default=None):
    """
    Fetch a key either from the root of a dict or from d['splits'].

    This supports older/newer manifest formats without hard-coding one schema.
    """
    if key in d:
        return d[key]
    s = d.get("splits", {})
    return s.get(key, default)


def resolve_splits(labels_window_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Construct (train_df, val_df) from window/_splits.json.

    Processing logic
    ----------------
    1) Load the split manifest.
    2) Identify the validation quarter(s) and validation mode.
    3) Build the validation dataset:
       - either from quarter=.../data.parquet files ("quarters" mode),
       - or from oos.parquet ("oos" mode).
    4) Build the training dataset from pooled.parquet and *explicitly exclude*
       validation quarters to prevent leakage.

    Returns
    -------
    df_train : pd.DataFrame
        Training dataset (pooled.parquet filtered to remove validation quarters).
    df_val : pd.DataFrame
        Validation dataset (quarters or oos).
    manifest : dict
        The original manifest plus a "_resolved" section with effective paths/mode.
    """
    splits_path = labels_window_dir / "_splits.json"
    pooled_path = labels_window_dir / "pooled.parquet"
    oos_path = labels_window_dir / "oos.parquet"

    if not splits_path.exists():
        raise SystemExit(f"[ERR] Splits file not found: {splits_path}")

    manifest = json.loads(splits_path.read_text(encoding="utf-8"))

    # --- 1) Determine validation strategy ---
    mode = (_pull(manifest, "validation_mode", None) or "quarters").lower()

    val_quarters = _pull(manifest, "validation_quarters", [])
    if not val_quarters:
        # Backward compatible fallback for older manifests.
        explicit = _pull(manifest, "explicit", {}) or {}
        val_quarters = list(explicit.get("validation_quarters", []))
        if not val_quarters:
            val_quarters = list(explicit.get("default_val_quarters", []))
        if not val_quarters and explicit.get("default_val_quarter"):
            val_quarters = [explicit["default_val_quarter"]]

    oos_quarters = _pull(manifest, "oos_quarters", [])
    if not oos_quarters:
        explicit = _pull(manifest, "explicit", {}) or {}
        oos_quarters = list(explicit.get("oos_quarters", []))

    # If quarters mode is requested but quarters are missing, attempt OOS fallback.
    if mode == "quarters" and not val_quarters:
        if oos_path.exists() and oos_quarters:
            print("[WARN] validation_quarters missing -> falling back to OOS mode.")
            mode = "oos"
        else:
            raise SystemExit("[ERR] No validation_quarters found and no OOS fallback available.")

    # --- 2) Build validation dataframe ---
    df_val = pd.DataFrame()
    used_quarters = []
    quarters_to_exclude_from_train = []

    if mode == "oos":
        if not oos_path.exists():
            raise SystemExit(f"[ERR] Validation 'oos.parquet' not found at {oos_path}")
        df_val = load_any(oos_path)
        used_quarters = oos_quarters

        # In strict OOS setups, pooled.parquet should already contain only train.
        # We do not filter in OOS mode unless you explicitly decide to do so.
    else:
        # "quarters" mode: validation is built by concatenating quarter files.
        files = [labels_window_dir / f"quarter={q}" / "data.parquet" for q in val_quarters]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise SystemExit("[ERR] Missing validation quarter files:\n  " + "\n  ".join(missing))

        df_val = concat_parquet(files)
        used_quarters = val_quarters
        quarters_to_exclude_from_train = val_quarters

    # --- 3) Build training dataframe from pooled.parquet (with leakage filtering) ---
    if not pooled_path.exists():
        raise SystemExit(f"[ERR] Train 'pooled.parquet' not found at {pooled_path}")

    df_train = None

    # Candidate split columns (depends on upstream labeling pipeline)
    split_col_candidates = ["vintage", "__file_quarter", "quarter"]

    # Attempt push-down filtering with pyarrow/pandas. If it fails, fallback to pandas filtering.
    try:
        import pyarrow.parquet as pq

        pq_file = pq.ParquetFile(pooled_path)
        cols_in_file = pq_file.schema.names

        split_col = next((c for c in split_col_candidates if c in cols_in_file), None)

        if split_col and quarters_to_exclude_from_train:
            print(
                f"[SPLIT] PyArrow optimization: excluding {len(quarters_to_exclude_from_train)} "
                f"quarter(s) using column '{split_col}'."
            )

            # NOTE: pandas supports `filters=` with pyarrow backend; behavior depends on versions.
            filters = [(split_col, "not in", quarters_to_exclude_from_train)]
            df_train = pd.read_parquet(pooled_path, filters=filters)
        else:
            df_train = pd.read_parquet(pooled_path)

    except Exception as e:
        print(f"[INFO] Optimized read unavailable ({e}); falling back to standard load.")
        df_train = load_any(pooled_path)
        split_col = next((c for c in split_col_candidates if c in df_train.columns), None)

    # Safety filter (double check) to guarantee no leakage remains.
    if quarters_to_exclude_from_train and split_col and split_col in df_train.columns:
        n_orig = len(df_train)
        mask_leak = df_train[split_col].isin(quarters_to_exclude_from_train)
        if mask_leak.any():
            print("[SPLIT] Applying safety filter (leakage prevention).")
            df_train = df_train[~mask_leak].copy()
            print(f"[SPLIT] Removed rows from train: {n_orig - len(df_train)}")

    if df_train is None or df_train.empty:
        print("[WARN] Training dataframe is empty after split resolution.", file=sys.stderr)

    # --- 4) Align columns across train/validation (schema consistency) ---
    all_cols = sorted(set(df_train.columns).union(df_val.columns))
    df_train = df_train.reindex(columns=all_cols)
    df_val = df_val.reindex(columns=all_cols)

    print(f"[SPLITS] mode: {mode}")
    print(f"[SPLITS] Train shape: {df_train.shape} (from pooled)")
    print(f"[SPLITS] Val   shape: {df_val.shape} (quarters: {used_quarters})")

    meta = {
        "train_file": str(pooled_path),
        "validation_mode": mode,
        "validation_used_quarters": used_quarters,
        "validation_paths": (
            [str(labels_window_dir / f"quarter={q}" / "data.parquet") for q in used_quarters]
            if mode != "oos"
            else [str(oos_path)]
        ),
        "train_excluded_quarters": quarters_to_exclude_from_train,
    }
    manifest["_resolved"] = meta
    return df_train, df_val, manifest


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    outdir = Path(args.outdir)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Select data source: explicit paths vs split-driven mode
    # -----------------------------------------------------------------------
    df_train: Optional[pd.DataFrame] = None
    df_val: Optional[pd.DataFrame] = None
    splits_meta: Optional[dict] = None

    if args.use_splits:
        if not args.labels_window_dir:
            raise SystemExit("[ERR] --use-splits requires --labels-window-dir")
        labels_dir = Path(args.labels_window_dir)
        if not labels_dir.exists():
            raise SystemExit(f"[ERR] labels_window_dir not found: {labels_dir}")
        df_train, df_val, splits_meta = resolve_splits(labels_dir)
    else:
        if not args.train_csv or not args.validation_csv:
            raise SystemExit(
                "[ERR] Provide --train-csv and --validation-csv, or use --use-splits with --labels-window-dir"
            )
        df_train = load_any(args.train_csv)
        df_val = load_any(args.validation_csv)

    # -----------------------------------------------------------------------
    # Optionally separate the target (never impute the target)
    # -----------------------------------------------------------------------
    y_train = y_val = None

    if args.target and args.target in df_train.columns:
        y_train = df_train[args.target].copy()
        df_train = df_train.drop(columns=[args.target])

    if args.target and args.target in df_val.columns:
        y_val = df_val[args.target].copy()
        df_val = df_val.drop(columns=[args.target])

    # -----------------------------------------------------------------------
    # Fit imputer on TRAIN only, then transform train and validation
    # -----------------------------------------------------------------------
    imputer = DataImputer(use_cohort=args.use_cohort, missing_flag=args.missing_flag)

    print(f"-> Fitting imputer on Train ({len(df_train)} rows)...")
    imputer.fit(df_train)

    print("-> Transforming Train...")
    df_train_imp = imputer.transform(df_train)

    print("-> Transforming Validation...")
    df_val_imp = imputer.transform(df_val)

    # Re-attach target if it was removed
    if y_train is not None:
        df_train_imp[args.target] = y_train.values
    if y_val is not None:
        df_val_imp[args.target] = y_val.values

    # Optional sanity check: ensure no NaNs remain
    if args.fail_on_nan:
        bad_train = df_train_imp.columns[df_train_imp.isna().any()].tolist()
        bad_val = df_val_imp.columns[df_val_imp.isna().any()].tolist()
        if bad_train or bad_val:
            raise SystemExit(
                "NaNs remain after imputation.\n"
                f"Train: {bad_train}\n"
                f"Validation: {bad_val}"
            )

    # -----------------------------------------------------------------------
    # Save imputed datasets
    # -----------------------------------------------------------------------
    if args.format == "parquet":
        save_parquet(df_train_imp, df_val_imp, outdir)
    else:
        save_csv_with_dtypes(df_train_imp, df_val_imp, outdir)

    # -----------------------------------------------------------------------
    # Persist the imputer + metadata
    # -----------------------------------------------------------------------
    dump(imputer, artifacts / "imputer.joblib")

    meta = {
        "target": args.target,
        "use_cohort": args.use_cohort,
        "missing_flag": args.missing_flag,
        "features": (
            list(df_train_imp.drop(columns=[args.target]).columns)
            if args.target
            else list(df_train_imp.columns)
        ),
        "output_dir": str(outdir.resolve()),
        "format": args.format,
        "mode": "splits" if args.use_splits else "explicit",
    }

    if args.use_splits and splits_meta is not None:
        meta["splits"] = splits_meta
    else:
        meta["train_path"] = str(Path(args.train_csv).resolve())
        meta["validation_path"] = str(Path(args.validation_csv).resolve())

    (artifacts / "imputer_meta.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"✔ Imputer saved to {artifacts / 'imputer.joblib'}")


if __name__ == "__main__":
    main()
