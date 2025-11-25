#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit max|Gini| bins on TRAIN and apply to VALIDATION.
Uses the robust 'src.features.binning' module (with Monotonicity constraint).

Outputs:
    - data/processed/merged/binned/train.parquet
    - data/processed/merged/binned/validation.parquet
    - artifacts/bins.json
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import propre (suppose que le script est lancé via 'poetry run python ...' ou PYTHONPATH=. set)
try:
    from features.binning import (
        run_binning_maxgini_on_df, 
        transform_with_learned_bins,
        save_bins_json,
        DENYLIST_STRICT_DEFAULT, 
        EXCLUDE_IDS_DEFAULT
    )
except ImportError:
    # Fallback si lancé en script direct sans contexte
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from features.binning import (
        run_binning_maxgini_on_df, 
        transform_with_learned_bins,
        save_bins_json,
        DENYLIST_STRICT_DEFAULT, 
        EXCLUDE_IDS_DEFAULT
    )

def parse_args():
    p = argparse.ArgumentParser(description="Fit Max-Gini Binning (Monotonic) & Apply")
    
    # I/O
    p.add_argument("--train", required=True, help="Path to train parquet/csv")
    p.add_argument("--validation", required=True, help="Path to validation parquet/csv")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument("--outdir", default="data/processed/binned", help="Output directory for datasets")
    p.add_argument("--artifacts", default="artifacts/binning_maxgini", help="Output directory for bins.json")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet")

    # Options de Binning
    p.add_argument("--bin-col-suffix", default="__BIN")
    p.add_argument("--include-missing", action="store_true", help="Create a specific bin for missing values")
    p.add_argument("--missing-label", default="__MISSING__")
    
    # Hyperparamètres (Tailles & Quantiles)
    p.add_argument("--max-bins-categ", type=int, default=6)
    p.add_argument("--min-bin-size-categ", type=int, default=200)
    
    p.add_argument("--max-bins-num", type=int, default=6)
    p.add_argument("--min-bin-size-num", type=int, default=200)
    p.add_argument("--n-quantiles-num", type=int, default=50, help="Granularity for initial numeric splits")
    
    p.add_argument("--min-gini-keep", type=float, default=None, help="Drop variables below this Gini threshold")
    
    # Flags de sécurité / Nettoyage
    p.add_argument("--no-denylist", action="store_true", help="Disable the strict denylist (dates, vintage...)")
    p.add_argument("--drop-missing-flags", action="store_true", help="Drop columns like 'was_missing_*'")
    
    # Note: Le parallélisme (n_jobs) a été retiré de l'implémentation backend pour robustesse/simplicité.
    
    return p.parse_args()

def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        logger.error(f"File not found: {p}")
        sys.exit(1)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)

def save_any(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def main():
    args = parse_args()
    
    outdir = Path(args.outdir)
    artifacts = Path(args.artifacts)
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    # 1. Chargement
    logger.info("Loading datasets...")
    tr = load_any(args.train)
    va = load_any(args.validation)

    if args.target not in tr.columns:
        logger.error(f"Target '{args.target}' missing from Train.")
        sys.exit(1)

    # 2. Fitting (Train)
    logger.info(f"Starting Binning on Train ({len(tr)} rows)...")
    logger.info(f"Target: {args.target} | Monotonicity Constraint: ACTIVE")

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
        exclude_ids=EXCLUDE_IDS_DEFAULT
    )

    # 3. Application (Validation)
    logger.info(f"Applying bins to Validation ({len(va)} rows)...")
    va_binned = transform_with_learned_bins(va, learned)

    # 4. Sauvegarde
    logger.info("Saving datasets...")
    if args.format == "parquet":
        save_any(tr_binned, outdir / "train.parquet")
        save_any(va_binned, outdir / "validation.parquet")
    else:
        save_any(tr_binned, outdir / "train.csv")
        save_any(va_binned, outdir / "validation.csv")

    # Sauvegarde des bins (JSON)
    json_path = artifacts / "bins.json"
    save_bins_json(learned, json_path)

    logger.info(f"✔ Bins saved: {json_path}")
    logger.info(f"✔ Data saved: {outdir}")

if __name__ == "__main__":
    main()