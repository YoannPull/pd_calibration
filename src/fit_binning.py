#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
from pathlib import Path
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features.binning import run_binning_maxgini_on_df, save_bins_json

def parse_args():
    p = argparse.ArgumentParser(description="Fit max|Gini| bins on TRAIN and apply to VALIDATION; save datasets + bins.json")
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--outdir", default="data/processed/merged/binned")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--format", choices=["parquet","csv"], default="parquet")
    # options binning
    p.add_argument("--bin-col-suffix", default="__BIN")
    p.add_argument("--include-missing", action="store_true")
    p.add_argument("--missing-label", default="__MISSING__")
    p.add_argument("--max-bins-categ", type=int, default=6)
    p.add_argument("--min-bin-size-categ", type=int, default=200)
    p.add_argument("--max-bins-num", type=int, default=6)
    p.add_argument("--min-bin-size-num", type=int, default=200)
    p.add_argument("--n-quantiles-num", type=int, default=50)
    p.add_argument("--min-gini-keep", type=float, default=None)
    return p.parse_args()

def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet",".pq"): return pd.read_parquet(p)
    return pd.read_csv(p)

def save_any(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".parquet",".pq"): df.to_parquet(path, index=False)
    else: df.to_csv(path, index=False)

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    artifacts = Path(args.artifacts); artifacts.mkdir(parents=True, exist_ok=True)

    tr = load_any(args.train); va = load_any(args.validation)
    if args.target not in tr.columns or args.target not in va.columns:
        raise SystemExit(f"Cible '{args.target}' absente du train/validation.")

    learned, tr_enriched, tr_binned = run_binning_maxgini_on_df(
        df=tr, target_col=args.target,
        include_missing=args.include_missing, missing_label=args.missing_label,
        max_bins_categ=args.max_bins_categ, min_bin_size_categ=args.min_bin_size_categ,
        max_bins_num=args.max_bins_num,   min_bin_size_num=args.min_bin_size_num,
        n_quantiles_num=args.n_quantiles_num,
        bin_col_suffix=args.bin_col_suffix,
        min_gini_keep=args.min_gini_keep
    )

    # Applique aux données de validation
    from features.binning import transform_with_learned_bins
    va_binned = transform_with_learned_bins(va, learned)

    # Sauvegardes datasets
    if args.format=="parquet":
        save_any(tr_binned, outdir / "train.parquet")
        save_any(va_binned, outdir / "validation.parquet")
    else:
        save_any(tr_binned, outdir / "train.csv")
        save_any(va_binned, outdir / "validation.csv")

    # bins.json
    save_bins_json(learned, artifacts / "bins.json")

    print("✔ Saved bins:", artifacts / "bins.json")
    print("✔ Datasets :", outdir)

if __name__ == "__main__":
    main()
