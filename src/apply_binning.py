#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
from pathlib import Path
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features.binning_maxgini import load_bins_json, transform_with_learned_bins

def parse_args():
    p = argparse.ArgumentParser(description="Apply learned max|Gini| bins (bins.json) to a new dataset")
    p.add_argument("--data", required=True)
    p.add_argument("--bins", required=True, help="Chemin artifacts/bins.json")
    p.add_argument("--out", required=True)
    return p.parse_args()

def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet",".pq"): return pd.read_parquet(p)
    return pd.read_csv(p)

def save_any(df: pd.DataFrame, path: str):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".parquet",".pq"): df.to_parquet(p, index=False)
    else: df.to_csv(p, index=False)

def main():
    args = parse_args()
    df = load_any(args.data)
    learned = load_bins_json(args.bins)
    df_binned = transform_with_learned_bins(df, learned)
    save_any(df_binned, args.out)
    print("✔ Binning appliqué →", args.out)

if __name__ == "__main__":
    main()
