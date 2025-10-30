#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os
import sys
import pandas as pd
from joblib import load

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Apply a saved binning (joblib) to a new dataset")
    p.add_argument("--data", required=True, help="Chemin dataset (CSV/Parquet)")
    p.add_argument("--binner", required=True, help="Chemin artifacts/binning.joblib")
    p.add_argument("--out", required=True, help="Chemin de sortie (CSV/Parquet)")
    return p.parse_args()


def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_any(df: pd.DataFrame, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)


def main():
    args = parse_args()
    df = load_any(args.data)
    binner = load(args.binner)

    df_binned = binner.transform(df)
    save_any(df_binned, args.out)
    print("✔ Binning appliqué →", args.out)


if __name__ == "__main__":
    main()
