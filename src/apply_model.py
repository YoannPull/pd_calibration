#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load a persisted model (joblib) and apply it to a new dataset (CSV/Parquet).
Appends `proba` column (class 1) and saves the result.
"""

import argparse
from pathlib import Path
import os
import sys
import pandas as pd
from joblib import load


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


def parse_args():
    p = argparse.ArgumentParser(description="Apply a saved model to a new dataset")
    p.add_argument("--data", required=True, help="CSV/Parquet with same features used in training")
    p.add_argument("--model", required=True, help="Path to artifacts/model_best.joblib")
    p.add_argument("--out", required=True, help="Output path (CSV/Parquet)")
    return p.parse_args()


def main():
    args = parse_args()

    bundle = load(args.model)  # dict: {"model": estimator, "features": [...], "target": "..."}
    model = bundle["model"]
    features = bundle["features"]

    df = load_any(args.data)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature(s) in data: {missing}")

    proba = model.predict_proba(df[features])[:, 1]
    out = df.copy()
    out["proba"] = proba

    save_any(out, args.out)
    print(f"✔ Predictions saved to: {args.out}")


if __name__ == "__main__":
    main()
