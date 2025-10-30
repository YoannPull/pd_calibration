#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from joblib import dump

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features.binning import BinningTransformer


def parse_args():
    p = argparse.ArgumentParser(description="Fit WOE binning on train, apply to validation, and save artifacts")
    p.add_argument("--train", required=True, help="Chemin train (CSV/Parquet)")
    p.add_argument("--validation", required=True, help="Chemin validation (CSV/Parquet)")
    p.add_argument("--target", required=True, help="Nom de la cible binaire (0/1)")
    p.add_argument("--outdir", default="data/processed/merged/binned", help="Dossier de sortie (datasets)")
    p.add_argument("--artifacts", default="artifacts", help="Dossier artifacts (binning.joblib, IV)")
    p.add_argument("--variables", default=None, help="Liste de variables à binner, séparées par des virgules")
    p.add_argument("--n-bins", type=int, default=10, help="Nombre de bins pour les numériques")
    p.add_argument("--output", choices=["woe", "bin_index", "both"], default="woe")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    p.add_argument("--drop-original", action="store_true", help="Ne garder que les features encodées")
    p.add_argument("--include-categorical", action="store_true", help="Inclure aussi les variables catégorielles")
    return p.parse_args()


def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_any(df_train: pd.DataFrame, df_val: pd.DataFrame, outdir: Path, fmt: str):
    outdir.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        (outdir / "train.parquet").unlink(missing_ok=True)
        (outdir / "validation.parquet").unlink(missing_ok=True)
        df_train.to_parquet(outdir / "train.parquet", index=False)
        df_val.to_parquet(outdir / "validation.parquet", index=False)
    else:
        df_train.to_csv(outdir / "train.csv", index=False)
        df_val.to_csv(outdir / "validation.csv", index=False)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    df_train = load_any(args.train)
    df_val = load_any(args.validation)

    # Extract target & feature frame
    if args.target not in df_train.columns or args.target not in df_val.columns:
        raise SystemExit(f"Cible '{args.target}' absente du train/validation.")

    y_train = df_train[args.target].astype(int).values
    y_val = df_val[args.target].astype(int).values
    X_train = df_train.drop(columns=[args.target])
    X_val = df_val.drop(columns=[args.target])

    variables = None
    if args.variables:
        variables = [c.strip() for c in args.variables.split(",") if c.strip()]

    binner = BinningTransformer(
        variables=variables,
        n_bins=args.n_bins,
        output=args.output,
        include_categorical=args.include_categorical,
        drop_original=args.drop_original,
    )
    binner.fit(X_train, y_train)

    train_binned = binner.transform(X_train)
    val_binned = binner.transform(X_val)

    # Contrôle (optionnel): pas de NaN dans les colonnes woe/bin
    new_cols = [c for c in train_binned.columns if c not in X_train.columns] if not args.drop_original else train_binned.columns
    if train_binned[new_cols].isna().any().any() or val_binned[new_cols].isna().any().any():
        raise SystemExit("NaN résiduels après binning/WOE. Vérifie les colonnes problématiques.")

    # Sauvegardes datasets
    save_any(train_binned.assign(**{args.target: y_train}), val_binned.assign(**{args.target: y_val}), outdir, args.format)

    # IV summary
    iv_df = binner.iv_summary_()
    iv_df.to_csv(artifacts / "binning_iv.csv", index=False)

    # Persist binner
    dump(binner, artifacts / "binning.joblib")

    # Petite meta
    meta = {
        "target": args.target,
        "variables": variables if variables else "auto",
        "n_bins": args.n_bins,
        "output": args.output,
        "include_categorical": args.include_categorical,
        "drop_original": args.drop_original,
        "train_path": str(Path(args.train).resolve()),
        "validation_path": str(Path(args.validation).resolve()),
        "outdir": str(outdir.resolve()),
    }
    (artifacts / "binning_meta.json").write_text(json.dumps(meta, indent=2))
    print("✔ Binning sauvé:", artifacts / "binning.joblib")
    print("✔ IV summary   :", artifacts / "binning_iv.csv")
    print("✔ Datasets     :", outdir)


if __name__ == "__main__":
    main()
