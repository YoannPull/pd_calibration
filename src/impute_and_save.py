#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Impute train/validation with DataImputer, then save either:
- Parquet files (types conservés nativement), or
- CSV + pickles des dtypes (parse_dates, cat_dtypes, other_dtypes)
Also persists the fitted imputer for reuse.
"""

import argparse
import os
from pathlib import Path
import json
import pickle

import pandas as pd
from joblib import dump

# Rendez le package importable depuis ./src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features.impute2 import DataImputer


def parse_args():
    p = argparse.ArgumentParser(description="Impute & save datasets + persist imputer")
    p.add_argument("--train-csv", required=True, help="Chemin vers le CSV d'entraînement")
    p.add_argument("--validation-csv", required=True, help="Chemin vers le CSV de validation")
    p.add_argument("--target", default=None, help="Nom de la colonne cible à conserver (optionnel)")
    p.add_argument("--outdir", default="data/processed/merged/imputed", help="Dossier de sortie")
    p.add_argument("--artifacts", default="artifacts", help="Dossier où sauvegarder l'imputer")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                   help="Format de sauvegarde des jeux imputés")
    p.add_argument("--use-cohort", action="store_true", help="Imputation par cohortes")
    p.add_argument("--missing-flag", action="store_true", help="Ajouter les indicateurs was_missing_")
    return p.parse_args()


def save_parquet(df_train_imp, df_val_imp, outdir: Path):
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "train.parquet").unlink(missing_ok=True)
        (outdir / "validation.parquet").unlink(missing_ok=True)
        df_train_imp.to_parquet(outdir / "train.parquet", index=False)
        df_val_imp.to_parquet(outdir / "validation.parquet", index=False)
        print(f"✔ Parquet écrit dans {outdir}")
    except Exception as e:
        raise SystemExit(
            "Échec d'écriture Parquet. Installe pyarrow ou fastparquet, ou utilise --format csv.\n"
            f"Détail: {e}"
        )


def save_csv_with_dtypes(df_train_imp, df_val_imp, outdir: Path):
    from pandas.api.types import CategoricalDtype  # import local pour cohérence avec pickles

    outdir.mkdir(parents=True, exist_ok=True)

    # Dtypes de référence (ceux du train)
    dtypes = df_train_imp.dtypes.to_dict()

    parse_dates = [c for c, dt in dtypes.items() if str(dt).startswith("datetime64")]
    cat_dtypes = {c: dt for c, dt in dtypes.items() if isinstance(dt, CategoricalDtype)}
    other_dtypes = {
        c: ("Int64" if str(dt).startswith("int") and df_train_imp[c].isna().any() else dt)
        for c, dt in dtypes.items()
        if c not in parse_dates and c not in cat_dtypes
    }

    # Sauvegardes des schémas
    with open(outdir / "parse_dates.pkl", "wb") as f:
        pickle.dump(parse_dates, f)
    with open(outdir / "cat_dtypes.pkl", "wb") as f:
        pickle.dump(cat_dtypes, f)
    with open(outdir / "other_dtypes.pkl", "wb") as f:
        pickle.dump(other_dtypes, f)

    # CSV
    df_train_imp.to_csv(outdir / "train_imputed.csv", index=False)
    df_val_imp.to_csv(outdir / "validation_imputed.csv", index=False)
    print(f"✔ CSV + dtypes picklés écrits dans {outdir}")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    # Chargement
    df_train = pd.read_parquet(args.train_csv)
    df_val = pd.read_parquet(args.validation_csv)

    # Séparer la cible si demandée (et la ré-attacher ensuite inchangée)
    y_train = y_val = None
    if args.target and args.target in df_train.columns:
        y_train = df_train[args.target].copy()
        df_train = df_train.drop(columns=[args.target])
    if args.target and args.target in df_val.columns:
        y_val = df_val[args.target].copy()
        df_val = df_val.drop(columns=[args.target])

    # Imputation
    imputer = DataImputer(use_cohort=args.use_cohort, missing_flag=args.missing_flag)
    imputer.fit(df_train)
    df_train_imp = imputer.transform(df_train)
    df_val_imp = imputer.transform(df_val)

    # Ré-attacher la cible si présente
    if y_train is not None:
        df_train_imp[args.target] = y_train.values
    if y_val is not None:
        df_val_imp[args.target] = y_val.values

    # Sauvegarde des datasets imputés
    if args.format == "parquet":
        save_parquet(df_train_imp, df_val_imp, outdir)
    else:
        save_csv_with_dtypes(df_train_imp, df_val_imp, outdir)

    # Persistance de l'imputer + petite méta
    dump(imputer, artifacts / "imputer.joblib")
    meta = {
        "target": args.target,
        "use_cohort": args.use_cohort,
        "missing_flag": args.missing_flag,
        "features": list(df_train_imp.drop(columns=[args.target]).columns) if args.target else list(df_train_imp.columns),
        "output_dir": str(outdir.resolve()),
        "format": args.format,
    }
    (artifacts / "imputer_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"✔ Imputer sauvegardé dans {artifacts / 'imputer.joblib'}")


if __name__ == "__main__":
    main()
