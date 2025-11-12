#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Impute train/validation with DataImputer, then save either:
- Parquet files (types conservés nativement), or
- CSV + pickles des dtypes (parse_dates, cat_dtypes, other_dtypes)

Nouveautés :
- --labels-window-dir + --use-splits : lit window=.../_splits.json pour construire
  automatiquement train (pooled.parquet) et validation (liste de quarters ou oos).
- Supporte plusieurs quarters de validation (concat).
- Rétro-compatible avec --train-csv / --validation-csv.

Sorties :
- data/processed/imputed/{train.parquet, validation.parquet} (ou CSV)
- artifacts/imputer/imputer.joblib + imputer_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from joblib import dump

# Rendez le package importable depuis ./src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features.impute import DataImputer  # adapte si besoin


# ----------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Impute & save datasets + persist imputer")

    # Mode explicite (rétro-compatible)
    p.add_argument("--train-csv", help="Fichier d'entraînement (CSV/Parquet)")
    p.add_argument("--validation-csv", help="Fichier de validation (CSV/Parquet)")

    # Mode basé sur les splits produits par make_labels.py
    p.add_argument(
        "--labels-window-dir",
        help="Chemin du dossier window=XXm (ex: data/processed/default_labels/window=24m)"
    )
    p.add_argument(
        "--use-splits",
        action="store_true",
        help="Utiliser window/_splits.json pour construire train/validation"
    )

    p.add_argument("--target", default=None, help="Nom de la colonne cible (optionnel)")
    p.add_argument("--outdir", default="data/processed/imputed", help="Dossier de sortie")
    p.add_argument(
        "--artifacts",
        default="artifacts/imputer",
        help="Dossier où sauvegarder l'imputer et la méta"
    )
    p.add_argument(
        "--format", choices=["parquet", "csv"], default="parquet",
        help="Format de sauvegarde des jeux imputés"
    )
    p.add_argument("--use-cohort", action="store_true", help="Imputation par cohortes")
    p.add_argument("--missing-flag", action="store_true", help="Ajouter les indicateurs was_missing_*")
    p.add_argument("--fail-on-nan", action="store_true", help="Échouer si des NaN subsistent après imputation")

    return p.parse_args()


# ----------------------------- I/O helpers -----------------------------
def load_any(path: str | Path) -> pd.DataFrame:
    """Charge un DataFrame depuis Parquet ou CSV (déduit de l'extension)."""
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def concat_parquet(paths: List[Path]) -> pd.DataFrame:
    """Concatène une liste de fichiers parquet (schema-align si nécessaire)."""
    if not paths:
        return pd.DataFrame()
    dfs = []
    for p in paths:
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        return pd.DataFrame()
    all_cols = sorted(set().union(*[df.columns for df in dfs]))
    dfs = [df.reindex(columns=all_cols) for df in dfs]
    return pd.concat(dfs, ignore_index=True)


def save_parquet(df_train_imp: pd.DataFrame, df_val_imp: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "train.parquet").unlink(missing_ok=True)
    (outdir / "validation.parquet").unlink(missing_ok=True)
    df_train_imp.to_parquet(outdir / "train.parquet", index=False)
    df_val_imp.to_parquet(outdir / "validation.parquet", index=False)
    print(f"✔ Parquet écrit dans {outdir}")


def save_csv_with_dtypes(df_train_imp: pd.DataFrame, df_val_imp: pd.DataFrame, outdir: Path):
    from pandas.api.types import CategoricalDtype  # import local pour cohérence pickle

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
    print(f"✔ CSV + dtypes picklés écrits dans {outdir}")


# ----------------------------- Splits loader -----------------------------
def _pull(d: dict, key: str, default=None):
    """Cherche une clé à la racine, sinon dans d['splits'], sinon default."""
    if key in d:
        return d[key]
    s = d.get("splits", {})
    return s.get(key, default)


def resolve_splits(labels_window_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Construit (train_df, val_df) à partir de window/_splits.json :
      - train = pooled.parquet
      - validation = concat(quartiers listés) OU oos.parquet selon 'validation_mode'
    Supporte les manifests où les clés sont à la racine OU sous "splits".
    """
    splits_path = labels_window_dir / "_splits.json"
    pooled_path = labels_window_dir / "pooled.parquet"
    oos_path = labels_window_dir / "oos.parquet"

    if not splits_path.exists():
        raise SystemExit(f"[ERR] Splits file not found: {splits_path}")

    manifest = json.loads(splits_path.read_text(encoding="utf-8"))

    # Train
    if not pooled_path.exists():
        raise SystemExit(f"[ERR] Train 'pooled.parquet' not found at {pooled_path}")
    df_train = load_any(pooled_path)

    # Récupération robuste des infos de validation
    mode = (_pull(manifest, "validation_mode", None) or "quarters").lower()

    # 1) validation_quarters normalisés
    val_quarters = _pull(manifest, "validation_quarters", [])
    # 2) rétro-compat via splits.explicit
    explicit = _pull(manifest, "explicit", {}) or {}
    if not val_quarters:
        val_quarters = list(explicit.get("validation_quarters", []))
    if not val_quarters:
        val_quarters = list(explicit.get("default_val_quarters", []))
    if not val_quarters and explicit.get("default_val_quarter"):
        val_quarters = [explicit["default_val_quarter"]]

    # 3) oos_quarters
    oos_quarters = _pull(manifest, "oos_quarters", [])
    if not oos_quarters:
        oos_quarters = list(explicit.get("oos_quarters", []))

    # Si mode=quarters mais aucune liste fournie, on bascule intelligemment
    if mode == "quarters" and not val_quarters:
        if oos_path.exists() and oos_quarters:
            print("[WARN] Pas de validation_quarters trouvés → fallback sur OOS.")
            mode = "oos"
        else:
            raise SystemExit("[ERR] No 'validation_quarters' found and no oos fallback available.")

    # Build validation
    if mode == "oos":
        if not oos_path.exists():
            raise SystemExit(f"[ERR] Validation 'oos.parquet' not found at {oos_path}")
        df_val = load_any(oos_path)
        used_quarters = oos_quarters
    else:
        files = [labels_window_dir / f"quarter={q}" / "data.parquet" for q in val_quarters]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise SystemExit(f"[ERR] Missing validation quarter files:\n  " + "\n  ".join(missing))
        df_val = concat_parquet(files)
        used_quarters = val_quarters

    # Align colonnes
    all_cols = sorted(set(df_train.columns).union(df_val.columns))
    df_train = df_train.reindex(columns=all_cols)
    df_val = df_val.reindex(columns=all_cols)

    # Logs utiles
    print(f"[SPLITS] mode: {mode}")
    if mode == "quarters":
        print(f"[SPLITS] validation_quarters: {used_quarters}")
    else:
        print(f"[SPLITS] oos_quarters: {used_quarters}")

    # Méta enrichie
    meta = {
        "train_file": str(pooled_path),
        "validation_mode": mode,
        "validation_used_quarters": used_quarters,
        "validation_paths": (
            [str(labels_window_dir / f"quarter={q}" / "data.parquet") for q in used_quarters]
            if mode != "oos" else [str(oos_path)]
        )
    }
    # Conserver le manifest brut pour traçabilité
    manifest["_resolved"] = meta
    return df_train, df_val, manifest


# ----------------------------- Main -----------------------------
def main():
    args = parse_args()

    outdir = Path(args.outdir)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    # ----------------- Choix de la source (explicite vs splits) -----------------
    df_train: Optional[pd.DataFrame] = None
    df_val: Optional[pd.DataFrame] = None
    splits_meta: Optional[dict] = None

    if args.use_splits:
        if not args.labels_window_dir:
            raise SystemExit("[ERR] --use-splits nécessite --labels-window-dir")
        labels_dir = Path(args.labels_window_dir)
        if not labels_dir.exists():
            raise SystemExit(f"[ERR] labels_window_dir introuvable : {labels_dir}")
        df_train, df_val, splits_meta = resolve_splits(labels_dir)
    else:
        if not args.train_csv or not args.validation_csv:
            raise SystemExit("[ERR] Fournir --train-csv et --validation-csv, ou utiliser --use-splits avec --labels-window-dir")
        df_train = load_any(args.train_csv)
        df_val = load_any(args.validation_csv)

    # ----------------- Séparer (optionnel) la cible -----------------
    y_train = y_val = None
    if args.target and args.target in df_train.columns:
        y_train = df_train[args.target].copy()
        df_train = df_train.drop(columns=[args.target])
    if args.target and args.target in df_val.columns:
        y_val = df_val[args.target].copy()
        df_val = df_val.drop(columns=[args.target])

    # ----------------- Imputation -----------------
    imputer = DataImputer(use_cohort=args.use_cohort, missing_flag=args.missing_flag)
    imputer.fit(df_train)
    df_train_imp = imputer.transform(df_train)
    df_val_imp = imputer.transform(df_val)

    # Ré-attacher la cible si présente
    if y_train is not None:
        df_train_imp[args.target] = y_train.values
    if y_val is not None:
        df_val_imp[args.target] = y_val.values

    # Optionnel : garde-fou qualité
    if args.fail_on_nan:
        bad_train = df_train_imp.columns[df_train_imp.isna().any()].tolist()
        bad_val = df_val_imp.columns[df_val_imp.isna().any()].tolist()
        if bad_train or bad_val:
            raise SystemExit(
                f"NaN résiduels après imputation.\n"
                f"Train: {bad_train}\nValidation: {bad_val}"
            )

    # ----------------- Sauvegarde des datasets -----------------
    if args.format == "parquet":
        save_parquet(df_train_imp, df_val_imp, outdir)
    else:
        save_csv_with_dtypes(df_train_imp, df_val_imp, outdir)

    # ----------------- Persistance imputer + méta -----------------
    dump(imputer, artifacts / "imputer.joblib")

    meta = {
        "target": args.target,
        "use_cohort": args.use_cohort,
        "missing_flag": args.missing_flag,
        "features": list(df_train_imp.drop(columns=[args.target]).columns) if args.target else list(df_train_imp.columns),
        "output_dir": str(outdir.resolve()),
        "format": args.format,
        "mode": "splits" if args.use_splits else "explicit",
    }

    if args.use_splits and splits_meta is not None:
        meta["splits"] = splits_meta
    else:
        meta["train_path"] = str(Path(args.train_csv).resolve())
        meta["validation_path"] = str(Path(args.validation_csv).resolve())

    (artifacts / "imputer_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"✔ Imputer sauvegardé dans {artifacts / 'imputer.joblib'}")


if __name__ == "__main__":
    main()
