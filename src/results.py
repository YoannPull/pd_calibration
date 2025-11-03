#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
results.py — Vintage × Risk Grade report

Pour chaque vintage et chaque grade (classe de risque), le script retourne :
- grade (label)
- bornes de la classe [lower, upper]
- effectif (n)
- nb de défauts (n_default) si la colonne cible est fournie
- taux de défaut (default_rate) si la colonne cible est fournie
- probabilité de défaut de la classe issue de l'entraînement (class_pd_train), si présente dans risk_buckets.json
- moyenne des probabilités prédites (mean_proba)
- (optionnel) n_train : effectif d'entraînement par grade si présent dans risk_buckets.json

Entrées attendues :
- un dataset "scored" avec au minimum : 'proba' et 'vintage'
- un fichier JSON de buckets (edges/labels/etc.) sauvegardé à l'entraînement

Exemple :
    python results.py \
        --data data/processed/scored/test_scored.parquet \
        --buckets artifacts/model/risk_buckets.json \
        --target default_24m \
        --out reports/by_vintage_grades.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------ I/O utils ------------------------------
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


# ------------------------------ Buckets utils ------------------------------
def load_buckets(buckets_path: str) -> Dict:
    """
    Attend un JSON de la forme (schéma souple, champs optionnels) :
    {
      "edges": [ ... ],                   # len = K+1 (obligatoire)
      "labels": ["G01", ...],             # len = K (optionnel)
      "bucket_pd_train": [p1..pK],        # PD de classe sur TRAIN/OOF (optionnel)
      "bucket_pd": [p1..pK],              # alias possible (optionnel)
      "pd_train": [p1..pK],               # alias possible (optionnel)
      "class_pd": [p1..pK],               # alias possible (optionnel)
      "bucket_n_train": [n1..nK]          # effectifs TRAIN par classe (optionnel)
    }
    """
    with open(buckets_path, "r") as f:
        cfg = json.load(f)

    if "edges" not in cfg or not isinstance(cfg["edges"], list) or len(cfg["edges"]) < 2:
        raise ValueError("risk_buckets.json: 'edges' manquant ou invalide.")

    edges = np.asarray(cfg["edges"], dtype=float)
    if not np.all(np.diff(edges) >= 0):
        raise ValueError("risk_buckets.json: 'edges' doivent être triés et non décroissants.")

    K = len(edges) - 1

    labels = cfg.get("labels")
    if labels is None or len(labels) != K:
        labels = [f"G{str(i+1).zfill(2)}" for i in range(K)]

    # PD de classe issue de l'entraînement (accepte plusieurs clés)
    pd_train = (
        cfg.get("bucket_pd_train")
        or cfg.get("bucket_pd")
        or cfg.get("pd_train")
        or cfg.get("class_pd")
    )
    if pd_train is not None:
        if len(pd_train) != K:
            raise ValueError("risk_buckets.json: longueur de 'bucket_pd*_train' incompatible avec edges.")
        pd_train = np.asarray(pd_train, dtype=float)

    # Effectifs d'entraînement par classe (optionnel)
    n_train = cfg.get("bucket_n_train")
    if n_train is not None:
        if len(n_train) != K:
            raise ValueError("risk_buckets.json: longueur de 'bucket_n_train' incompatible avec edges.")
        n_train = np.asarray(n_train, dtype=int)

    return {
        "edges": edges,
        "labels": labels,
        "pd_train": pd_train,   # peut être None
        "n_train": n_train,     # peut être None
        "n_bins": K,
    }


def assign_bucket_indices(proba: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Assigne chaque probabilité au bin correspondant, avec la convention right=True :
        bins[i-1] < x <= bins[i]
    Retourne des indices de 0 à K-1.
    """
    idx = np.digitize(proba, edges, right=True) - 1
    # clamp (sécurité pour valeurs extrêmes ou proba==edges[0])
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx


# ------------------------------ Vintage sorting ------------------------------
def _parse_vintage(v: str) -> Tuple[int, int]:
    """
    Convertit 'YYYYQn' en (YYYY, n) pour trier correctement.
    Si parsing impossible, renvoie (très grand) pour pousser en bas.
    """
    try:
        year = int(v[:4])
        q = int(v[-1])
        return year, q
    except Exception:
        return (10**9, 9)


# ------------------------------ Core reporting ------------------------------
def build_bucket_meta(bkt: Dict) -> pd.DataFrame:
    edges = bkt["edges"]
    labels = bkt["labels"]
    pd_train = bkt.get("pd_train")
    n_train = bkt.get("n_train")
    K = len(labels)

    lower = edges[:-1]
    upper = edges[1:]

    meta = pd.DataFrame({
        "bucket_index": np.arange(K, dtype=int),
        "grade": labels,
        "lower": lower,
        "upper": upper
    })

    meta["class_pd_train"] = np.nan if pd_train is None else pd_train
    if n_train is not None:
        meta["n_train"] = n_train

    return meta


def make_report(
    df: pd.DataFrame,
    bkt: Dict,
    proba_col: str = "proba",
    vintage_col: str = "vintage",
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Produit un DataFrame agrégé :
    [vintage, grade, bucket_index, lower, upper, class_pd_train, (n_train?), n, n_default, default_rate, mean_proba]
    """
    if proba_col not in df.columns:
        raise ValueError(f"Colonne '{proba_col}' absente du dataset.")

    if vintage_col not in df.columns:
        raise ValueError(f"Colonne '{vintage_col}' absente du dataset.")

    edges = bkt["edges"]
    idx = assign_bucket_indices(df[proba_col].to_numpy(dtype=float), edges)

    meta = build_bucket_meta(bkt)

    work = df.copy()
    work["bucket_index"] = idx

    if target_col and target_col in work.columns:
        y = work[target_col].astype(int)
        work = work.assign(_y=y)
        agg = work.groupby([vintage_col, "bucket_index"], as_index=False).agg(
            n=("bucket_index", "size"),
            n_default=("_y", "sum"),
            mean_proba=(proba_col, "mean"),
        )
        agg["default_rate"] = agg["n_default"] / agg["n"]
    else:
        agg = work.groupby([vintage_col, "bucket_index"], as_index=False).agg(
            n=("bucket_index", "size"),
            mean_proba=(proba_col, "mean"),
        )
        agg["n_default"] = np.nan
        agg["default_rate"] = np.nan

    # jointure avec les métadonnées de classe
    rep = agg.merge(meta, on="bucket_index", how="left")

    # tri par vintage chronologique puis par bucket
    rep["__v_key"] = rep[vintage_col].astype(str).map(_parse_vintage)
    rep = rep.sort_values(["__v_key", "bucket_index"]).drop(columns="__v_key")

    # colonnes finales (inclut n_train si présent)
    base_cols = [
        vintage_col, "grade", "bucket_index",
        "lower", "upper", "class_pd_train",
    ]
    if "n_train" in rep.columns:
        base_cols.append("n_train")
    base_cols += ["n", "n_default", "default_rate", "mean_proba"]
    rep = rep[base_cols]

    # arrondis élégants
    for c in ("lower", "upper", "class_pd_train", "default_rate", "mean_proba"):
        if c in rep.columns:
            rep[c] = rep[c].astype(float).round(6)

    return rep


# ------------------------------ CLI ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Vintage × grade report à partir d'un dataset scoré et d'un fichier de buckets.")
    p.add_argument("--data", required=True, help="CSV/Parquet contenant au minimum 'proba' et 'vintage'.")
    p.add_argument("--buckets", required=True, help="risk_buckets.json sauvegardé à l'entraînement (edges, labels, bucket_pd_train...).")
    p.add_argument("--target", default=None, help="Nom de la colonne cible binaire (ex: default_24m).")
    p.add_argument("--proba-col", default="proba", help="Nom de la colonne des probabilités.")
    p.add_argument("--vintage-col", default="vintage", help="Nom de la colonne vintage (format 'YYYYQn').")
    p.add_argument("--out", default=None, help="Chemin de sortie (CSV/Parquet). Si non fourni, imprime sur stdout.")
    return p.parse_args()


def main():
    args = parse_args()

    df = load_any(args.data)
    buckets = load_buckets(args.buckets)

    report = make_report(
        df=df,
        bkt=buckets,
        proba_col=args.proba_col,
        vintage_col=args.vintage_col,
        target_col=args.target,
    )

    if args.out:
        save_any(report, args.out)
        print(f"✔ Report written to: {args.out}")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 160, "display.max_columns", None):
            print(report)


if __name__ == "__main__":
    main()
