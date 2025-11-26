#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APPLY MODEL — CLEAN VERSION
===========================

Fonction :
- Applique le modèle issu de train_model.py
- Applique WOE depuis les maps
- Ajoute interactions
- Génère log-odds, score TTC, PD calibrée, grade
- Compatible exactly with Makefile

"""

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd
import numpy as np
import joblib


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if "credit_score_WOE" in df.columns and "original_cltv_WOE" in df.columns:
        df["inter_score_x_cltv"] = df["credit_score_WOE"] * df["original_cltv_WOE"]

    if "credit_score_WOE" in df.columns and "original_dti_WOE" in df.columns:
        df["inter_score_x_dti"] = df["credit_score_WOE"] * df["original_dti_WOE"]

    if "original_cltv_WOE" in df.columns and "original_dti_WOE" in df.columns:
        df["inter_cltv_x_dti"] = df["original_cltv_WOE"] * df["original_dti_WOE"]

    return df


# ---------------------------------------------------------------------------
# Score scaling (same formula as train)
# ---------------------------------------------------------------------------

def scale_score(log_odds, base_points=600, base_odds=50, pdo=20):
    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)
    return np.round(offset - factor * log_odds).astype(int)


# ---------------------------------------------------------------------------
# Apply WOE
# ---------------------------------------------------------------------------

def apply_woe(df: pd.DataFrame, woe_maps: dict, bin_suffix: str) -> pd.DataFrame:
    cols = []
    for raw, info in woe_maps.items():
        bin1 = f"{bin_suffix}{raw}"
        bin2 = f"{raw}{bin_suffix}"

        colbin = bin1 if bin1 in df.columns else (bin2 if bin2 in df.columns else None)

        default = float(info["default"])
        mapping = {int(k): float(v) for k, v in info["map"].items()}

        if colbin is None:
            cols.append(pd.Series(default, index=df.index, name=f"{raw}_WOE"))
        else:
            cols.append(df[colbin].map(mapping).fillna(default).rename(f"{raw}_WOE"))

    return pd.concat(cols, axis=1)


# ---------------------------------------------------------------------------
# Apply bucket edges
# ---------------------------------------------------------------------------

def apply_buckets(scores, edges):
    return np.digitize(scores, edges[1:], right=True) + 1


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--imputer", required=True)   # ignored (data already binned)
    p.add_argument("--bins", required=True)      # ignored (we only use woe_maps)
    p.add_argument("--model", required=True)
    p.add_argument("--buckets", required=True)
    p.add_argument("--target", default=None)
    p.add_argument("--id-col", default="loan_sequence_number")
    args = p.parse_args()

    # ------------------------
    # Load data
    # ------------------------
    path = Path(args.data)
    df = pd.read_parquet(path) if path.suffix in [".parquet", ".pq"] else pd.read_csv(path)

    # ------------------------
    # Load model artifacts
    # ------------------------
    pkg = joblib.load(args.model)
    best_lr = pkg["best_lr"]
    model_pd = pkg["model_pd"]
    woe_maps = pkg["woe_maps"]
    kept_features = pkg["kept_features"]

    edges = json.loads(Path(args.buckets).read_text())["edges"]

    # ------------------------
    # Apply WOE
    # ------------------------
    df_woe = apply_woe(df, woe_maps, "__BIN")
    df_woe = add_interactions(df_woe)

    X = df_woe[kept_features].fillna(0)

    # ------------------------
    # Score
    # ------------------------
    log_odds = best_lr.decision_function(X)
    score = scale_score(log_odds)

    pd_hat = model_pd.predict_proba(X)[:, 1]
    grade = apply_buckets(score, edges)

    # ------------------------
    # Output
    # ------------------------
    out = pd.DataFrame({
        args.id_col: df[args.id_col] if args.id_col in df.columns else df.index,
        "score": score,
        "pd": pd_hat,
        "grade": grade,
    })

    if args.target and args.target in df.columns:
        out[args.target] = df[args.target]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"✔ Output saved: {args.out}")


if __name__ == "__main__":
    main()
