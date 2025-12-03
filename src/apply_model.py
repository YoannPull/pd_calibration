#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APPLY MODEL — FULL PIPELINE (OOS)
=================================

Objectif :
- Rejouer toute la pipeline pour un nouveau fichier (ex: oos.parquet)
  à partir des artefacts appris sur le train :

  1) Imputation          (imputer.joblib)
  2) Binning             (bins.json -> load_bins_json + transform_with_learned_bins)
  3) WOE + interactions  (woe_maps dans model_best.joblib)
  4) Sélection features  (kept_features dans model_best.joblib)
  5) LR + calibration    (best_lr + model_pd dans model_best.joblib)
  6) Score TTC + grade   (risk_buckets.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# --- binning: on réutilise EXACTEMENT les helpers du module binning ---
try:
    from features.binning import load_bins_json, transform_with_learned_bins
except ImportError:
    # fallback si exécuté en mode script sans PYTHONPATH=src explicite
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from features.binning import load_bins_json, transform_with_learned_bins


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Score scaling (même que train_model.py)
# ---------------------------------------------------------------------------

def scale_score(log_odds, base_points=600, base_odds=50, pdo=20):
    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)
    return np.round(offset - factor * log_odds).astype(int)


# ---------------------------------------------------------------------------
# Interactions (copié de train_model.py)
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
# Apply WOE (même logique que train_model.py)
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
            # Pas de variable binned correspondante -> WOE global par défaut
            cols.append(pd.Series(default, index=df.index, name=f"{raw}_WOE"))
        else:
            cols.append(df[colbin].map(mapping).fillna(default).rename(f"{raw}_WOE"))

    return pd.concat(cols, axis=1)


# ---------------------------------------------------------------------------
# Buckets -> grade
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Buckets -> grade
# ---------------------------------------------------------------------------

def apply_buckets(scores: np.ndarray, edges: list[float]) -> np.ndarray:
    """
    Affecte un grade de risque à partir des scores TTC et des edges de la master scale.

    Convention (alignée sur train_model.py) :
      - grade = 1 : moins risqué (scores les plus ÉLEVÉS)
      - grade = n : plus risqué (scores les plus FAIBLES)
    """
    # Bucket "brut" : 1 = scores les plus faibles (plus risqués)
    raw_bucket = np.digitize(scores, edges[1:], right=True) + 1

    # Nombre total de buckets (len(edges) = n_buckets + 1)
    n_buckets = len(edges) - 1

    # Convention finale : 1 = moins risqué, n = plus risqué
    grade = n_buckets + 1 - raw_bucket
    return grade



# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Apply full credit-risk pipeline on new data (OOS).")
    p.add_argument("--data", required=True, help="Fichier OOS brut ou labellisé (parquet/csv).")
    p.add_argument("--out", required=True, help="Fichier de sortie scored (parquet/csv selon extension).")
    p.add_argument("--imputer", required=True, help="imputer.joblib appris sur le train.")
    p.add_argument("--bins", required=True, help="bins.json (sérialisé via save_bins_json).")
    p.add_argument("--model", required=True, help="model_best.joblib (LR + calibration + WOE).")
    p.add_argument("--buckets", required=True, help="risk_buckets.json (master scale).")
    p.add_argument("--target", default=None, help="Nom de la colonne cible si présente (optionnel).")
    p.add_argument("--id-col", default="loan_sequence_number", help="Identifiant de ligne (pour la sortie).")
    return p.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    print(f"[INFO] Loading raw data: {data_path}")
    df_raw = load_any(str(data_path))

    # ------------------------
    # 1. Imputation
    # ------------------------
    print(f"[INFO] Loading imputer: {args.imputer}")
    imputer = joblib.load(args.imputer)

    print("[INFO] Applying imputation (same transformer as impute_and_save.py)...")
    X_imp = imputer.transform(df_raw)

    if isinstance(X_imp, pd.DataFrame):
        df_imp = X_imp
    else:
        if hasattr(imputer, "feature_names_in_"):
            cols = list(imputer.feature_names_in_)
        else:
            cols = list(df_raw.columns)
        df_imp = pd.DataFrame(X_imp, columns=cols, index=df_raw.index)

    # ------------------------
    # 2. Binning : REUSE transform_with_learned_bins
    # ------------------------
    print(f"[INFO] Loading learned bins from JSON: {args.bins}")
    learned_bins = load_bins_json(args.bins)

    print("[INFO] Applying learned bins to imputed data (transform_with_learned_bins)...")
    df_binned = transform_with_learned_bins(df_imp, learned_bins)
    bin_suffix = learned_bins.bin_col_suffix

    # ------------------------
    # 3. WOE + interactions
    # ------------------------
    print(f"[INFO] Loading model artifacts: {args.model}")
    pkg = joblib.load(args.model)
    best_lr = pkg["best_lr"]
    model_pd = pkg["model_pd"]
    woe_maps = pkg["woe_maps"]
    kept_features = pkg["kept_features"]

    print("[INFO] Applying WOE transformation...")
    df_woe = apply_woe(df_binned, woe_maps, bin_suffix=bin_suffix)
    df_woe = add_interactions(df_woe)

    # On ne garde que les features retenues au training
    X = df_woe[kept_features].fillna(0).values

    # ------------------------
    # 4. Score TTC + PD + grade
    # ------------------------
    print("[INFO] Computing scores and PD...")
    log_odds = best_lr.decision_function(X)
    score_ttc = scale_score(log_odds)

    pd_hat = model_pd.predict_proba(X)[:, 1]

    print(f"[INFO] Loading buckets (master scale): {args.buckets}")
    edges = json.loads(Path(args.buckets).read_text())["edges"]
    grade = apply_buckets(score_ttc, edges)

    # ------------------------
    # 5. Construction du DataFrame de sortie
    # ------------------------
    print("[INFO] Building output DataFrame...")

    out_df = pd.DataFrame(index=df_raw.index)

    # ID colonne
    if args.id_col in df_raw.columns:
        out_df[args.id_col] = df_raw[args.id_col]
    else:
        out_df[args.id_col] = df_raw.index

    # Propage vintage si présent
    if "vintage" in df_raw.columns:
        out_df["vintage"] = df_raw["vintage"]

    # Variables de score
    out_df["score_ttc"] = score_ttc
    out_df["pd"] = pd_hat
    out_df["grade"] = grade

    # Cible brute (non touchée par l'imputer)
    if args.target and args.target in df_raw.columns:
        out_df[args.target] = df_raw[args.target]

    # ------------------------
    # 6. Sauvegarde
    # ------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix in (".parquet", ".pq"):
        out_df.to_parquet(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False)

    print(f"✔ Scoring terminé. Résultat sauvegardé dans : {out_path}")


if __name__ == "__main__":
    main()
