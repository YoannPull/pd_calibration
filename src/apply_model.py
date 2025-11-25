#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/apply_model.py
------------------
Application du Modèle de Risque.
Logique corrigée : Ne fait plus l'Imputation/Binning si l'entrée est déjà binée.
Recalage corrigé : Préserve la courbe Isotonique apprise.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import joblib

# Import des modules internes
try:
    from features.impute import DataImputer
    from features.binning import load_bins_json, transform_with_learned_bins
except ImportError:
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from features.impute import DataImputer
    from features.binning import load_bins_json, transform_with_learned_bins


# ==============================================================================
# LOGIQUE ADD_INTERACTIONS (Répliquée de train_model.py)
# ==============================================================================
def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des variables d'interaction clés pour le risque de crédit.
    Multiplie les WOE pour capturer les risques conjoints (ex: Haut LTV ET Mauvais Score).
    """
    if 'credit_score_WOE' in df.columns and 'original_cltv_WOE' in df.columns:
        df['inter_score_x_cltv'] = df['credit_score_WOE'] * df['original_cltv_WOE']
        
    if 'credit_score_WOE' in df.columns and 'original_dti_WOE' in df.columns:
        df['inter_score_x_dti'] = df['credit_score_WOE'] * df['original_dti_WOE']

    if 'original_cltv_WOE' in df.columns and 'original_dti_WOE' in df.columns:
        df['inter_cltv_x_dti'] = df['original_cltv_WOE'] * df['original_dti_WOE']

    return df

# ==============================================================================
# LOGIQUE SCALING
# ==============================================================================
def scale_score(log_odds: np.ndarray, base_points=600, base_odds=50, pdo=20) -> np.ndarray:
    """Convertit les log-odds en points."""
    factor = pdo / np.log(2)
    offset = base_points - (factor * np.log(base_odds))
    scores = offset - (factor * log_odds)
    return np.round(scores).astype(int)

# ==============================================================================
# LOGIQUE WOE
# ==============================================================================
def apply_woe_from_artifact(df: pd.DataFrame, woe_maps: Dict[str, Any], bin_suffix: str) -> pd.DataFrame:
    woe_cols = []
    for raw_feat, map_info in woe_maps.items():
        cand_p = f"{bin_suffix}{raw_feat}"
        cand_s = f"{raw_feat}{bin_suffix}"
        
        col_bin = None
        if cand_p in df.columns: col_bin = cand_p
        elif cand_s in df.columns: col_bin = cand_s
        
        if col_bin is None:
            woe_val = float(map_info["default"])
            woe_cols.append(pd.Series(woe_val, index=df.index, name=f"{raw_feat}_WOE"))
            continue

        mapping = {int(k): float(v) for k, v in map_info["map"].items()}
        default_woe = float(map_info["default"])
        # On applique le mapping WOE
        series_woe = df[col_bin].map(mapping).fillna(default_woe).astype(float) 
        series_woe.name = f"{raw_feat}_WOE"
        woe_cols.append(series_woe)
        
    return pd.concat(woe_cols, axis=1)

# ==============================================================================
# LOGIQUE BUCKETING
# ==============================================================================
def apply_buckets(scores: np.ndarray, edges: List[float]) -> np.ndarray:
    return np.digitize(scores, edges[1:], right=True) + 1

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imputer", required=True)
    parser.add_argument("--bins", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--buckets", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--id-col", default="loan_sequence_number")
    
    # ARGUMENT : Re-alignement
    parser.add_argument("--realign-to", type=float, default=None, 
                        help="Force la moyenne des scores à cette valeur (ex: 600) pour corriger le drift.")

    args = parser.parse_args()
    start_time = time.time()
    
    # 1. Chargement
    print(f"-> Chargement : {args.data}")
    p = Path(args.data)
    df = pd.read_parquet(p) if p.suffix in [".parquet", ".pq"] else pd.read_csv(p)

    # 2. Artefacts
    # Pas besoin de charger l'imputer/bins si l'input est déjà biné, 
    # mais gardons le chargement pour la cohérence des paths si on changeait d'avis plus tard.
    imputer = joblib.load(args.imputer) 
    learned_bins = load_bins_json(args.bins)
    model_pkg = joblib.load(args.model)
    with open(args.buckets, 'r') as f:
        bucket_edges = json.load(f)["edges"]

    best_lr = model_pkg["best_lr"]
    model_pd = model_pkg["model_pd"]
    woe_maps = model_pkg["woe_maps"]
    kept_features = model_pkg["kept_features"]
    
    # Définition des constantes de scaling pour le recalage
    pdo = 20
    factor = pdo / np.log(2)

    # 3. Transform
    print("-> Transformations (WOE & Interactions uniquement)...")
    
    # ⚠️ Nous supposons que les données d'entrée sont data/processed/binned/*.parquet.
    # L'imputation et le binning ne sont donc PAS refaits ici.
    df_woe = apply_woe_from_artifact(df.copy(), woe_maps, learned_bins.bin_col_suffix)
    df_woe = add_interactions(df_woe)
    
    try:
        X = df_woe[kept_features].fillna(0.0) # fillna(0) pour la sécurité des interactions
    except KeyError:
        missing = [f for f in kept_features if f not in df_woe.columns]
        print(f"[FATAL ERR] Colonnes WOE manquantes ou mal nommées pour le modèle. Manquantes: {missing}")
        sys.exit(1)

    # 4. Scoring
    print("-> Scoring...")
    raw_log_odds = best_lr.decision_function(X)
    
    # --- LOGIQUE DE RE-ALIGNEMENT (CORRIGÉE) ---
    if args.realign_to is not None:
        
        scores_before_shift = scale_score(raw_log_odds)
        
        current_mean = np.mean(scores_before_shift)
        shift_points = args.realign_to - current_mean
        shift_log_odds = - (shift_points / factor)
        
        print(f"\n⚡ RE-ALIGNEMENT ACTIF")
        print(f"   Moyenne actuelle (Score) : {current_mean:.2f}")
        print(f"   Cible (Score)            : {args.realign_to:.2f}")
        print(f"   Shift Log-Odds appliqué : {shift_log_odds:+.4f}")
        
        raw_log_odds = raw_log_odds + shift_log_odds
        
    # Scores final recalé (ou non)
    scores = scale_score(raw_log_odds) 
    
    # Application du Calibrateur Isotonique (model_pd) sur les log-odds recalés
    # Ceci utilise le modèle CalibratedClassifierCV.
    pds = model_pd.predict_proba(X)[:, 1]

    # 5. Bucketing
    grades = apply_buckets(scores, bucket_edges)

    # 6. Export
    out = pd.DataFrame()
    out[args.id_col] = df[args.id_col] if args.id_col in df.columns else df.index
    out["score"] = scores
    out["pd"] = pds
    out["grade"] = grades
    
    if args.target and args.target in df.columns:
        out[args.target] = df[args.target]
        # Calc perf
        from sklearn.metrics import roc_auc_score
        mask = out[args.target].notna()
        if mask.sum() > 0:
            try:
                auc = roc_auc_score(out.loc[mask, args.target], out.loc[mask, "pd"])
                print(f"\n--- PERF (AUC) : {auc:.4f} ---")
            except: pass

    p_out = Path(args.out)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p_out, index=False)
    print(f"✔ Fichier généré : {p_out}")

if __name__ == "__main__":
    main()