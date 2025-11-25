#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entraînement Modèle Risque de Crédit (Bank-Grade) - Version Améliorée.
- Robustesse : Augmentation de l'espace de régularisation (C).
- Stabilité : Utilisation de class_weight='balanced'.
- Calibration : Calibration honnête sur le split WOE (Part 1).
- Audit : Ajout du Brier Score.
"""

from __future__ import annotations
import argparse
import json
import time
import contextlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

# ==============================================================================
# 1. UTILITAIRES I/O & TIMING
# ==============================================================================
def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable : {p}")
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)

def save_json(obj, path: Path):
    """Sauvegarde JSON robuste."""
    def default_fmt(o):
        if isinstance(o, (np.integer, np.int64, np.int32)): return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default_fmt), encoding="utf-8")

class Timer:
    def __init__(self, live: bool = False):
        self.records: Dict[str, float] = {}
        self.live = live

    @contextlib.contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        if self.live:
            print(f"▶ {name} ...", file=sys.stdout, flush=True)
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.records[name] = self.records.get(name, 0.0) + dt
            if self.live:
                print(f"  ✓ {name:22s} {dt:8.3f}s", file=sys.stdout, flush=True)

# ==============================================================================
# 2. SCALING (LOG-ODDS -> POINTS)
# ==============================================================================
def scale_score(log_odds: np.ndarray, base_points=600, base_odds=50, pdo=20) -> np.ndarray:
    factor = pdo / np.log(2)
    offset = base_points - (factor * np.log(base_odds))
    scores = offset - (factor * log_odds)
    return np.round(scores).astype(int)

# ==============================================================================
# 3. WOE TRANSFORMATION
# ==============================================================================
def find_bin_columns(df: pd.DataFrame, tag: str) -> List[str]:
    return sorted([c for c in df.columns if c.startswith(tag) or c.endswith(tag)])

def raw_name_from_bin(col: str, tag: str) -> str:
    if col.startswith(tag): return col[len(tag):]
    if col.endswith(tag):  return col[:-len(tag)]
    return col

def build_woe_maps_from_bins(df: pd.DataFrame, target: str, raw_to_bin: Dict[str, str], smooth: float = 0.5) -> Dict:
    maps = {}
    y = df[target].astype(int)
    B_all, G_all = float(y.sum()), float(len(y) - y.sum())
    
    # Global default
    global_woe = float(np.log((B_all + smooth) / (G_all + smooth)))
    
    for raw, bcol in raw_to_bin.items():
        tab = df.groupby(bcol, dropna=True)[target].agg(["sum", "count"])
        if tab.empty:
            maps[raw] = {"map": {}, "default": global_woe}
            continue
            
        tab["good"] = tab["count"] - tab["sum"]
        B, G = float(tab["sum"].sum()), float(tab["good"].sum())
        
        bad_rate_i = (tab["sum"] + smooth) / (B + smooth * len(tab))
        good_rate_i = (tab["good"] + smooth) / (G + smooth * len(tab))
        
        woe_series = np.log(bad_rate_i / good_rate_i)
        
        maps[raw] = {
            "map": {int(k): float(v) for k, v in woe_series.items()},
            "default": global_woe
        }
    return maps

def apply_woe_with_maps(df: pd.DataFrame, maps: Dict, kept_vars_raw: List[str], bin_tag: str) -> pd.DataFrame:
    cols = []
    for raw in kept_vars_raw:
        cand_p, cand_s = f"{bin_tag}{raw}", f"{raw}{bin_tag}"
        bcol = cand_p if cand_p in df.columns else (cand_s if cand_s in df.columns else None)
        
        if bcol is None or raw not in maps:
            continue
            
        wmap = maps[raw]["map"]
        wdef = float(maps[raw]["default"])
        
        x = df[bcol].map(wmap).astype(float).fillna(wdef)
        cols.append(pd.Series(x, name=f"{raw}_WOE", index=df.index))
        
    if not cols:
        return pd.DataFrame(index=df.index)
    return pd.concat(cols, axis=1)

# ==============================================================================
# 4. INTERACTIONS (FEATURE ENGINEERING)
# ==============================================================================
def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des variables d'interaction clés pour le risque de crédit.
    Multiplie les WOE pour capturer les risques conjoints (ex: Haut LTV ET Mauvais Score).
    """
    # Liste des candidats à l'interaction (Noms exacts des colonnes WOE attendues)
    # Ces noms dépendent de tes données brutes + suffixe _WOE
    
    # 1. Interaction Score x LTV (Le classique)
    if 'credit_score_WOE' in df.columns and 'original_cltv_WOE' in df.columns:
        df['inter_score_x_cltv'] = df['credit_score_WOE'] * df['original_cltv_WOE']
        
    # 2. Interaction Score x DTI (Capacité de remboursement)
    if 'credit_score_WOE' in df.columns and 'original_dti_WOE' in df.columns:
        df['inter_score_x_dti'] = df['credit_score_WOE'] * df['original_dti_WOE']

    # 3. Interaction LTV x DTI (Fragilité structurelle)
    if 'original_cltv_WOE' in df.columns and 'original_dti_WOE' in df.columns:
        df['inter_cltv_x_dti'] = df['original_cltv_WOE'] * df['original_dti_WOE']

    return df

# ==============================================================================
# 5. FEATURE SELECTION
# ==============================================================================
def select_features_robust(X: pd.DataFrame, corr_thr: float = 0.85) -> List[str]:
    """
    Sélectionne les features. 
    Note : Les interactions seront gardées si elles ont une forte variance 
    et ne sont pas trop corrélées aux variables mères.
    """
    # On remplit les NaNs éventuels des interactions par 0 (neutre)
    X = X.fillna(0)
    
    # On préfère l'approche Gloutonne ici
    order = X.var().sort_values(ascending=False).index.tolist()
    kept = []
    corr_matrix = X.corr().abs()
    
    for col in order:
        if not kept:
            kept.append(col)
            continue
        existing_corr = corr_matrix.loc[col, kept]
        if existing_corr.max() < corr_thr:
            kept.append(col)
    return kept

# ==============================================================================
# 6. CORE TRAINING ROUTINE
# ==============================================================================
def train_pipeline(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    target: str,
    bin_suffix: str,
    corr_threshold: float,
    cv_folds: int,
    calibration_method: str,
    timer: Optional[Timer]
) -> Dict[str, Any]:

    tctx = (lambda name: timer.section(name)) if timer else (lambda name: contextlib.nullcontext())

    # --- A. Préparation & Sécurité ---
    with tctx("Data Prep"):
        # 1. Identification des variables sûres
        bin_cols = find_bin_columns(df_tr, bin_suffix)
        
        # BLACKLIST ANTI-LEAKAGE
        BLACKLIST_KEYWORDS = [
            "quarter", "year", "month", "vintage", "time", "date",
            "property_valuation_method", "first_payment", "maturity",
            "mi_cancellation", "interest_rate", "loan_sequence_number", "postal_code"
        ]
        
        def is_safe(col_name):
            raw = raw_name_from_bin(col_name, bin_suffix).lower()
            if raw.startswith("__"): return False
            for bad_word in BLACKLIST_KEYWORDS:
                if bad_word in raw: return False
            return True

        bin_cols = [c for c in bin_cols if is_safe(c)]
        if not bin_cols: raise ValueError("Aucune colonne bin détectée après filtrage.")
        raw_to_bin = {raw_name_from_bin(c, bin_suffix): c for c in bin_cols}
        
        print(f"  -> Features Candidates : {len(bin_cols)}")

    # --- B. SPLIT TRAIN (WOE vs MODEL) ---
    with tctx("Splitting Train"):
        # 50/50 Split pour intégrité du WOE
        df_woe_learn, df_model_learn = train_test_split(
            df_tr, test_size=0.5, stratify=df_tr[target], random_state=42
        )
        print(f"  -> Split appliqué : {len(df_woe_learn)} (WOE Fit) / {len(df_model_learn)} (Model Fit)")

    # --- C. Calcul WOE & Interactions ---
    with tctx("WOE & Interactions"):
        # 1. Learn WOE on Part 1
        woe_maps = build_woe_maps_from_bins(df_woe_learn, target, raw_to_bin)
        keep_raw = sorted(woe_maps.keys())
        
        # 2. Transform Part 2 (Model Fit Split)
        Xtr_woe_model = apply_woe_with_maps(df_model_learn, woe_maps, keep_raw, bin_suffix)
        Xtr_woe_model = add_interactions(Xtr_woe_model)
        y_tr_model = df_model_learn[target].values.astype(int)
        
        # 3. Transform Part 1 (Calibration Split)
        Xcalib_woe = apply_woe_with_maps(df_woe_learn, woe_maps, keep_raw, bin_suffix)
        Xcalib_woe = add_interactions(Xcalib_woe)
        y_calib = df_woe_learn[target].values.astype(int)
        
        # 4. Transform Validation
        Xva_woe_all = apply_woe_with_maps(df_va, woe_maps, keep_raw, bin_suffix)
        Xva_woe_all = add_interactions(Xva_woe_all)
        y_va = df_va[target].values.astype(int) if target in df_va.columns else None

    # --- D. Sélection Variables ---
    with tctx("Feature Selection"):
        kept_features = select_features_robust(Xtr_woe_model, corr_thr=corr_threshold)
        Xtr = Xtr_woe_model[kept_features]
        Xva = Xva_woe_all[kept_features]
        # Filtrer aussi le jeu de calibration pour garder les mêmes colonnes
        Xcalib = Xcalib_woe[kept_features] 
        print(f"  -> Features kept: {len(kept_features)} (incluant interactions)")

    # --- E. Régression Logistique ---
    with tctx("Logistic Regression"):
        # GRILLE ÉLARGIE pour auditer le faible besoin de régularisation C
        grid = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0]}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # AJOUT : class_weight='balanced' pour une meilleure robustesse au Population Shift
        lr_base = LogisticRegression(
            penalty="l2", 
            solver="lbfgs", 
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'
        )
        gs = GridSearchCV(lr_base, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        
        gs.fit(Xtr, y_tr_model)
        best_lr = gs.best_estimator_
        print(f"  -> Best Params: {gs.best_params_} | Best CV AUC (Honest): {gs.best_score_:.4f}")

    # --- F. Scaling & Calibration ---
    with tctx("Scaling & Calibration"):
        raw_tr = best_lr.decision_function(Xtr)
        raw_va = best_lr.decision_function(Xva)
        score_tr = scale_score(raw_tr)
        score_va = scale_score(raw_va)
        print(f"  -> Score Train (Part 2): Mean={score_tr.mean():.0f}")

        if calibration_method != "none":
            # CALIBRATION HONNÊTE : fit sur le split WOE (Xcalib, y_calib), jamais vu par best_lr lors du fit.
            calibrator = CalibratedClassifierCV(best_lr, method=calibration_method, cv="prefit")
            calibrator.fit(Xcalib, y_calib) # Fit sur la partie 1 du Train
            model_pd = calibrator
        else:
            model_pd = best_lr
            
        pd_tr = model_pd.predict_proba(Xtr)[:, 1]
        pd_va = model_pd.predict_proba(Xva)[:, 1]

    # --- G. Metrics ---
    with tctx("Metrics Calculation"):
        metrics = {
            "train_auc": float(roc_auc_score(y_tr_model, pd_tr)),
            "val_auc": float(roc_auc_score(y_va, pd_va)),
            "train_logloss": float(log_loss(y_tr_model, pd_tr)),
            "val_logloss": float(log_loss(y_va, pd_va)),
            # AJOUT DU BRIER SCORE pour auditer la qualité de la probabilité prédite
            "train_brier_score": float(brier_score_loss(y_tr_model, pd_tr)),
            "val_brier_score": float(brier_score_loss(y_va, pd_va)),
            "n_features": len(kept_features)
        }

    return {
        "model_pd": model_pd,
        "best_lr": best_lr,
        "woe_maps": woe_maps,
        "kept_features": kept_features,
        "metrics": metrics,
        "score_tr": score_tr,
        "score_va": score_va,
        "y_tr": y_tr_model,
        "y_va": y_va
    }

# ==============================================================================
# 7. BUCKETING & MASTER SCALE
# ==============================================================================
def create_risk_buckets(
    scores: np.ndarray, 
    y: np.ndarray, 
    n_buckets: int = 10, 
    fixed_edges: Optional[List[float]] = None
) -> Tuple[List[float], pd.DataFrame, bool]:
    df = pd.DataFrame({"score": scores, "y": y})
    
    if fixed_edges is not None:
        edges = np.array(fixed_edges)
        edges[0] = -np.inf
        edges[-1] = np.inf
    else:
        qs = np.linspace(0, 1, n_buckets + 1)
        edges = np.unique(np.quantile(scores, qs))
        edges[0] = -np.inf
        edges[-1] = np.inf

    df["bucket"] = np.digitize(df["score"], edges[1:], right=True) + 1
    
    stats = df.groupby("bucket").agg(
        count=("y", "size"),
        bad=("y", "sum"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        mean_score=("score", "mean")
    ).reset_index()
    
    stats["good"] = stats["count"] - stats["bad"]
    stats["pd"] = stats["bad"] / stats["count"]
    
    pds = stats["pd"].values
    # Check monotonicity strictly
    is_monotonic_decreasing = np.all(np.diff(pds) <= 0) 
    stats["monotonic_check"] = is_monotonic_decreasing
    
    return edges.tolist(), stats, is_monotonic_decreasing

# ==============================================================================
# MAIN
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--artifacts", default="artifacts/model_from_binned")
    p.add_argument("--bin-suffix", default="__BIN")
    p.add_argument("--corr-threshold", type=float, default=0.85)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--calibration", default="isotonic", help="none, sigmoid, isotonic")
    p.add_argument("--n-buckets", type=int, default=10)
    p.add_argument("--risk-buckets-in", default=None, help="JSON avec edges existants")
    p.add_argument("--timing", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    
    timer = Timer(live=True)
    
    df_tr = load_any(args.train)
    df_va = load_any(args.validation)
    
    out = train_pipeline(
        df_tr, df_va, args.target, args.bin_suffix,
        args.corr_threshold, args.cv_folds, args.calibration,
        timer
    )
    
    print("\n--- Construction de la Grille de Notation (Master Scale) ---")
    
    fixed_edges = None
    if args.risk_buckets_in and Path(args.risk_buckets_in).exists():
        print(f"-> Chargement de la grille existante : {args.risk_buckets_in}")
        with open(args.risk_buckets_in) as f:
            fixed_edges = json.load(f)["edges"]

    edges, stats_tr, mono_tr = create_risk_buckets(out["score_tr"], out["y_tr"], args.n_buckets, fixed_edges)
    _, stats_va, mono_va = create_risk_buckets(out["score_va"], out["y_va"], args.n_buckets, edges)

    print("\n[Statistiques TRAIN (Model Fit Split)]")
    print(stats_tr[["bucket", "min_score", "max_score", "count", "pd"]].to_string(index=False))
    
    print("\n[Statistiques VALIDATION]")
    print(stats_va[["bucket", "min_score", "max_score", "count", "pd"]].to_string(index=False))
    
    # Affichage des métriques complètes
    print("\n[Métriques Complètes]")
    for k, v in out['metrics'].items():
         print(f"  {k:20s}: {v:.4f}")

    if not mono_tr: print("\n⚠️  ATTENTION: Grille non monotone sur TRAIN !")
    else: print("\n✅ Grille robuste et monotone.")

    dump({
        "model_pd": out["model_pd"],
        "best_lr": out["best_lr"],
        "woe_maps": out["woe_maps"],
        "kept_features": out["kept_features"],
        "calibration": args.calibration
    }, artifacts / "model_best.joblib")
    
    save_json({"edges": edges}, artifacts / "risk_buckets.json")
    
    save_json({
        "train": stats_tr.to_dict(orient="records"),
        "validation": stats_va.to_dict(orient="records"),
        "metrics": out["metrics"]
    }, artifacts / "bucket_stats.json")
    
    print(f"\n✔ Modèle sauvegardé : {artifacts / 'model_best.joblib'}")

if __name__ == "__main__":
    main()