#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train + select best model with least score drift (PSI / JS) under an AUC constraint.
- Loads TRAIN/VALIDATION (binned) datasets
- Builds candidates (LogReg, RandomForest, HistGBDT)
- Gets OOF predictions on train for drift baseline
- Evaluates metrics on validation (AUC, Brier, LogLoss)
- Computes score drift between OOF(train) and VAL predictions (PSI + JS)
- Selects model with minimal drift (optionally subject to min AUC)
- Optionally calibrates the best model (sigmoid|isotonic) via CV on TRAIN
- Saves best model + metrics + drift reports
"""

import argparse
import json
from pathlib import Path
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


# -----------------------------------------
# Utils: loading / saving
# -----------------------------------------
def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


# -----------------------------------------
# Drift metrics (PSI / JS) on score distributions
# -----------------------------------------
def make_quantile_bins(base_scores: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile cutpoints based on base distribution (e.g. OOF train)."""
    qs = np.linspace(0, 1, n_bins + 1) * 100
    edges = np.unique(np.percentile(base_scores, qs))
    # ensure at least 2 edges
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    # stretch bounds to cover [0,1] safely
    edges[0] = min(edges[0], 0.0)
    edges[-1] = max(edges[-1], 1.0)
    return edges


def hist_proportions(x: np.ndarray, edges: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    counts, _ = np.histogram(x, bins=edges)
    p = counts.astype(float)
    p = (p + eps) / (p.sum() + eps * len(p))
    return p


def psi(p: np.ndarray, q: np.ndarray) -> float:
    """Population Stability Index between distributions p and q."""
    return float(np.sum((p - q) * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between distributions p and q."""
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def score_drift(oof_scores: np.ndarray, val_scores: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    edges = make_quantile_bins(oof_scores, n_bins=n_bins)
    p = hist_proportions(oof_scores, edges)
    q = hist_proportions(val_scores, edges)
    return {
        "psi": psi(p, q),
        "js": js_divergence(p, q),
        "bins": len(p),
    }


# -----------------------------------------
# Candidates
# -----------------------------------------
def build_candidates(
    standardize: bool = True,
) -> Dict[str, Pipeline]:
    """
    Retourne un dict de modèles candidats (pipelines sklearn).
    - LogReg L2 + StandardScaler (optionnel)
    - RandomForest
    - HistGradientBoosting
    """
    models = {}

    # Logistic Regression (class_weight balanced)
    steps_lr = []
    if standardize:
        steps_lr.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    steps_lr.append(("clf",
        LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            n_jobs=None
        )
    ))
    models["logreg_l2"] = Pipeline(steps_lr)

    # Random Forest
    models["rf"] = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42
    )

    # HistGradientBoosting (robuste et rapide)
    models["hgbdt"] = HistGradientBoostingClassifier(
        max_depth=None,
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.0,
        random_state=42
    )

    return models


# -----------------------------------------
# OOF predictions
# -----------------------------------------
def get_oof_predictions(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Renvoie les prédictions OOF (proba classe 1) pour X."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(shape=(len(X),), fill_value=np.nan, dtype=float)
    for train_idx, valid_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr = y[train_idx]
        mdl = model
        # re-clone léger (sklearn clone pas obligatoire si on reconstruit build_candidates à chaque boucle externe)
        mdl.fit(X_tr, y_tr)
        proba = mdl.predict_proba(X_va)[:, 1]
        oof[valid_idx] = proba
    # sécurité
    if np.isnan(oof).any():
        raise RuntimeError("NaN in OOF predictions.")
    return oof


# -----------------------------------------
# Metrics
# -----------------------------------------
def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "logloss": float(log_loss(y_true, np.clip(proba, 1e-15, 1 - 1e-15))),
        "mean": float(np.mean(proba)),
        "std": float(np.std(proba)),
    }


# -----------------------------------------
# Feature selection utility (default: WOE features if present)
# -----------------------------------------
def default_feature_list(df: pd.DataFrame, target: str) -> List[str]:
    cols = [c for c in df.columns if c != target]
    woe = [c for c in cols if c.startswith("woe__")]
    if woe:
        return woe
    # sinon, toutes numériques
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


# -----------------------------------------
# Training routine
# -----------------------------------------
def train_and_select(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    min_auc: float = 0.60,
    drift_bins: int = 10,
    calibrate: str = "none",  # "none" | "sigmoid" | "isotonic"
) -> Tuple[str, object, Dict[str, dict]]:
    """
    Retourne: (best_name, best_model_fitted_on_train, reports_per_model)
    """
    reports = {}
    candidates = build_candidates(standardize=True)

    best_name = None
    best_model = None
    best_key = None  # (drift_psi, -auc, logloss)

    for name, model in candidates.items():
        # OOF for drift baseline
        oof = get_oof_predictions(model, X_train, y_train, n_splits=5, random_state=42)

        # Fit on full train, evaluate on validation
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]

        metr_val = compute_metrics(y_val, val_proba)
        drift = score_drift(oof, val_proba, n_bins=drift_bins)

        rep = {
            "metrics_val": metr_val,
            "drift": drift,
            "oof_mean": float(oof.mean()),
            "oof_std": float(oof.std()),
            "model_name": name,
        }
        reports[name] = rep

        # skip if AUC below threshold
        if metr_val["auc"] < min_auc:
            continue

        # selection key: minimize drift PSI, then maximize AUC (=> minimize -AUC), then minimize logloss
        sel_key = (drift["psi"], -metr_val["auc"], metr_val["logloss"])
        if (best_key is None) or (sel_key < best_key):
            best_key = sel_key
            best_name = name
            best_model = model

    if best_model is None:
        # si aucun ne passe le seuil AUC: on prend le moins de drift, point
        for name, rep in reports.items():
            metr_val = rep["metrics_val"]
            drift = rep["drift"]
            sel_key = (drift["psi"], -metr_val["auc"], metr_val["logloss"])
            if (best_key is None) or (sel_key < best_key):
                best_key = sel_key
                best_name = name
                best_model = candidates[name]

        # refit best on full train
        best_model.fit(X_train, y_train)

    # Optional calibration on TRAIN via CV
    if calibrate in ("sigmoid", "isotonic"):
        best_model = CalibratedClassifierCV(
            estimator=best_model, method=calibrate, cv=5, n_jobs=None
        )
        best_model.fit(X_train, y_train)
        # update metrics with calibrated proba
        val_proba = best_model.predict_proba(X_val)[:, 1]
        reports[best_name]["metrics_val_calibrated"] = compute_metrics(y_val, val_proba)

    return best_name, best_model, reports


# -----------------------------------------
# CLI
# -----------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train models and select the one with least drift")
    p.add_argument("--train", required=True, help="Train (CSV/Parquet) — already imputed + binned")
    p.add_argument("--validation", required=True, help="Validation (CSV/Parquet) — already imputed + binned")
    p.add_argument("--target", required=True, help="Target column (0/1)")
    p.add_argument("--artifacts", default="artifacts", help="Artifacts dir to save model + reports")
    p.add_argument("--min-auc", type=float, default=0.60, help="Minimum AUC to be eligible")
    p.add_argument("--drift-bins", type=int, default=10, help="Number of quantile bins for PSI")
    p.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"], default="none")
    p.add_argument("--features", default=None, help="Comma-separated list of features to use (default: auto WOE or numeric)")
    return p.parse_args()


def main():
    args = parse_args()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    df_tr = load_any(args.train)
    df_va = load_any(args.validation)

    if args.target not in df_tr.columns or args.target not in df_va.columns:
        raise SystemExit(f"Target '{args.target}' not found in train/validation.")

    y_tr = df_tr[args.target].astype(int).values
    y_va = df_va[args.target].astype(int).values
    if args.features:
        feats = [c.strip() for c in args.features.split(",") if c.strip()]
    else:
        feats = default_feature_list(df_tr, args.target)

    X_tr = df_tr[feats].copy()
    X_va = df_va[feats].copy()

    best_name, best_model, reports = train_and_select(
        X_tr, y_tr, X_va, y_va,
        min_auc=args.min_auc,
        drift_bins=args.drift_bins,
        calibrate=args.calibration,
    )

    # Save artifacts
    model_path = artifacts / "model_best.joblib"
    dump({"model": best_model, "features": feats, "target": args.target}, model_path)

    # Flat metrics
    flat_reports = []
    for name, rep in reports.items():
        row = {
            "model": name,
            **{f"val_{k}": v for k, v in rep["metrics_val"].items()},
            "drift_psi": rep["drift"]["psi"],
            "drift_js": rep["drift"]["js"],
            "drift_bins": rep["drift"]["bins"],
            "oof_mean": rep["oof_mean"],
            "oof_std": rep["oof_std"],
        }
        if "metrics_val_calibrated" in rep:
            for k, v in rep["metrics_val_calibrated"].items():
                row[f"val_cal_{k}"] = v
        flat_reports.append(row)

    rep_df = pd.DataFrame(flat_reports).sort_values(["drift_psi", "val_auc"], ascending=[True, False])
    rep_df.to_csv(artifacts / "model_reports.csv", index=False)

    meta = {
        "selected_model": best_name,
        "model_path": str(model_path.resolve()),
        "features": feats,
        "target": args.target,
        "min_auc": args.min_auc,
        "calibration": args.calibration,
        "train_path": str(Path(args.train).resolve()),
        "validation_path": str(Path(args.validation).resolve()),
    }
    save_json(meta, artifacts / "model_meta.json")
    print(f"✔ Best model: {best_name}")
    print(f"✔ Saved: {model_path}")
    print(f"✔ Reports: {(artifacts / 'model_reports.csv')}")


if __name__ == "__main__":
    main()
