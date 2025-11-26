#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRAINING PIPELINE — CLEAN & ROBUST VERSION (BANK-GRADE)
=======================================================

Principes :
- WOE appris sur 100 % du TRAIN (comme dans les vrais modèles IRB).
- Calibration sur un split interne (20 %) non vu par la LR.
- Logistic Regression L2 propre (sans class_weight='balanced').
- Score TTC basé sur log-odds non calibrés (standard bancaire).
- PD calibrée via isotonic regression.
- Master Scale (10 buckets) monotone.
- Code propre, lisible, industrialisable.

Auteur : ChatGPT — Version A (clean seulement train + apply)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import contextlib
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from joblib import dump


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)

def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


class Timer:
    def __init__(self, live=False):
        self.records = {}
        self.live = live

    @contextlib.contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        if self.live:
            print(f"▶ {name} ...")
        yield
        dt = time.perf_counter() - t0
        self.records[name] = dt
        if self.live:
            print(f"  ✓ {name:25s} {dt:.3f}s")


# ---------------------------------------------------------------------------
# WOE utilities
# ---------------------------------------------------------------------------

def raw_name_from_bin(col: str, tag: str) -> str:
    return col.replace(tag, "") if tag in col else col

def build_woe_maps(df: pd.DataFrame, target: str, raw_to_bin: Dict[str, str], smooth=0.5) -> Dict[str, Any]:
    """WOE map learned on full TRAIN (robuste)."""
    y = df[target].astype(int)
    tot_bad = y.sum()
    tot_good = len(y) - tot_bad

    global_woe = float(np.log((tot_bad + smooth) / (tot_good + smooth)))

    maps = {}

    for raw, bin_col in raw_to_bin.items():
        tab = df.groupby(bin_col)[target].agg(["sum", "count"])
        tab["good"] = tab["count"] - tab["sum"]

        # WOE(i) = log( (bad_i+prior)/(good_i+prior) ) - log(global odds)
        bad_i = tab["sum"].values.astype(float)
        good_i = tab["good"].values.astype(float)

        woe_i = np.log((bad_i + smooth) / (good_i + smooth))

        maps[raw] = {
            "map": {int(k): float(v) for k, v in zip(tab.index, woe_i)},
            "default": global_woe
        }
    return maps

def apply_woe(df: pd.DataFrame, woe_maps: Dict[str, Any], bin_suffix: str) -> pd.DataFrame:
    cols = []
    for raw, info in woe_maps.items():
        bin_col1 = f"{bin_suffix}{raw}"
        bin_col2 = f"{raw}{bin_suffix}"
        colbin = bin_col1 if bin_col1 in df.columns else (bin_col2 if bin_col2 in df.columns else None)

        mapping = info["map"]
        default = info["default"]

        if colbin is None:
            cols.append(pd.Series(default, index=df.index, name=f"{raw}_WOE"))
        else:
            cols.append(df[colbin].map(mapping).fillna(default).rename(f"{raw}_WOE"))

    return pd.concat(cols, axis=1)


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
# Score scaling (TTC)
# ---------------------------------------------------------------------------

def scale_score(log_odds, base_points=600, base_odds=50, pdo=20):
    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)
    return np.round(offset - factor * log_odds).astype(int)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(X: pd.DataFrame, corr_thr=0.85):
    X = X.fillna(0)
    order = X.var().sort_values(ascending=False).index.tolist()

    kept = []
    corr = X.corr().abs()

    for c in order:
        if not kept:
            kept.append(c)
            continue
        if corr.loc[c, kept].max() < corr_thr:
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Master Scale
# ---------------------------------------------------------------------------

def create_risk_buckets(scores, y, n_buckets=10, fixed_edges=None):
    df = pd.DataFrame({"score": scores, "y": y})

    if fixed_edges is None:
        qs = np.linspace(0, 1, n_buckets + 1)
        edges = np.quantile(scores, qs)
        edges[0] = -np.inf
        edges[-1] = np.inf
    else:
        edges = np.array(fixed_edges)
        edges[0] = -np.inf
        edges[-1] = np.inf

    df["bucket"] = np.digitize(df["score"], edges[1:], right=True) + 1

    stats = df.groupby("bucket").agg(
        count=("y", "size"),
        bad=("y", "sum"),
        min_score=("score", "min"),
        max_score=("score", "max")
    ).reset_index()

    stats["pd"] = stats["bad"] / stats["count"]

    mono = bool(np.all(np.diff(stats["pd"].values) <= 0))

    return edges.tolist(), stats, mono


# ---------------------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------------------------

def train_pipeline(df_tr, df_va, target, bin_suffix, corr_threshold, cv_folds, calibration_method, timer):

    # -------------------------------------------------
    # Identify binned features
    # -------------------------------------------------
    with timer.section("Identify BIN features"):
        bin_cols = [c for c in df_tr.columns if bin_suffix in c]

        BLACKLIST = ["quarter", "year", "month", "vintage", "date", "time",
                     "maturity", "first_payment", "mi_cancellation",
                     "interest_rate", "loan_sequence_number"]

        def safe(col):
            raw = raw_name_from_bin(col, bin_suffix).lower()
            return not any(bad in raw for bad in BLACKLIST)

        bin_cols = [c for c in bin_cols if safe(c)]
        raw_to_bin = {raw_name_from_bin(c, bin_suffix): c for c in bin_cols}

    # -------------------------------------------------
    # WOE learn on full TRAIN
    # -------------------------------------------------
    with timer.section("Learn WOE"):
        woe_maps = build_woe_maps(df_tr, target, raw_to_bin)

        Xtr_full = apply_woe(df_tr, woe_maps, bin_suffix)
        Xtr_full = add_interactions(Xtr_full)
        ytr_full = df_tr[target].astype(int).values

        Xva_full = apply_woe(df_va, woe_maps, bin_suffix)
        Xva_full = add_interactions(Xva_full)
        yva_full = df_va[target].astype(int).values

    # -------------------------------------------------
    # Calibration split
    # -------------------------------------------------
    with timer.section("Split for Calibration"):
        X_model, X_cal, y_model, y_cal = train_test_split(
            Xtr_full, ytr_full, test_size=0.2, stratify=ytr_full, random_state=42
        )

    # -------------------------------------------------
    # Feature selection on X_model
    # -------------------------------------------------
    with timer.section("Feature Selection"):
        kept_features = select_features(pd.DataFrame(X_model), corr_thr=corr_threshold)
        X_model = pd.DataFrame(X_model)[kept_features].values
        X_cal = pd.DataFrame(X_cal)[kept_features].values
        X_va = pd.DataFrame(Xva_full)[kept_features].values

    # -------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------
    with timer.section("Fit Logistic Regression"):
        grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        gs = GridSearchCV(lr, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_model, y_model)

        best_lr = gs.best_estimator_

    # -------------------------------------------------
    # Score (raw log-odds)
    # -------------------------------------------------
    with timer.section("Raw score scaling"):
        log_odds_tr = best_lr.decision_function(X_model)
        log_odds_va = best_lr.decision_function(X_va)

        score_tr = scale_score(log_odds_tr)
        score_va = scale_score(log_odds_va)

    # -------------------------------------------------
    # Calibration Isotonic Regression
    # -------------------------------------------------
    with timer.section("Calibration"):
        if calibration_method != "none":
            calibrator = CalibratedClassifierCV(best_lr, method=calibration_method, cv="prefit")
            calibrator.fit(X_cal, y_cal)
            model_pd = calibrator
        else:
            model_pd = best_lr

        pd_tr = model_pd.predict_proba(X_model)[:, 1]
        pd_va = model_pd.predict_proba(X_va)[:, 1]

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    metrics = {
        "train_auc": float(roc_auc_score(y_model, pd_tr)),
        "val_auc": float(roc_auc_score(yva_full, pd_va)),
        "train_logloss": float(log_loss(y_model, pd_tr)),
        "val_logloss": float(log_loss(yva_full, pd_va)),
        "train_brier": float(brier_score_loss(y_model, pd_tr)),
        "val_brier": float(brier_score_loss(yva_full, pd_va)),
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
        "y_tr": y_model,
        "y_va": yva_full,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--artifacts", default="artifacts/model_from_binned")
    p.add_argument("--bin-suffix", default="__BIN")
    p.add_argument("--corr-threshold", type=float, default=0.85)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--calibration", default="isotonic")
    p.add_argument("--n-buckets", type=int, default=10)
    p.add_argument("--risk-buckets-in", default=None)
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
        df_tr, df_va, args.target,
        args.bin_suffix, args.corr_threshold,
        args.cv_folds, args.calibration, timer
    )

    # MASTER SCALE
    fixed_edges = None
    if args.risk_buckets_in and Path(args.risk_buckets_in).exists():
        fixed_edges = json.loads(Path(args.risk_buckets_in).read_text())["edges"]

    edges, stats_tr, mono_tr = create_risk_buckets(
        out["score_tr"], out["y_tr"], args.n_buckets, fixed_edges
    )

    _, stats_va, mono_va = create_risk_buckets(
        out["score_va"], out["y_va"], args.n_buckets, edges
    )

    print("\n[MÉTRIQUES]")
    for k, v in out["metrics"].items():
        print(f"  {k:20s} : {v:.4f}")

    print("\n[MONOTONICITÉ TRAIN] :", "OK" if mono_tr else "NON MONOTONE")
    print("[MONOTONICITÉ VAL]   :", "OK" if mono_va else "NON MONOTONE")

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


if __name__ == "__main__":
    main()
