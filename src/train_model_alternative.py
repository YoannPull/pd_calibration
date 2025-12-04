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
- Sauvegarde train_scored / validation_scored avec vintage & loan_sequence_number
  dans data/processed/scored (et plus dans artifacts).

Option PD TTC :
- --ttc-mode train      : PD TTC par grade = PD observée sur TRAIN uniquement
- --ttc-mode train_val  : PD TTC par grade = PD observée sur TRAIN + VALIDATION
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import contextlib
import time
from typing import Dict, Any, List

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
    def __init__(self, live: bool = False):
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

def build_woe_maps(
    df: pd.DataFrame,
    target: str,
    raw_to_bin: Dict[str, str],
    smooth: float = 0.5
) -> Dict[str, Any]:
    """WOE map learned on full TRAIN (robuste)."""
    y = df[target].astype(int)
    tot_bad = y.sum()
    tot_good = len(y) - tot_bad

    global_woe = float(np.log((tot_bad + smooth) / (tot_good + smooth)))

    maps: Dict[str, Any] = {}

    for raw, bin_col in raw_to_bin.items():
        tab = df.groupby(bin_col)[target].agg(["sum", "count"])
        tab["good"] = tab["count"] - tab["sum"]

        bad_i = tab["sum"].values.astype(float)
        good_i = tab["good"].values.astype(float)

        # WOE(i) = log( (bad_i+prior)/(good_i+prior) )
        woe_i = np.log((bad_i + smooth) / (good_i + smooth))

        maps[raw] = {
            "map": {int(k): float(v) for k, v in zip(tab.index, woe_i)},
            "default": global_woe,
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

    kept: List[str] = []
    corr = X.corr().abs()

    for c in order:
        if not kept:
            kept.append(c)
            continue
        if corr.loc[c, kept].max() < corr_thr:
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Master Scale — version optimisée
# ---------------------------------------------------------------------------

def create_risk_buckets(
    scores,
    y,
    n_buckets: int = 10,
    fixed_edges=None,
    n_init_bins: int = 100,
    min_count: int = 5000,
    min_bad: int = 50,
):
    """
    Construction des grades à partir du score TTC.

    - Si fixed_edges est fourni : on applique simplement ces edges (cas validation
      ou rerun avec grille figée).
    - Sinon : on construit une grille optimisée sur le TRAIN uniquement :
        * bins fins par quantiles (n_init_bins),
        * fusion adjacente pour imposer la monotonicité PD(score),
        * fusion pour respecter des minima de volumétrie (min_count, min_bad),
        * fusion pour arriver à exactement n_buckets grades.
    """

    scores = np.asarray(scores)
    y = np.asarray(y).astype(int)

    # ------------------------------------------------------------------
    # Cas 1 : edges fournis → on ne fait qu'appliquer / calculer les stats
    # ------------------------------------------------------------------
    if fixed_edges is not None:
        edges = np.array(fixed_edges, dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf

        raw_bucket = np.digitize(scores, edges[1:], right=True) + 1
        # Convention finale : 1 = meilleur grade (score le plus élevé)
        bucket = n_buckets + 1 - raw_bucket

        df = pd.DataFrame({"bucket": bucket, "y": y, "score": scores})
        stats = df.groupby("bucket").agg(
            count=("y", "size"),
            bad=("y", "sum"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        ).reset_index().sort_values("bucket")
        stats["pd"] = stats["bad"] / stats["count"]

        mono = bool(np.all(np.diff(stats["pd"].values) >= -1e-12))
        return edges.tolist(), stats, mono

    # ------------------------------------------------------------------
    # Cas 2 : optimisation des buckets sur le TRAIN
    # ------------------------------------------------------------------
    df = pd.DataFrame({"score": scores, "y": y})

    # Binning initial fin par quantiles de score
    n_init_bins = min(n_init_bins, max(2, df["score"].nunique()))
    df["init_bin"] = pd.qcut(
        df["score"], q=n_init_bins, labels=False, duplicates="drop"
    )

    grp = (
        df.groupby("init_bin")
        .agg(
            count=("y", "size"),
            bad=("y", "sum"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reset_index()
        .sort_values("min_score")  # scores croissants
    )

    # On travaille sur une liste de "blocs" adjacents sur l'axe du score
    blocks: List[Dict[str, float]] = []
    for _, row in grp.iterrows():
        blocks.append(
            {
                "count": int(row["count"]),
                "bad": int(row["bad"]),
                "min_score": float(row["min_score"]),
                "max_score": float(row["max_score"]),
            }
        )

    def pd_block(b):
        return b["bad"] / b["count"] if b["count"] > 0 else 0.0

    # 1) Monotonicité PD(score) décroissante lorsque le score augmente
    #    (score élevé = moins risqué, donc PD plus faible).
    i = 0
    while i < len(blocks) - 1:
        if pd_block(blocks[i + 1]) > pd_block(blocks[i]):
            # Violation : la PD du bloc "plus bon score" est > PD du bloc précédent
            b1, b2 = blocks[i], blocks[i + 1]
            merged = {
                "count": b1["count"] + b2["count"],
                "bad": b1["bad"] + b2["bad"],
                "min_score": min(b1["min_score"], b2["min_score"]),
                "max_score": max(b1["max_score"], b2["max_score"]),
            }
            blocks[i] = merged
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1

    # 2) Contraintes de volumétrie (min_count, min_bad) tant qu'on a
    #    plus de blocs que n_buckets (sinon on n'a plus de marge de fusion).
    changed = True
    while changed and len(blocks) > n_buckets:
        changed = False
        for i, b in enumerate(blocks):
            if (b["count"] < min_count) or (b["bad"] < min_bad):
                if len(blocks) == 1:
                    break
                # Choix du voisin le plus proche en PD pour limiter les ruptures
                if i == 0:
                    j = 1
                elif i == len(blocks) - 1:
                    j = i - 1
                else:
                    pd_i = pd_block(b)
                    pd_left = pd_block(blocks[i - 1])
                    pd_right = pd_block(blocks[i + 1])
                    j = i - 1 if abs(pd_i - pd_left) <= abs(pd_i - pd_right) else i + 1

                b2 = blocks[j]
                merged = {
                    "count": b["count"] + b2["count"],
                    "bad": b["bad"] + b2["bad"],
                    "min_score": min(b["min_score"], b2["min_score"]),
                    "max_score": max(b["max_score"], b2["max_score"]),
                }
                lo, hi = sorted([i, j])
                blocks[lo] = merged
                del blocks[hi]
                changed = True
                break

    # 3) Réduction à exactement n_buckets en fusionnant les blocs
    #    adjacents ayant les PD les plus proches.
    while len(blocks) > n_buckets:
        pds = [pd_block(b) for b in blocks]
        diffs = [abs(pds[i + 1] - pds[i]) for i in range(len(blocks) - 1)]
        k = int(np.argmin(diffs))
        b1, b2 = blocks[k], blocks[k + 1]
        merged = {
            "count": b1["count"] + b2["count"],
            "bad": b1["bad"] + b2["bad"],
            "min_score": min(b1["min_score"], b2["min_score"]),
            "max_score": max(b1["max_score"], b2["max_score"]),
        }
        blocks[k] = merged
        del blocks[k + 1]

    # ------------------------------------------------------------------
    # Construction des edges à partir des blocs agrégés
    # ------------------------------------------------------------------
    blocks_sorted = sorted(blocks, key=lambda b: b["min_score"])  # scores croissants

    edges = [-np.inf]
    for i in range(len(blocks_sorted) - 1):
        hi = blocks_sorted[i]["max_score"]
        lo_next = blocks_sorted[i + 1]["min_score"]
        thr = (hi + lo_next) / 2.0  # seuil entre deux blocs
        edges.append(thr)
    edges.append(np.inf)
    edges = np.array(edges)

    # Attribution des buckets comme dans ton code original
    raw_bucket = np.digitize(scores, edges[1:], right=True) + 1  # 1 = scores les plus bas
    bucket = n_buckets + 1 - raw_bucket  # 1 = meilleur grade

    df2 = pd.DataFrame({"bucket": bucket, "y": y, "score": scores})
    stats = (
        df2.groupby("bucket")
        .agg(
            count=("y", "size"),
            bad=("y", "sum"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reset_index()
        .sort_values("bucket")
    )
    stats["pd"] = stats["bad"] / stats["count"]

    mono = bool(np.all(np.diff(stats["pd"].values) >= -1e-12))

    return edges.tolist(), stats, mono


# ---------------------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------------------------

META_COLS = ["vintage", "loan_sequence_number"]  # conservés, jamais utilisés comme features


def train_pipeline(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    target: str,
    bin_suffix: str,
    corr_threshold: float,
    cv_folds: int,
    calibration_method: str,
    timer: Timer,
):

    # -------------------------------------------------
    # Meta (pour analyse future, pas features)
    # -------------------------------------------------
    meta_cols_present = [c for c in META_COLS if c in df_tr.columns and c in df_va.columns]
    meta_tr = df_tr[meta_cols_present].copy()
    meta_va = df_va[meta_cols_present].copy()

    # -------------------------------------------------
    # Identify binned features
    # -------------------------------------------------
    with timer.section("Identify BIN features"):
        bin_cols = [c for c in df_tr.columns if bin_suffix in c]

        BLACKLIST = [
            "quarter", "year", "month", "vintage", "date", "time",
            "maturity", "first_payment", "mi_cancellation",
            "interest_rate", "loan_sequence_number",
        ]

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
        ytr_full = df_tr[target].astype(int)

        Xva_full = apply_woe(df_va, woe_maps, bin_suffix)
        Xva_full = add_interactions(Xva_full)
        yva_full = df_va[target].astype(int)

    # -------------------------------------------------
    # Calibration split
    # -------------------------------------------------
    with timer.section("Split for Calibration"):
        X_model, X_cal, y_model, y_cal = train_test_split(
            Xtr_full, ytr_full, test_size=0.2, stratify=ytr_full, random_state=42
        )

    # -------------------------------------------------
    # Feature selection (sur X_model)
    # -------------------------------------------------
    with timer.section("Feature Selection"):
        kept_features = select_features(X_model, corr_thr=corr_threshold)

        X_model_kept = X_model[kept_features].values
        X_cal_kept = X_cal[kept_features].values
        Xtr_full_kept = Xtr_full[kept_features].values
        Xva_full_kept = Xva_full[kept_features].values

    # -------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------
    with timer.section("Fit Logistic Regression"):
        grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        gs = GridSearchCV(lr, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_model_kept, y_model.values)

        best_lr = gs.best_estimator_

    # -------------------------------------------------
    # Score TTC (log-odds sur TOUT le train + TOUTE la validation)
    # -------------------------------------------------
    with timer.section("Raw score scaling"):
        log_odds_tr_full = best_lr.decision_function(Xtr_full_kept)
        log_odds_va_full = best_lr.decision_function(Xva_full_kept)

        score_tr_full = scale_score(log_odds_tr_full)
        score_va_full = scale_score(log_odds_va_full)

    # -------------------------------------------------
    # Calibration Isotonic Regression
    # -------------------------------------------------
    with timer.section("Calibration"):
        if calibration_method != "none":
            calibrator = CalibratedClassifierCV(best_lr, method=calibration_method, cv="prefit")
            calibrator.fit(X_cal_kept, y_cal.values)
            model_pd = calibrator
        else:
            model_pd = best_lr

        # PD pour les jeux utilisés pour les métriques
        pd_tr_model = model_pd.predict_proba(X_model_kept)[:, 1]
        pd_va_full = model_pd.predict_proba(Xva_full_kept)[:, 1]

        # PD sur TOUT le train (pour train_scored)
        pd_tr_full = model_pd.predict_proba(Xtr_full_kept)[:, 1]

    # -------------------------------------------------
    # Metrics (sur subset modèle pour le train, full pour la val)
    # -------------------------------------------------
    metrics = {
        "train_auc": float(roc_auc_score(y_model, pd_tr_model)),
        "val_auc": float(roc_auc_score(yva_full, pd_va_full)),
        "train_logloss": float(log_loss(y_model, pd_tr_model)),
        "val_logloss": float(log_loss(yva_full, pd_va_full)),
        "train_brier": float(brier_score_loss(y_model, pd_tr_model)),
        "val_brier": float(brier_score_loss(yva_full, pd_va_full)),
        "n_features": len(kept_features),
    }

    return {
        "model_pd": model_pd,
        "best_lr": best_lr,
        "woe_maps": woe_maps,
        "kept_features": kept_features,
        "metrics": metrics,
        "score_tr": score_tr_full,      # score TTC full train
        "score_va": score_va_full,      # score TTC full val
        "pd_tr": pd_tr_full,            # PD full train
        "pd_va": pd_va_full,            # PD full val
        "y_tr": ytr_full.values,
        "y_va": yva_full.values,
        "meta_tr": meta_tr,
        "meta_va": meta_va,
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
    p.add_argument(
        "--scored-outdir",
        default="data/processed/scored",
        help="Répertoire où écrire train_scored / validation_scored (parquet).",
    )
    p.add_argument("--timing", action="store_true")
    p.add_argument(
        "--ttc-mode",
        choices=["train", "train_val"],
        default="train",
        help="Source pour la PD TTC par grade dans bucket_stats.json : "
             "'train' = train uniquement (par défaut), 'train_val' = train + validation (long run).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    scored_dir = Path(args.scored_outdir)
    scored_dir.mkdir(parents=True, exist_ok=True)

    timer = Timer(live=args.timing)

    df_tr = load_any(args.train)
    df_va = load_any(args.validation)

    out = train_pipeline(
        df_tr, df_va, args.target,
        args.bin_suffix, args.corr_threshold,
        args.cv_folds, args.calibration, timer
    )

    # MASTER SCALE (buckets sur scores TTC full train / full val)
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

    # ------------------------------------------------------------------
    # Construction d'une PD TTC "long run" train + validation (optionnel)
    # ------------------------------------------------------------------
    # stats_tr / stats_va ont : bucket, count, bad, min_score, max_score, pd
    stats_tr_base = stats_tr.copy()
    stats_va_base = stats_va.copy()

    stats_longrun = stats_tr_base[["bucket", "count", "bad", "min_score", "max_score"]].merge(
        stats_va_base[["bucket", "count", "bad", "min_score", "max_score"]],
        on="bucket",
        how="outer",
        suffixes=("_tr", "_va")
    )

    # Remplir les NaN pour les compteurs
    for col in ["count_tr", "count_va", "bad_tr", "bad_va"]:
        stats_longrun[col] = stats_longrun[col].fillna(0)

    stats_longrun["count"] = stats_longrun["count_tr"] + stats_longrun["count_va"]
    stats_longrun["bad"] = stats_longrun["bad_tr"] + stats_longrun["bad_va"]
    stats_longrun["min_score"] = stats_longrun[["min_score_tr", "min_score_va"]].min(axis=1)
    stats_longrun["max_score"] = stats_longrun[["max_score_tr", "max_score_va"]].max(axis=1)
    stats_longrun["pd"] = stats_longrun["bad"] / stats_longrun["count"]
    stats_longrun = stats_longrun[["bucket", "count", "bad", "min_score", "max_score", "pd"]].sort_values("bucket")

    # ------------------------------------------------------------------
    # Choix de la PD TTC utilisée dans la clé 'train' de bucket_stats.json
    # ------------------------------------------------------------------
    if args.ttc_mode == "train":
        stats_train_for_json = stats_tr_base
    else:  # "train_val"
        stats_train_for_json = stats_longrun

    # Sauvegarde du modèle + artefacts grille (toujours dans artifacts)
    dump({
        "model_pd": out["model_pd"],
        "best_lr": out["best_lr"],
        "woe_maps": out["woe_maps"],
        "kept_features": out["kept_features"],
        "calibration": args.calibration,
    }, artifacts / "model_best.joblib")

    save_json({"edges": edges}, artifacts / "risk_buckets.json")

    # bucket_stats.json contient :
    # - 'train'           : PD TTC utilisée (train ou long-run selon ttc_mode)
    # - 'train_raw'       : stats train seules
    # - 'train_val_longrun' : stats train+validation
    # - 'validation'      : stats validation
    # - 'metrics'         : métriques de perf du modèle
    # - 'ttc_mode'        : mode utilisé
    bucket_stats_payload = {
        "train": stats_train_for_json.to_dict(orient="records"),
        "train_raw": stats_tr_base.to_dict(orient="records"),
        "train_val_longrun": stats_longrun.to_dict(orient="records"),
        "validation": stats_va_base.to_dict(orient="records"),
        "metrics": out["metrics"],
        "ttc_mode": args.ttc_mode,
    }
    save_json(bucket_stats_payload, artifacts / "bucket_stats.json")

    # ------------------------------------------------------------------
    # train_scored / validation_scored -> data/processed/scored
    # ------------------------------------------------------------------
    # Grades : 1 = moins risqué, n = plus risqué
    raw_grade_tr = np.digitize(out["score_tr"], edges[1:], right=True) + 1
    raw_grade_va = np.digitize(out["score_va"], edges[1:], right=True) + 1

    grade_tr = args.n_buckets + 1 - raw_grade_tr
    grade_va = args.n_buckets + 1 - raw_grade_va

    # Train scored
    train_scored = out["meta_tr"].copy()
    train_scored[args.target] = out["y_tr"]
    train_scored["score_ttc"] = out["score_tr"]
    train_scored["pd"] = out["pd_tr"]
    train_scored["grade"] = grade_tr

    train_scored_path = scored_dir / "train_scored.parquet"
    train_scored.to_parquet(train_scored_path, index=False)
    print(f"✔ train_scored sauvegardé : {train_scored_path}")

    # Validation scored
    validation_scored = out["meta_va"].copy()
    validation_scored[args.target] = out["y_va"]
    validation_scored["score_ttc"] = out["score_va"]
    validation_scored["pd"] = out["pd_va"]
    validation_scored["grade"] = grade_va

    validation_scored_path = scored_dir / "validation_scored.parquet"
    validation_scored.to_parquet(validation_scored_path, index=False)
    print(f"✔ validation_scored sauvegardé : {validation_scored_path}")

    print(f"\nMode PD TTC utilisé pour 'train' dans bucket_stats.json : {args.ttc_mode}")


if __name__ == "__main__":
    main()
