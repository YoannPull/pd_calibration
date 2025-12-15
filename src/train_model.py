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
  dans data/processed/scored.

NOUVEAU (grille + TTC windowées) :
- Si --risk-buckets-in n'est PAS fourni :
  - la grille (edges) est construite sur une fenêtre glissante (par défaut 10 ans)
    terminant au dernier trimestre (vintage) présent dans VALIDATION.
  - les PD TTC par grade (bucket_stats['train'] / 'train_val_longrun_window') sont aussi
    calculées sur la même fenêtre glissante.

Option PD TTC (dans bucket_stats.json) :
- --ttc-mode train      : PD TTC par grade = PD observée sur TRAIN (window)
- --ttc-mode train_val  : PD TTC par grade = PD observée sur TRAIN+VALIDATION (window)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import statsmodels.api as sm
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


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
# Time helpers (vintage window)
# ---------------------------------------------------------------------------



def parse_vintage_to_period(s: pd.Series, freq: str = "Q") -> pd.PeriodIndex:
    """
    Convertit une colonne vintage en PeriodIndex (Q ou M).
    Supporte :
      - '2015Q1' ou '2015-Q1'
      - dates parseables (ex '2015-03-31') -> conversion en trimestre/mois
      - Period déjà présent
    """
    freq = freq.upper()

    if isinstance(s.dtype, pd.PeriodDtype):
        return s.astype(f"period[{freq}]").array

    s_str = s.astype(str)

    # formats type 'YYYYQn' ou 'YYYY-Qn'
    m = s_str.str.match(r"^\d{4}[- ]?Q[1-4]$")
    if m.all() and freq.startswith("Q"):
        cleaned = s_str.str.replace(" ", "", regex=False).str.replace("-", "", regex=False)
        return pd.PeriodIndex(cleaned, freq="Q")

    # fallback: parse date
    dt = pd.to_datetime(s_str, errors="coerce")
    if dt.notna().mean() < 0.95:
        bad = s_str[dt.isna()].head(5).tolist()
        raise ValueError(f"Impossible de parser vintage. Exemples non parseables: {bad}")

    return dt.dt.to_period(freq)


def window_mask_from_end(vintage_series: pd.Series, end: pd.Period, years: int, freq: str = "Q") -> np.ndarray:
    freq = freq.upper()
    per = parse_vintage_to_period(vintage_series, freq=freq)

    if freq.startswith("Q"):
        n = years * 4
    elif freq.startswith("M"):
        n = years * 12
    else:
        raise ValueError("freq doit être 'Q' ou 'M'.")

    start = end - (n - 1)

    mask = (per >= start) & (per <= end)
    return np.asarray(mask)   # <-- au lieu de .to_numpy()



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
        edges = np.array(fixed_edges, dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf

    # Bucket "brut" : 1 = scores les plus faibles (plus risqués)
    raw_bucket = np.digitize(df["score"], edges[1:], right=True) + 1

    # Convention finale :
    #   bucket 1 = moins risqué (scores les plus élevés)
    #   bucket n = plus risqué (scores les plus faibles)
    df["bucket"] = n_buckets + 1 - raw_bucket

    stats = df.groupby("bucket").agg(
        count=("y", "size"),
        bad=("y", "sum"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).reset_index()

    stats["pd"] = stats["bad"] / stats["count"]
    stats = stats.sort_values("bucket")

    # La PD doit être CROISSANTE avec le numéro de bucket
    mono = bool(np.all(np.diff(stats["pd"].values) >= 0))

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
        gs = GridSearchCV(lr, grid, cv=cv, scoring="neg_log_loss", n_jobs=-1)
        gs.fit(X_model_kept, y_model.values)
        best_lr = gs.best_estimator_

    # -------------------------------------------------
    # Coefficients + erreurs standard + z + p-value (statsmodels)
    # -------------------------------------------------
    with timer.section("Coefficient statistics (statsmodels)"):
        X_sm = sm.add_constant(X_model_kept, has_constant="add")
        logit_sm = sm.Logit(y_model.values, X_sm)
        res_sm = logit_sm.fit(disp=False)

        feature_names = ["Intercept"] + kept_features
        coef_table = pd.DataFrame({
            "feature": feature_names,
            "coef": res_sm.params,
            "std_err": res_sm.bse,
            "z": res_sm.tvalues,
            "p_value": res_sm.pvalues,
        })

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
            calibrator = CalibratedClassifierCV(FrozenEstimator(best_lr), method=calibration_method)
            calibrator.fit(X_cal_kept, y_cal.values)
            model_pd = calibrator
        else:
            model_pd = best_lr

        pd_tr_model = model_pd.predict_proba(X_model_kept)[:, 1]
        pd_va_full = model_pd.predict_proba(Xva_full_kept)[:, 1]
        pd_tr_full = model_pd.predict_proba(Xtr_full_kept)[:, 1]

    # -------------------------------------------------
    # Metrics
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
        "score_tr": score_tr_full,
        "score_va": score_va_full,
        "pd_tr": pd_tr_full,
        "pd_va": pd_va_full,
        "y_tr": ytr_full.values,
        "y_va": yva_full.values,
        "meta_tr": meta_tr,
        "meta_va": meta_va,
        "coef_table": coef_table,
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
             "'train' = train uniquement (window), 'train_val' = train + validation (window).",
    )

    # --- NOUVEAU : fenêtre de grille/TTC ancrée sur dernier vintage de validation ---
    p.add_argument("--grid-window-years", type=int, default=10,
                   help="Fenêtre (années) pour construire la grille (edges), ancrée sur le dernier vintage de validation.")
    p.add_argument("--grid-time-col", default="vintage",
                   help="Colonne temporelle utilisée pour la fenêtre (ex: vintage).")
    p.add_argument("--grid-time-freq", default="Q", choices=["Q", "M"],
                   help="Fréquence de la colonne temporelle (Q ou M).")
    p.add_argument("--ttc-window-years", type=int, default=10,
                   help="Fenêtre (années) pour calculer les PD TTC par grade (bucket_stats), ancrée sur le dernier vintage de validation.")

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

    # Sauvegarde des stats de coefficients
    coef_path = artifacts / "coefficients_stats.csv"
    out["coef_table"].to_csv(coef_path, index=False)
    print(f"✔ coefficients + erreurs standard + z + p-value sauvegardés : {coef_path}")

    # ------------------------------------------------------------------
    # MASTER SCALE (edges)
    # - si --risk-buckets-in fourni : on fige les edges
    # - sinon : edges appris sur fenêtre 10 ans finissant au dernier vintage de validation
    # ------------------------------------------------------------------
    fixed_edges = None
    if args.risk_buckets_in and Path(args.risk_buckets_in).exists():
        fixed_edges = json.loads(Path(args.risk_buckets_in).read_text(encoding="utf-8"))["edges"]

    time_col = args.grid_time_col
    freq = args.grid_time_freq

    if fixed_edges is None:
        if time_col not in out["meta_va"].columns or time_col not in out["meta_tr"].columns:
            raise ValueError(f"Colonne '{time_col}' absente des meta; impossible de windower la grille.")

        va_periods = parse_vintage_to_period(out["meta_va"][time_col], freq=freq)
        end_period = va_periods.max()

        mask_tr_grid = window_mask_from_end(out["meta_tr"][time_col], end_period, args.grid_window_years, freq=freq)
        mask_va_grid = window_mask_from_end(out["meta_va"][time_col], end_period, args.grid_window_years, freq=freq)

        scores_grid = np.concatenate([out["score_tr"][mask_tr_grid], out["score_va"][mask_va_grid]])
        y_grid = np.concatenate([out["y_tr"][mask_tr_grid], out["y_va"][mask_va_grid]])

        if len(y_grid) == 0:
            raise ValueError("Fenêtre de grille vide. Vérifie vintage/freq et les bornes.")

        edges, stats_grid, mono_grid = create_risk_buckets(
            scores_grid, y_grid, args.n_buckets, fixed_edges=None
        )
        print(f"\n[GRILLE] window={args.grid_window_years}y | end={end_period} | monotone(grid)={'OK' if mono_grid else 'NON'}")
    else:
        edges = fixed_edges
        print("\n[GRILLE] edges fournis via --risk-buckets-in (grille figée)")

    # Stats TRAIN/VAL (full) avec ces edges
    _, stats_tr_full, mono_tr = create_risk_buckets(out["score_tr"], out["y_tr"], args.n_buckets, edges)
    _, stats_va_full, mono_va = create_risk_buckets(out["score_va"], out["y_va"], args.n_buckets, edges)

    print("\n[MÉTRIQUES]")
    for k, v in out["metrics"].items():
        print(f"  {k:20s} : {v:.4f}")

    print("\n[MONOTONICITÉ TRAIN] :", "OK" if mono_tr else "NON MONOTONE")
    print("[MONOTONICITÉ VAL]   :", "OK" if mono_va else "NON MONOTONE")

    # ------------------------------------------------------------------
    # PD TTC (bucket_stats) : fenêtre 10 ans finissant au dernier vintage de validation
    # ------------------------------------------------------------------
    if time_col not in out["meta_va"].columns or time_col not in out["meta_tr"].columns:
        raise ValueError(f"Colonne '{time_col}' absente; impossible de calculer TTC window.")

    va_periods = parse_vintage_to_period(out["meta_va"][time_col], freq=freq)
    end_period = va_periods.max()

    mask_tr_ttc = window_mask_from_end(out["meta_tr"][time_col], end_period, args.ttc_window_years, freq=freq)
    mask_va_ttc = window_mask_from_end(out["meta_va"][time_col], end_period, args.ttc_window_years, freq=freq)

    _, stats_tr_win, _ = create_risk_buckets(out["score_tr"][mask_tr_ttc], out["y_tr"][mask_tr_ttc], args.n_buckets, edges)
    _, stats_va_win, _ = create_risk_buckets(out["score_va"][mask_va_ttc], out["y_va"][mask_va_ttc], args.n_buckets, edges)

    # Long-run window (train + validation)
    stats_longrun_win = stats_tr_win[["bucket", "count", "bad", "min_score", "max_score"]].merge(
        stats_va_win[["bucket", "count", "bad", "min_score", "max_score"]],
        on="bucket",
        how="outer",
        suffixes=("_tr", "_va")
    )

    for col in ["count_tr", "count_va", "bad_tr", "bad_va"]:
        stats_longrun_win[col] = stats_longrun_win[col].fillna(0)

    stats_longrun_win["count"] = stats_longrun_win["count_tr"] + stats_longrun_win["count_va"]
    stats_longrun_win["bad"] = stats_longrun_win["bad_tr"] + stats_longrun_win["bad_va"]
    stats_longrun_win["min_score"] = stats_longrun_win[["min_score_tr", "min_score_va"]].min(axis=1)
    stats_longrun_win["max_score"] = stats_longrun_win[["max_score_tr", "max_score_va"]].max(axis=1)
    stats_longrun_win["pd"] = stats_longrun_win["bad"] / stats_longrun_win["count"]
    stats_longrun_win = stats_longrun_win[["bucket", "count", "bad", "min_score", "max_score", "pd"]].sort_values("bucket")

    # Choix de la PD TTC utilisée dans 'train' (sur la fenêtre !)
    if args.ttc_mode == "train":
        stats_train_for_json = stats_tr_win
    else:
        stats_train_for_json = stats_longrun_win

    # ------------------------------------------------------------------
    # Sauvegarde modèle + artefacts
    # ------------------------------------------------------------------
    dump({
        "model_pd": out["model_pd"],
        "best_lr": out["best_lr"],
        "woe_maps": out["woe_maps"],
        "kept_features": out["kept_features"],
        "calibration": args.calibration,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }, artifacts / "model_best.joblib")

    save_json({"edges": edges}, artifacts / "risk_buckets.json")

    bucket_stats_payload = {
        # PD TTC "officielle" (window, train ou train_val selon ttc_mode)
        "train": stats_train_for_json.to_dict(orient="records"),

        # Debug / transparence
        "train_raw_full": stats_tr_full.to_dict(orient="records"),
        "validation_full": stats_va_full.to_dict(orient="records"),
        "train_window": stats_tr_win.to_dict(orient="records"),
        "validation_window": stats_va_win.to_dict(orient="records"),
        "train_val_longrun_window": stats_longrun_win.to_dict(orient="records"),

        "metrics": out["metrics"],
        "ttc_mode": args.ttc_mode,

        "grid_window": {
            "time_col": time_col,
            "freq": freq,
            "end": str(end_period),
            "years": args.grid_window_years,
        },
        "ttc_window": {
            "time_col": time_col,
            "freq": freq,
            "end": str(end_period),
            "years": args.ttc_window_years,
        },
    }
    save_json(bucket_stats_payload, artifacts / "bucket_stats.json")

    # ------------------------------------------------------------------
    # train_scored / validation_scored -> data/processed/scored
    # ------------------------------------------------------------------
    # Grades : 1 = moins risqué, n = plus risqué
    raw_grade_tr = np.digitize(out["score_tr"], np.array(edges)[1:], right=True) + 1
    raw_grade_va = np.digitize(out["score_va"], np.array(edges)[1:], right=True) + 1

    grade_tr = args.n_buckets + 1 - raw_grade_tr
    grade_va = args.n_buckets + 1 - raw_grade_va

    # Train scored
    train_scored = out["meta_tr"].copy()
    train_scored[args.target] = out["y_tr"]
    train_scored["score_ttc"] = out["score_tr"]
    train_scored["pd"] = out["pd_tr"]
    train_scored["grade"] = grade_tr.astype(int)

    train_scored_path = scored_dir / "train_scored.parquet"
    train_scored.to_parquet(train_scored_path, index=False)
    print(f"✔ train_scored sauvegardé : {train_scored_path}")

    # Validation scored
    validation_scored = out["meta_va"].copy()
    validation_scored[args.target] = out["y_va"]
    validation_scored["score_ttc"] = out["score_va"]
    validation_scored["pd"] = out["pd_va"]
    validation_scored["grade"] = grade_va.astype(int)

    validation_scored_path = scored_dir / "validation_scored.parquet"
    validation_scored.to_parquet(validation_scored_path, index=False)
    print(f"✔ validation_scored sauvegardé : {validation_scored_path}")

    print(f"\nMode PD TTC utilisé pour 'train' dans bucket_stats.json : {args.ttc_mode}")
    print(f"Fenêtre grille: {args.grid_window_years}y fin={end_period} | Fenêtre TTC: {args.ttc_window_years}y fin={end_period}")


if __name__ == "__main__":
    main()
