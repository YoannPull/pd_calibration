#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRAINING PIPELINE — CLEAN & ROBUST VERSION (BANK-GRADE)
=======================================================

Principes :
- WOE appris sur 100 % du TRAIN (comme dans les vrais modèles IRB).
- Calibration sur un split interne (20 %) non vu par la LR.
- Logistic Regression L2 propre (pas de class_weight='balanced').
- Score TTC basé sur log-odds non calibrés (standard bancaire).
- PD calibrée via CalibratedClassifierCV + FrozenEstimator (isotonic/sigmoid).
- Master Scale (n_buckets) monotone (contrôle de monotonie).
- Sauvegarde train_scored / validation_scored avec vintage & loan_sequence_number.

NOUVEAU (grille + TTC windowées) :
- Si --risk-buckets-in n'est PAS fourni :
  - la grille (edges) est construite sur une fenêtre glissante (grid-window-years)
    terminant au dernier trimestre (vintage) présent dans VALIDATION.
  - les PD TTC par grade (bucket_stats['train'] / 'train_val_longrun_window') sont aussi
    calculées sur une fenêtre glissante (ttc-window-years) ancrée au même end.

Espace de recherche (modèle) :
- Recherche large sur C via logspace, avec deux modes :
  - --search grid    : exhaustif
  - --search halving : Successive Halving (explore large, élimine tôt)

Option Statsmodels :
- --coef-stats statsmodels : calcule coef/std_err/z/p_value via sm.Logit (plus lent)
- --coef-stats none        : skip (accélère)

Timer / progress :
- --timing : affiche les sections + temps cumulé
- --sklearn-verbose : verbose GridSearch/Halving (attention aux logs trop bavards)

Outputs:
- artifacts/model_from_binned/model_best.joblib
- artifacts/model_from_binned/risk_buckets.json
- artifacts/model_from_binned/bucket_stats.json
- artifacts/model_from_binned/coefficients_stats.csv
- data/processed/scored/train_scored.parquet
- data/processed/scored/validation_scored.parquet
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import statsmodels.api as sm
from joblib import dump

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# Successive Halving (sklearn >= 0.24, mais activé via experimental)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _fmt_seconds(s: float) -> str:
    s = int(round(s))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:d}m{s:02d}s"


class Timer:
    """
    Timer de sections avec temps cumulé.
    Coût négligeable (perf_counter + prints si live).
    """
    def __init__(self, live: bool = False):
        self.records: Dict[str, float] = {}
        self.live = live
        self.t0 = time.perf_counter()

    @contextlib.contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        if self.live:
            total = time.perf_counter() - self.t0
            print(f"▶ {name}  (total={_fmt_seconds(total)})", flush=True)
        yield
        dt = time.perf_counter() - t0
        self.records[name] = dt
        if self.live:
            total = time.perf_counter() - self.t0
            print(f"  ✓ {name:35s} {dt:9.3f}s   (total={_fmt_seconds(total)})", flush=True)

    def summary(self, top: int = 20) -> str:
        items = sorted(self.records.items(), key=lambda kv: kv[1], reverse=True)[:top]
        lines = ["\n[TIMING SUMMARY]"]
        for k, v in items:
            lines.append(f"  - {k:35s} {v:9.3f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Time helpers (vintage window)
# ---------------------------------------------------------------------------

def parse_vintage_to_period(s: pd.Series, freq: str = "Q") -> pd.PeriodIndex:
    """
    Convertit une colonne vintage en PeriodIndex (Q ou M).
    Supporte :
      - '2015Q1' ou '2015-Q1' (si freq=Q)
      - dates parseables (ex '2015-03-31') -> conversion en trimestre/mois
      - Period déjà présent
    """
    freq = freq.upper()

    if isinstance(s.dtype, pd.PeriodDtype):
        return pd.PeriodIndex(s.astype(f"period[{freq}]"))

    s_str = s.astype(str)

    # formats type 'YYYYQn' ou 'YYYY-Qn' (uniquement si freq=Q)
    m = s_str.str.match(r"^\d{4}[- ]?Q[1-4]$")
    if m.all() and freq.startswith("Q"):
        cleaned = s_str.str.replace(" ", "", regex=False).str.replace("-", "", regex=False)
        return pd.PeriodIndex(cleaned, freq="Q")

    # fallback: parse date
    dt = pd.to_datetime(s_str, errors="coerce")
    ok_ratio = float(dt.notna().mean())
    if ok_ratio < 0.95:
        bad = s_str[dt.isna()].head(5).tolist()
        raise ValueError(
            f"Impossible de parser vintage (ratio ok={ok_ratio:.2%}). Exemples non parseables: {bad}"
        )

    return dt.dt.to_period(freq).astype(f"period[{freq}]")  # type: ignore


def window_mask_from_end(vintage_series: pd.Series, end: pd.Period, years: int, freq: str = "Q") -> np.ndarray:
    """
    Masque booléen sélectionnant une fenêtre [end - (years*freq) + 1, end].
    """
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
    return np.asarray(mask)


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
    """WOE map learned on full TRAIN."""
    y = df[target].astype(int)
    tot_bad = float(y.sum())
    tot_good = float(len(y) - y.sum())
    global_woe = float(np.log((tot_bad + smooth) / (tot_good + smooth)))

    maps: Dict[str, Any] = {}
    for raw, bin_col in raw_to_bin.items():
        tab = df.groupby(bin_col, observed=True)[target].agg(["sum", "count"])
        tab["good"] = tab["count"] - tab["sum"]

        bad_i = tab["sum"].to_numpy(dtype=float)
        good_i = tab["good"].to_numpy(dtype=float)

        woe_i = np.log((bad_i + smooth) / (good_i + smooth))

        maps[raw] = {
            "map": {int(k): float(v) for k, v in zip(tab.index, woe_i)},
            "default": global_woe,
        }
    return maps


def apply_woe(df: pd.DataFrame, woe_maps: Dict[str, Any], bin_suffix: str, dtype=np.float32) -> pd.DataFrame:
    """
    Applique les mappings WOE. Optimisé pour éviter concat de centaines de Series.
    """
    out = pd.DataFrame(index=df.index)
    for raw, info in woe_maps.items():
        bin_col1 = f"{bin_suffix}{raw}"
        bin_col2 = f"{raw}{bin_suffix}"
        colbin = bin_col1 if bin_col1 in df.columns else (bin_col2 if bin_col2 in df.columns else None)

        mapping = info["map"]
        default = info["default"]

        if colbin is None:
            out[f"{raw}_WOE"] = np.asarray(default, dtype=dtype)
        else:
            out[f"{raw}_WOE"] = df[colbin].map(mapping).fillna(default).astype(dtype)

    return out


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    # Interactions explicites (limitées)
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

def scale_score(log_odds: np.ndarray, base_points: int = 600, base_odds: int = 50, pdo: int = 20) -> np.ndarray:
    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)
    return np.round(offset - factor * log_odds).astype(int)


# ---------------------------------------------------------------------------
# Feature selection (optimisée — évite X.corr() O(p^2))
# ---------------------------------------------------------------------------

def select_features(X: pd.DataFrame, corr_thr: float = 0.85) -> List[str]:
    """
    Greedy selection :
      - ordre décroissant de variance
      - garde une feature si corr(|.|) avec toutes les kept < corr_thr

    Optimisation :
      - pas de matrice corr p×p
      - calcule uniquement corr(feature_candidate, features_kept)
    """
    X = X.fillna(0).astype(np.float32)

    cols = X.var().sort_values(ascending=False).index.to_list()
    Xv = X[cols].to_numpy(copy=False)  # (n, p)

    n, p = Xv.shape
    means = Xv.mean(axis=0)
    stds = Xv.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0

    kept_cols: List[str] = []
    kept_idx: List[int] = []

    for j in range(p):
        if not kept_idx:
            kept_cols.append(cols[j])
            kept_idx.append(j)
            continue

        x = Xv[:, j]
        K = Xv[:, kept_idx]  # (n, k)

        xk_mean = (K.T @ x) / n
        cov = xk_mean - means[kept_idx] * means[j]
        corr = np.abs(cov / (stds[kept_idx] * stds[j]))

        if float(corr.max()) < corr_thr:
            kept_cols.append(cols[j])
            kept_idx.append(j)

    return kept_cols


# ---------------------------------------------------------------------------
# Master Scale
# ---------------------------------------------------------------------------

def create_risk_buckets(
    scores: np.ndarray,
    y: np.ndarray,
    n_buckets: int = 10,
    fixed_edges: Optional[List[float]] = None
) -> Tuple[List[float], pd.DataFrame, bool]:
    """
    Convention :
      - bucket 1 = moins risqué (scores les plus élevés)
      - bucket n = plus risqué (scores les plus faibles)
      - PD doit être croissante avec bucket
    """
    scores = np.asarray(scores)
    y = np.asarray(y).astype(int)

    if fixed_edges is None:
        qs = np.linspace(0, 1, n_buckets + 1)
        edges = np.quantile(scores, qs)

        edges[0] = -np.inf
        edges[-1] = np.inf

        if len(np.unique(edges)) < len(edges):
            print("[WARN] Quantile edges contiennent des doublons. Risque de buckets vides.", flush=True)
    else:
        edges = np.array(fixed_edges, dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf

    raw_bucket = np.digitize(scores, edges[1:], right=True) + 1
    bucket = (n_buckets + 1 - raw_bucket).astype(int)

    df = pd.DataFrame({"bucket": bucket, "y": y, "score": scores})

    stats = df.groupby("bucket", observed=True).agg(
        count=("y", "size"),
        bad=("y", "sum"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).reset_index()

    stats["pd"] = stats["bad"] / stats["count"]
    stats = stats.sort_values("bucket")

    mono = bool(np.all(np.diff(stats["pd"].to_numpy()) >= 0))

    return edges.tolist(), stats, mono


def choose_n_buckets(
    scores_grid: np.ndarray,
    y_grid: np.ndarray,
    candidates: List[int],
    min_bucket_count: int,
    min_bucket_bad: int,
) -> int:
    """
    Choisit le plus grand nombre de buckets respectant :
      - monotonie PD sur la grille
      - count >= min_bucket_count
      - bad >= min_bucket_bad
    Sinon fallback: le premier candidat.
    """
    best = None
    for nb in sorted(candidates):
        _, stats, mono = create_risk_buckets(scores_grid, y_grid, n_buckets=nb, fixed_edges=None)
        ok_counts = bool((stats["count"] >= min_bucket_count).all())
        ok_bads = bool((stats["bad"] >= min_bucket_bad).all())
        if mono and ok_counts and ok_bads:
            best = nb
    return best if best is not None else candidates[0]


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
    *,
    search: str,
    c_min_exp: float,
    c_max_exp: float,
    c_num: int,
    halving_factor: int,
    lr_solver: str,
    lr_max_iter: int,
    coef_stats: str,
    sklearn_verbose: int,
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

        def safe(col: str) -> bool:
            raw = raw_name_from_bin(col, bin_suffix).lower()
            return not any(bad in raw for bad in BLACKLIST)

        bin_cols = [c for c in bin_cols if safe(c)]
        raw_to_bin = {raw_name_from_bin(c, bin_suffix): c for c in bin_cols}

    # -------------------------------------------------
    # WOE learn on full TRAIN
    # -------------------------------------------------
    with timer.section("Learn + Apply WOE"):
        woe_maps = build_woe_maps(df_tr, target, rawaws_to_bin := raw_to_bin)  # noqa: F841

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

        X_model_kept = X_model[kept_features].to_numpy(dtype=np.float32, copy=False)
        X_cal_kept = X_cal[kept_features].to_numpy(dtype=np.float32, copy=False)
        Xtr_full_kept = Xtr_full[kept_features].to_numpy(dtype=np.float32, copy=False)
        Xva_full_kept = Xva_full[kept_features].to_numpy(dtype=np.float32, copy=False)

    # -------------------------------------------------
    # Logistic Regression (GridSearch ou Halving)
    # -------------------------------------------------
    with timer.section("Fit Logistic Regression (search)"):
        C_grid = np.logspace(c_min_exp, c_max_exp, c_num).tolist()
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        print(f"[SEARCH] mode={search} | C in 10^[{c_min_exp},{c_max_exp}] ({c_num} pts) | folds={cv_folds}", flush=True)
        if search == "grid":
            print(f"[SEARCH] expected fits ~= {len(C_grid)} × {cv_folds} = {len(C_grid)*cv_folds}", flush=True)

        lr = LogisticRegression(
            penalty="l2",
            solver=lr_solver,
            max_iter=lr_max_iter,
        )

        if search == "halving":
            gs = HalvingGridSearchCV(
                lr,
                param_grid={"C": C_grid},
                scoring="neg_log_loss",
                cv=cv,
                n_jobs=-1,
                factor=halving_factor,
                verbose=sklearn_verbose,
            )
        else:
            gs = GridSearchCV(
                lr,
                param_grid={"C": C_grid},
                scoring="neg_log_loss",
                cv=cv,
                n_jobs=-1,
                verbose=sklearn_verbose,
            )

        gs.fit(X_model_kept, y_model.to_numpy())
        best_lr: LogisticRegression = gs.best_estimator_

    # -------------------------------------------------
    # Coefficients stats (optionnel)
    # -------------------------------------------------
    with timer.section("Coefficient statistics (optional)"):
        feature_names = ["Intercept"] + kept_features

        if coef_stats == "statsmodels":
            try:
                X_sm = sm.add_constant(X_model_kept, has_constant="add")
                logit_sm = sm.Logit(y_model.to_numpy(), X_sm)
                res_sm = logit_sm.fit(disp=False)

                coef_table = pd.DataFrame({
                    "feature": feature_names,
                    "coef": res_sm.params,
                    "std_err": res_sm.bse,
                    "z": res_sm.tvalues,
                    "p_value": res_sm.pvalues,
                })
            except Exception as e:
                print(f"[WARN] statsmodels a échoué ({type(e).__name__}: {e}). Fallback coef sklearn.", flush=True)
                coefs = np.r_[best_lr.intercept_[0], best_lr.coef_[0]]
                coef_table = pd.DataFrame({
                    "feature": feature_names,
                    "coef": coefs,
                    "std_err": np.nan,
                    "z": np.nan,
                    "p_value": np.nan,
                })
        else:
            coefs = np.r_[best_lr.intercept_[0], best_lr.coef_[0]]
            coef_table = pd.DataFrame({
                "feature": feature_names,
                "coef": coefs,
                "std_err": np.nan,
                "z": np.nan,
                "p_value": np.nan,
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
    # Calibration (isotonic / sigmoid / none)
    # -------------------------------------------------
    with timer.section("Calibration"):
        if calibration_method != "none":
            calibrator = CalibratedClassifierCV(
                FrozenEstimator(best_lr),
                method=calibration_method,
            )
            calibrator.fit(X_cal_kept, y_cal.to_numpy())
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
        "n_features": int(len(kept_features)),
        "search": search,
        "c_min_exp": float(c_min_exp),
        "c_max_exp": float(c_max_exp),
        "c_num": int(c_num),
        "best_C": float(best_lr.C),
        "solver": lr_solver,
        "coef_stats": coef_stats,
        "calibration": calibration_method,
        "halving_factor": int(halving_factor),
        "lr_max_iter": int(lr_max_iter),
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
        "y_tr": ytr_full.to_numpy(),
        "y_va": yva_full.to_numpy(),
        "meta_tr": meta_tr,
        "meta_va": meta_va,
        "coef_table": coef_table,
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--target", required=True)

    # io
    p.add_argument("--artifacts", default="artifacts/model_from_binned")
    p.add_argument("--scored-outdir", default="data/processed/scored")
    p.add_argument("--risk-buckets-in", default=None)

    # feature engineering
    p.add_argument("--bin-suffix", default="__BIN")
    p.add_argument("--corr-threshold", type=float, default=0.85)

    # model / search
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--calibration", choices=["none", "isotonic", "sigmoid"], default="isotonic")

    p.add_argument("--search", choices=["grid", "halving"], default="halving",
                   help="grid=exhaustif, halving=successive halving (explore large + prune tôt).")
    p.add_argument("--c-min-exp", type=float, default=-6.0, help="C_min = 10^(c-min-exp)")
    p.add_argument("--c-max-exp", type=float, default=3.0, help="C_max = 10^(c-max-exp)")
    p.add_argument("--c-num", type=int, default=30, help="Nombre de points dans la grille logspace(C)")
    p.add_argument("--halving-factor", type=int, default=3, help="Facteur d'élimination HalvingGridSearchCV")
    p.add_argument("--lr-solver", choices=["lbfgs", "saga", "newton-cg"], default="lbfgs")
    p.add_argument("--lr-max-iter", type=int, default=3000)

    # sklearn verbosity
    p.add_argument("--sklearn-verbose", type=int, default=0,
                   help="Verbosity sklearn pour (Halving)GridSearchCV. 0=silent, 1=some, 2=chatty.")

    # statsmodels option
    p.add_argument("--coef-stats", choices=["statsmodels", "none"], default="statsmodels",
                   help="statsmodels=coef/std_err/z/p_value ; none=skip pour accélérer.")

    # master scale
    p.add_argument("--n-buckets", type=int, default=10)

    # optional: search best n_buckets under constraints (governance)
    p.add_argument("--n-buckets-candidates", default=None,
                   help="Optionnel: liste CSV ex '7,10,12,15' pour choisir nb buckets max sous contraintes.")
    p.add_argument("--min-bucket-count", type=int, default=300, help="Contraintes si n-buckets-candidates est utilisé.")
    p.add_argument("--min-bucket-bad", type=int, default=5, help="Contraintes si n-buckets-candidates est utilisé.")

    # ttc mode
    p.add_argument("--ttc-mode", choices=["train", "train_val"], default="train",
                   help="'train' = PD TTC sur TRAIN(window), 'train_val' = TRAIN+VAL(window).")

    # windowing
    default_window = 5
    p.add_argument("--grid-window-years", type=int, default=default_window,
                   help="Fenêtre (années) pour construire la grille (edges), ancrée sur le dernier vintage de validation.")
    p.add_argument("--grid-time-col", default="vintage",
                   help="Colonne temporelle utilisée pour la fenêtre (ex: vintage).")
    p.add_argument("--grid-time-freq", default="Q", choices=["Q", "M"],
                   help="Fréquence de la colonne temporelle (Q ou M).")
    p.add_argument("--ttc-window-years", type=int, default=default_window,
                   help="Fenêtre (années) pour calculer les PD TTC par grade (bucket_stats), ancrée sur le dernier vintage de validation.")

    # misc
    p.add_argument("--timing", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

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
        df_tr=df_tr,
        df_va=df_va,
        target=args.target,
        bin_suffix=args.bin_suffix,
        corr_threshold=args.corr_threshold,
        cv_folds=args.cv_folds,
        calibration_method=args.calibration,
        timer=timer,
        search=args.search,
        c_min_exp=args.c_min_exp,
        c_max_exp=args.c_max_exp,
        c_num=args.c_num,
        halving_factor=args.halving_factor,
        lr_solver=args.lr_solver,
        lr_max_iter=args.lr_max_iter,
        coef_stats=args.coef_stats,
        sklearn_verbose=args.sklearn_verbose,
    )

    # save coef stats
    coef_path = artifacts / "coefficients_stats.csv"
    out["coef_table"].to_csv(coef_path, index=False)
    print(f"✔ coefficients stats sauvegardés : {coef_path}", flush=True)

    # ------------------------------------------------------------------
    # MASTER SCALE (edges)
    # - si --risk-buckets-in fourni : edges figés
    # - sinon : edges appris sur fenêtre (grid-window-years) finissant au dernier vintage de validation
    # ------------------------------------------------------------------
    fixed_edges = None
    if args.risk_buckets_in and Path(args.risk_buckets_in).exists():
        fixed_edges = json.loads(Path(args.risk_buckets_in).read_text(encoding="utf-8"))["edges"]

    time_col = args.grid_time_col
    freq = args.grid_time_freq

    if time_col not in out["meta_va"].columns or time_col not in out["meta_tr"].columns:
        raise ValueError(f"Colonne '{time_col}' absente des meta; impossible de windower la grille/TTC.")

    va_periods = parse_vintage_to_period(out["meta_va"][time_col], freq=freq)
    end_period = va_periods.max()

    if fixed_edges is None:
        mask_tr_grid = window_mask_from_end(out["meta_tr"][time_col], end_period, args.grid_window_years, freq=freq)
        mask_va_grid = window_mask_from_end(out["meta_va"][time_col], end_period, args.grid_window_years, freq=freq)

        scores_grid = np.concatenate([out["score_tr"][mask_tr_grid], out["score_va"][mask_va_grid]])
        y_grid = np.concatenate([out["y_tr"][mask_tr_grid], out["y_va"][mask_va_grid]])

        if len(y_grid) == 0:
            raise ValueError("Fenêtre de grille vide. Vérifie vintage/freq et les bornes.")

        n_buckets = int(args.n_buckets)
        if args.n_buckets_candidates:
            cands = [int(x.strip()) for x in args.n_buckets_candidates.split(",") if x.strip()]
            if not cands:
                raise ValueError("--n-buckets-candidates fourni mais vide.")
            n_buckets = choose_n_buckets(
                scores_grid=scores_grid,
                y_grid=y_grid,
                candidates=cands,
                min_bucket_count=args.min_bucket_count,
                min_bucket_bad=args.min_bucket_bad,
            )
            print(f"[N_BUCKETS] choisi automatiquement = {n_buckets} (candidats={cands})", flush=True)

        edges, stats_grid, mono_grid = create_risk_buckets(scores_grid, y_grid, n_buckets=n_buckets, fixed_edges=None)
        print(
            f"\n[GRILLE] window={args.grid_window_years}y | end={end_period} "
            f"| monotone(grid)={'OK' if mono_grid else 'NON'}",
            flush=True
        )

        args.n_buckets = n_buckets
    else:
        edges = fixed_edges
        print("\n[GRILLE] edges fournis via --risk-buckets-in (grille figée)", flush=True)

    # Stats TRAIN/VAL (full) avec ces edges
    _, stats_tr_full, mono_tr = create_risk_buckets(out["score_tr"], out["y_tr"], n_buckets=args.n_buckets, fixed_edges=edges)
    _, stats_va_full, mono_va = create_risk_buckets(out["score_va"], out["y_va"], n_buckets=args.n_buckets, fixed_edges=edges)

    print("\n[MÉTRIQUES]", flush=True)
    for k, v in out["metrics"].items():
        if isinstance(v, float):
            print(f"  {k:22s} : {v:.6f}", flush=True)
        else:
            print(f"  {k:22s} : {v}", flush=True)

    print("\n[MONOTONICITÉ TRAIN] :", "OK" if mono_tr else "NON MONOTONE", flush=True)
    print("[MONOTONICITÉ VAL]   :", "OK" if mono_va else "NON MONOTONE", flush=True)

    # ------------------------------------------------------------------
    # PD TTC (bucket_stats) : fenêtre ttc-window-years finissant au dernier vintage de validation
    # ------------------------------------------------------------------
    mask_tr_ttc = window_mask_from_end(out["meta_tr"][time_col], end_period, args.ttc_window_years, freq=freq)
    mask_va_ttc = window_mask_from_end(out["meta_va"][time_col], end_period, args.ttc_window_years, freq=freq)

    _, stats_tr_win, _ = create_risk_buckets(
        out["score_tr"][mask_tr_ttc],
        out["y_tr"][mask_tr_ttc],
        n_buckets=args.n_buckets,
        fixed_edges=edges
    )
    _, stats_va_win, _ = create_risk_buckets(
        out["score_va"][mask_va_ttc],
        out["y_va"][mask_va_ttc],
        n_buckets=args.n_buckets,
        fixed_edges=edges
    )

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

    stats_train_for_json = stats_tr_win if args.ttc_mode == "train" else stats_longrun_win

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    dump({
        "model_pd": out["model_pd"],
        "best_lr": out["best_lr"],
        "woe_maps": out["woe_maps"],
        "kept_features": out["kept_features"],
        "calibration": args.calibration,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": out["metrics"],
    }, artifacts / "model_best.joblib")

    save_json({"edges": edges}, artifacts / "risk_buckets.json")

    bucket_stats_payload = {
        # PD TTC officielle (window, train ou train_val selon ttc_mode)
        "train": stats_train_for_json.to_dict(orient="records"),

        # transparence / debug
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
        "n_buckets": int(args.n_buckets),
    }
    save_json(bucket_stats_payload, artifacts / "bucket_stats.json")

    # ------------------------------------------------------------------
    # train_scored / validation_scored
    # ------------------------------------------------------------------
    edges_arr = np.array(edges, dtype=float)

    raw_grade_tr = np.digitize(out["score_tr"], edges_arr[1:], right=True) + 1
    raw_grade_va = np.digitize(out["score_va"], edges_arr[1:], right=True) + 1

    grade_tr = (args.n_buckets + 1 - raw_grade_tr).astype(int)
    grade_va = (args.n_buckets + 1 - raw_grade_va).astype(int)

    train_scored = out["meta_tr"].copy()
    train_scored[args.target] = out["y_tr"]
    train_scored["score_ttc"] = out["score_tr"]
    train_scored["pd"] = out["pd_tr"]
    train_scored["grade"] = grade_tr

    train_scored_path = scored_dir / "train_scored.parquet"
    train_scored.to_parquet(train_scored_path, index=False)
    print(f"✔ train_scored sauvegardé : {train_scored_path}", flush=True)

    validation_scored = out["meta_va"].copy()
    validation_scored[args.target] = out["y_va"]
    validation_scored["score_ttc"] = out["score_va"]
    validation_scored["pd"] = out["pd_va"]
    validation_scored["grade"] = grade_va

    validation_scored_path = scored_dir / "validation_scored.parquet"
    validation_scored.to_parquet(validation_scored_path, index=False)
    print(f"✔ validation_scored sauvegardé : {validation_scored_path}", flush=True)

    print(f"\nMode PD TTC utilisé pour 'train' : {args.ttc_mode}", flush=True)
    print(
        f"Fenêtre grille: {args.grid_window_years}y fin={end_period} | "
        f"Fenêtre TTC: {args.ttc_window_years}y fin={end_period}",
        flush=True
    )

    if args.timing:
        print(timer.summary(), flush=True)


if __name__ == "__main__":
    main()
