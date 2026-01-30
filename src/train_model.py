#!/usr/bin/env python3
# src/train_model.py
# -*- coding: utf-8 -*-

"""
TRAINING PIPELINE — BANK-GRADE (TIME-AWARE CV + OPTIONAL INTERACTIONS)
=====================================================================

Principles
----------
- WOE is learned on 100% of the TRAIN sample.
- An internal calibration split is created (default: "time_last" to mimic OOT usage).
- Logistic Regression with L2 penalty (no class_weight='balanced').
- TTC score is computed from *uncalibrated* log-odds.
- PD is calibrated via CalibratedClassifierCV + FrozenEstimator (isotonic/sigmoid).
- Master Scale (n_buckets) is checked for monotonicity (sanity control).
- Train/validation scored outputs are saved with 'vintage' and 'loan_sequence_number'.

New
---
- Wide search over C using logspace:
    --search grid    : exhaustive grid-search
    --search halving : Successive Halving (broad exploration, early pruning)
- Time-aware CV to select C:
    --cv-scheme time       : expanding window over vintages (recommended for OOT)
    --cv-scheme stratified : StratifiedKFold (fallback)
- Time-aware calibration split:
    --calibration-split time_last   : last vintages from train (recommended)
    --calibration-split stratified  : random stratified split
- Optional interactions:
    --no-interactions : disables WOE interactions

Windows (grid + TTC)
--------------------
- If --risk-buckets-in is NOT provided:
  - score edges are learned on a rolling window (grid-window-years) ending at the last validation vintage
  - TTC PDs by grade are computed on another rolling window (ttc-window-years) ending at the same end

Optional statsmodels
--------------------
- --coef-stats statsmodels : coef/std_err/z/p_value via sm.Logit (slower)
- --coef-stats none        : skip (faster)

Outputs
-------
- artifacts/.../model_best.joblib
- artifacts/.../risk_buckets.json
- artifacts/.../bucket_stats.json
- artifacts/.../coefficients_stats.csv
- data/.../train_scored.parquet
- data/.../validation_scored.parquet
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

from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Successive Halving (sklearn >= 0.24, enabled via experimental)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV

# Optional (coefficient statistics)
try:
    import statsmodels.api as sm
except Exception:
    sm = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_any(path: str) -> pd.DataFrame:
    """Load a DataFrame from parquet or csv (based on file extension)."""
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_json(obj: dict, path: Path):
    """Write a JSON artifact with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


class Timer:
    """Lightweight wall-clock timer to profile main pipeline sections."""
    def __init__(self, live: bool = False):
        self.records: Dict[str, float] = {}
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
            print(f"  ✓ {name:40s} {dt:.3f}s")


# ---------------------------------------------------------------------------
# Time helpers (vintage window + time-aware CV)
# ---------------------------------------------------------------------------

def parse_vintage_to_period(s: pd.Series, freq: str = "Q") -> pd.PeriodIndex:
    """
    Convert a vintage column into a PeriodIndex (Q or M).

    Supported inputs:
      - '2015Q1' or '2015-Q1' (if freq='Q')
      - parseable date strings (e.g., '2015-03-31') -> converted to quarter/month
      - already-a-Period dtype
    """
    freq = freq.upper()

    if isinstance(s.dtype, pd.PeriodDtype):
        return pd.PeriodIndex(s.astype(f"period[{freq}]"))

    s_str = s.astype(str)

    # Formats like 'YYYYQn' or 'YYYY-Qn' (only for quarterly frequency)
    m = s_str.str.match(r"^\d{4}[- ]?Q[1-4]$")
    if m.all() and freq.startswith("Q"):
        cleaned = s_str.str.replace(" ", "", regex=False).str.replace("-", "", regex=False)
        return pd.PeriodIndex(cleaned, freq="Q")

    # Fallback: parse as dates
    dt = pd.to_datetime(s_str, errors="coerce")
    ok_ratio = float(dt.notna().mean())
    if ok_ratio < 0.95:
        bad = s_str[dt.isna()].head(5).tolist()
        raise ValueError(
            f"Cannot parse vintage values (ok ratio={ok_ratio:.2%}). Examples: {bad}"
        )

    return dt.dt.to_period(freq).astype(f"period[{freq}]")  # type: ignore


def window_mask_from_end(vintage_series: pd.Series, end: pd.Period, years: int, freq: str = "Q") -> np.ndarray:
    """
    Return a boolean mask selecting the rolling window:
        [end - (years*freq) + 1, end]
    """
    freq = freq.upper()
    per = parse_vintage_to_period(vintage_series, freq=freq)

    if freq.startswith("Q"):
        n = years * 4
    elif freq.startswith("M"):
        n = years * 12
    else:
        raise ValueError("freq must be 'Q' or 'M'.")

    start = end - (n - 1)
    mask = (per >= start) & (per <= end)
    return np.asarray(mask)


class VintageExpandingWindowSplit:
    """
    Expanding-window cross-validation over vintages (time periods).

    Fold i:
      - train = all periods strictly before the test block
      - test  = one contiguous block of periods
    """
    def __init__(self, n_splits: int = 5):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("VintageExpandingWindowSplit requires groups=periods.")
        g = np.asarray(groups)

        uniq = np.unique(g)
        uniq = np.sort(uniq)

        # Split periods into n_splits test blocks (oldest -> newest)
        blocks = np.array_split(uniq, self.n_splits)

        # Expanding window: start at i=1 so that train is non-empty
        for i in range(1, len(blocks)):
            test_periods = blocks[i]
            train_periods = np.concatenate(blocks[:i])

            test_idx = np.where(np.isin(g, test_periods))[0]
            train_idx = np.where(np.isin(g, train_periods))[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits - 1


def _time_aware_calibration_split(
    X: pd.DataFrame,
    y: pd.Series,
    vintages: pd.Series,
    frac: float,
    freq: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-aware calibration split ("time_last"):
      - calibration set = last vintages (approximately frac of the unique periods)
      - model set       = remaining earlier vintages
    """
    per = parse_vintage_to_period(vintages, freq=freq)
    uniq = np.sort(per.unique())

    if len(uniq) < 3:
        # Not enough periods -> fallback to a simple stratified split
        return _stratified_calibration_split(X, y, frac=frac, seed=42)

    n_cal = max(1, int(np.ceil(len(uniq) * frac)))
    cal_periods = set(uniq[-n_cal:])

    mask_cal = per.isin(cal_periods)
    mask_model = ~mask_cal

    X_model = X.loc[mask_model]
    y_model = y.loc[mask_model]
    X_cal = X.loc[mask_cal]
    y_cal = y.loc[mask_cal]

    # If calibration is too small or degenerate -> fallback stratified
    if len(X_cal) < 100 or y_cal.nunique() < 2 or y_model.nunique() < 2:
        return _stratified_calibration_split(X, y, frac=frac, seed=42)

    return X_model, X_cal, y_model, y_cal


def _stratified_calibration_split(
    X: pd.DataFrame,
    y: pd.Series,
    frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified random split used as a robust fallback."""
    from sklearn.model_selection import train_test_split
    X_model, X_cal, y_model, y_cal = train_test_split(
        X, y, test_size=frac, stratify=y, random_state=seed
    )
    return X_model, X_cal, y_model, y_cal


# ---------------------------------------------------------------------------
# WOE utilities
# ---------------------------------------------------------------------------

def raw_name_from_bin(col: str, tag: str) -> str:
    """Recover the raw feature name from a binned column name."""
    return col.replace(tag, "") if tag in col else col


def build_woe_maps(
    df: pd.DataFrame,
    target: str,
    raw_to_bin: Dict[str, str],
    smooth: float = 0.5
) -> Dict[str, Any]:
    """Learn WOE mappings on the full TRAIN set (grade-level smoothing via pseudo-counts)."""
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
    Apply WOE mappings.

    Implementation note:
    - We build the output DataFrame column-by-column to avoid concatenating many Series,
      which is typically faster and more memory-friendly.
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
    """Add a small set of explicit interaction terms (optional)."""
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
    """Convert log-odds into a TTC score using a standard points-to-double-odds scaling."""
    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)
    return np.round(offset - factor * log_odds).astype(int)


# ---------------------------------------------------------------------------
# Feature selection (avoids full O(p^2) correlation matrix)
# ---------------------------------------------------------------------------

def select_features(X: pd.DataFrame, corr_thr: float = 0.85) -> List[str]:
    """
    Greedy correlation-based selection:
      - sort features by decreasing variance
      - keep a candidate feature if its absolute correlation with all kept features is < corr_thr

    Efficiency:
      - we do NOT compute the full p×p correlation matrix
      - we only compute correlations between the candidate and the already-kept set
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
    Create master-scale buckets from TTC scores.

    Convention:
      - bucket 1 = least risky (highest scores)
      - bucket n = most risky  (lowest scores)
      - observed PD is expected to be non-decreasing with bucket index
    """
    scores = np.asarray(scores)
    y = np.asarray(y).astype(int)

    if fixed_edges is None:
        qs = np.linspace(0, 1, n_buckets + 1)
        edges = np.quantile(scores, qs)
        edges[0] = -np.inf
        edges[-1] = np.inf

        if len(np.unique(edges)) < len(edges):
            print("[WARN] Quantile edges contain duplicates (risk of empty buckets).")
    else:
        edges = np.array(fixed_edges, dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf

    raw_bucket = np.digitize(scores, edges[1:], right=True) + 1  # 1=low score (riskier)
    bucket = (n_buckets + 1 - raw_bucket).astype(int)            # 1=high score (less risky)

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
    Pick the *largest* number of buckets that satisfies:
      - monotonic PD over the grid
      - count >= min_bucket_count in every bucket
      - bad   >= min_bucket_bad   in every bucket

    If none satisfies the constraints, fall back to the first candidate.
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

META_COLS = ["vintage", "loan_sequence_number"]  # kept for reporting; never used as features


def _build_cv_splitter(
    scheme: str,
    X_model: pd.DataFrame,
    y_model: pd.Series,
    meta_tr: pd.DataFrame,
    time_col: str,
    freq: str,
    cv_folds: int,
) -> Tuple[object, Optional[np.ndarray]]:
    """
    Return (cv_splitter, groups):
      - stratified -> StratifiedKFold, groups=None
      - time       -> VintageExpandingWindowSplit + integer-encoded groups
    """
    scheme = scheme.lower()
    if scheme == "stratified":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        return cv, None

    # Time-aware scheme
    if time_col not in meta_tr.columns:
        print(f"[WARN] cv-scheme=time but '{time_col}' is missing. Falling back to stratified.")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        return cv, None

    vint = meta_tr.loc[X_model.index, time_col]
    per = parse_vintage_to_period(vint, freq=freq)

    # Encode groups as ordered integers (stable mapping)
    uniq = np.sort(per.unique())
    mapping = {p: i for i, p in enumerate(uniq)}
    groups = np.array([mapping[p] for p in per], dtype=int)

    cv_time = VintageExpandingWindowSplit(n_splits=cv_folds)

    # Sanity check: ensure each test fold has both classes
    bad_fold = False
    for tr_idx, te_idx in cv_time.split(X_model, y_model, groups=groups):
        if pd.Series(y_model.to_numpy()[te_idx]).nunique() < 2:
            bad_fold = True
            break
    if bad_fold:
        print("[WARN] At least one time-aware fold has a single class in test. Falling back to stratified.")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        return cv, None

    return cv_time, groups

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
    search_verbose: int,
    lr_solver: str,
    lr_max_iter: int,
    coef_stats: str,
    use_interactions: bool,
    cv_scheme: str,
    calibration_split: str,
    calibration_size: float,
    cv_time_col: str,
    cv_time_freq: str,
):
    # Meta (kept for reporting; never used as features)
    meta_cols_present = [c for c in META_COLS if c in df_tr.columns and c in df_va.columns]
    meta_tr = df_tr[meta_cols_present].copy()
    meta_va = df_va[meta_cols_present].copy()

    # Identify binned features
    with timer.section("Identify BIN features"):
        bin_cols = [c for c in df_tr.columns if bin_suffix in c]

        BLACKLIST = [
            "quarter", "year", "month", "vintage", "date", "time",
            "maturity", "first_payment", "mi_cancellation",
            "interest_rate", "loan_sequence_number", "amortization_type",
            "window", "interest_only_indicator", "relief_refinance_indicator", 
            "super_conforming_flag",
        ]

        def safe(col: str) -> bool:
            raw = raw_name_from_bin(col, bin_suffix).lower()
            return not any(bad in raw for bad in BLACKLIST)

        bin_cols = [c for c in bin_cols if safe(c)]
        raw_to_bin = {raw_name_from_bin(c, bin_suffix): c for c in bin_cols}

    # Learn WOE on TRAIN and apply to TRAIN/VALIDATION
    with timer.section("Learn + Apply WOE"):
        woe_maps = build_woe_maps(df_tr, target, raw_to_bin)

        Xtr_full = apply_woe(df_tr, woe_maps, bin_suffix)
        ytr_full = df_tr[target].astype(int)

        Xva_full = apply_woe(df_va, woe_maps, bin_suffix)
        yva_full = df_va[target].astype(int)

        if use_interactions:
            Xtr_full = add_interactions(Xtr_full)
            Xva_full = add_interactions(Xva_full)

    # Calibration split (time-aware by default)
    with timer.section("Split for Calibration"):
        if calibration_split == "time_last":
            if cv_time_col not in meta_tr.columns:
                print(f"[WARN] calibration-split=time_last but '{cv_time_col}' is missing. Falling back to stratified.")
                X_model, X_cal, y_model, y_cal = _stratified_calibration_split(
                    Xtr_full, ytr_full, frac=calibration_size, seed=42
                )
            else:
                X_model, X_cal, y_model, y_cal = _time_aware_calibration_split(
                    Xtr_full,
                    ytr_full,
                    vintages=meta_tr[cv_time_col],
                    frac=calibration_size,
                    freq=cv_time_freq,
                )
        else:
            X_model, X_cal, y_model, y_cal = _stratified_calibration_split(
                Xtr_full, ytr_full, frac=calibration_size, seed=42
            )

    # Feature selection (correlation-based, greedy)
    with timer.section("Feature Selection"):
        kept_features = select_features(X_model, corr_thr=corr_threshold)

        X_model_kept = X_model[kept_features].to_numpy(dtype=np.float32, copy=False)
        X_cal_kept = X_cal[kept_features].to_numpy(dtype=np.float32, copy=False)
        Xtr_full_kept = Xtr_full[kept_features].to_numpy(dtype=np.float32, copy=False)
        Xva_full_kept = Xva_full[kept_features].to_numpy(dtype=np.float32, copy=False)

    # Cross-validation scheme for hyperparameter search (time-aware or stratified)
    with timer.section("Build CV splitter"):
        cv, groups = _build_cv_splitter(
            scheme=cv_scheme,
            X_model=X_model,
            y_model=y_model,
            meta_tr=meta_tr,
            time_col=cv_time_col,
            freq=cv_time_freq,
            cv_folds=cv_folds,
        )

    # Logistic regression hyperparameter search (C)
    with timer.section("Fit Logistic Regression (search)"):
        C_grid = np.logspace(c_min_exp, c_max_exp, c_num).tolist()

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
                verbose=search_verbose,
            )
        else:
            gs = GridSearchCV(
                lr,
                param_grid={"C": C_grid},
                scoring="neg_log_loss",
                cv=cv,
                n_jobs=-1,
                verbose=search_verbose,
            )

        if groups is not None:
            gs.fit(X_model_kept, y_model.to_numpy(), groups=groups)
        else:
            gs.fit(X_model_kept, y_model.to_numpy())

        best_lr: LogisticRegression = gs.best_estimator_

    # Coefficient statistics (optional; statsmodels if available)
    with timer.section("Coefficient statistics (optional)"):
        feature_names = ["Intercept"] + kept_features

        if coef_stats == "statsmodels":
            if sm is None:
                print("[WARN] statsmodels not available. Falling back to sklearn coefficients.")
                coefs = np.r_[best_lr.intercept_[0], best_lr.coef_[0]]
                coef_table = pd.DataFrame({
                    "feature": feature_names,
                    "coef": coefs,
                    "std_err": np.nan,
                    "z": np.nan,
                    "p_value": np.nan,
                })
            else:
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
                    print(f"[WARN] statsmodels failed ({type(e).__name__}: {e}). Falling back to sklearn coefficients.")
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

    # TTC score (raw score scaling from uncalibrated log-odds)
    with timer.section("Raw score scaling"):
        log_odds_tr_full = best_lr.decision_function(Xtr_full_kept)
        log_odds_va_full = best_lr.decision_function(Xva_full_kept)

        score_tr_full = scale_score(log_odds_tr_full)
        score_va_full = scale_score(log_odds_va_full)

    # PD calibration (isotonic/sigmoid) fitted on the calibration split
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

    # Summary metrics
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
        "cv_scheme": cv_scheme,
        "calibration_split": calibration_split,
        "use_interactions": bool(use_interactions),
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

    # Data
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--target", required=True)

    # I/O
    p.add_argument("--artifacts", default="artifacts/model_from_binned")
    p.add_argument("--scored-outdir", default="data/processed/scored")
    p.add_argument("--risk-buckets-in", default=None)

    # Feature engineering
    p.add_argument("--bin-suffix", default="__BIN")
    p.add_argument("--corr-threshold", type=float, default=0.85)
    p.add_argument(
        "--no-interactions",
        action="store_true",
        help="Disable WOE interactions (often more robust OOT)."
    )

    # Model / search
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--calibration", choices=["none", "isotonic", "sigmoid"], default="isotonic")

    p.add_argument(
        "--search",
        choices=["grid", "halving"],
        default="halving",
        help="grid=exhaustive, halving=successive halving (broad exploration + early pruning)."
    )
    p.add_argument("--c-min-exp", type=float, default=-8.0, help="C_min = 10^(c-min-exp)")
    p.add_argument("--c-max-exp", type=float, default=4.0, help="C_max = 10^(c-max-exp)")
    p.add_argument("--c-num", type=int, default=60, help="Number of points in the logspace(C) grid.")
    p.add_argument("--halving-factor", type=int, default=3)
    p.add_argument("--search-verbose", type=int, default=0)

    p.add_argument("--lr-solver", choices=["lbfgs", "saga", "newton-cg"], default="lbfgs")
    p.add_argument("--lr-max-iter", type=int, default=4000)

    # Time-aware CV + calibration split
    p.add_argument(
        "--cv-scheme",
        choices=["time", "stratified"],
        default="time",
        help="time=recommended for OOT (expanding window over vintages)."
    )
    p.add_argument(
        "--calibration-split",
        choices=["time_last", "stratified"],
        default="time_last",
        help="time_last=recommended (calibration on the last train vintages)."
    )
    p.add_argument("--calibration-size", type=float, default=0.20)

    p.add_argument("--cv-time-col", default="vintage")
    p.add_argument("--cv-time-freq", choices=["Q", "M"], default="Q")

    # statsmodels option
    p.add_argument(
        "--coef-stats",
        choices=["statsmodels", "none"],
        default="none",
        help="statsmodels=coef/std_err/z/p_value ; none=skip to speed up."
    )

    # Master scale
    p.add_argument("--n-buckets", type=int, default=10)
    p.add_argument(
        "--n-buckets-candidates",
        default=None,
        help="Optional: comma-separated list (e.g., '7,10,12,15') to pick the largest n_buckets under constraints."
    )
    p.add_argument("--min-bucket-count", type=int, default=300)
    p.add_argument("--min-bucket-bad", type=int, default=5)

    # TTC mode
    p.add_argument(
        "--ttc-mode",
        choices=["train", "train_val"],
        default="train",
        help="'train' = TTC PDs from TRAIN(window), 'train_val' = TTC PDs from TRAIN+VAL(window)."
    )

    # Windowing
    default_window = 5
    p.add_argument(
        "--grid-window-years",
        type=int,
        default=default_window,
        help="Window length (years) to build score edges, anchored at the last validation vintage."
    )
    p.add_argument("--grid-time-col", default="vintage")
    p.add_argument("--grid-time-freq", default="Q", choices=["Q", "M"])
    p.add_argument(
        "--ttc-window-years",
        type=int,
        default=default_window,
        help="Window length (years) to compute TTC PDs by grade, anchored at the last validation vintage."
    )

    # Misc
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
        search_verbose=args.search_verbose,
        lr_solver=args.lr_solver,
        lr_max_iter=args.lr_max_iter,
        coef_stats=args.coef_stats,
        use_interactions=not args.no_interactions,
        cv_scheme=args.cv_scheme,
        calibration_split=args.calibration_split,
        calibration_size=args.calibration_size,
        cv_time_col=args.cv_time_col,
        cv_time_freq=args.cv_time_freq,
    )

    # Save coefficient statistics
    coef_path = artifacts / "coefficients_stats.csv"
    out["coef_table"].to_csv(coef_path, index=False)
    print(f"✔ Coefficient stats saved: {coef_path}")

    # ------------------------------------------------------------------
    # MASTER SCALE (edges)
    # ------------------------------------------------------------------
    fixed_edges = None
    if args.risk_buckets_in and Path(args.risk_buckets_in).exists():
        fixed_edges = json.loads(Path(args.risk_buckets_in).read_text(encoding="utf-8"))["edges"]

    time_col = args.grid_time_col
    freq = args.grid_time_freq

    if fixed_edges is None:
        if time_col not in out["meta_va"].columns or time_col not in out["meta_tr"].columns:
            raise ValueError(f"Column '{time_col}' is missing in meta; cannot window the grid.")

        va_periods = parse_vintage_to_period(out["meta_va"][time_col], freq=freq)
        end_period = va_periods.max()

        mask_tr_grid = window_mask_from_end(out["meta_tr"][time_col], end_period, args.grid_window_years, freq=freq)
        mask_va_grid = window_mask_from_end(out["meta_va"][time_col], end_period, args.grid_window_years, freq=freq)

        scores_grid = np.concatenate([out["score_tr"][mask_tr_grid], out["score_va"][mask_va_grid]])
        y_grid = np.concatenate([out["y_tr"][mask_tr_grid], out["y_va"][mask_va_grid]])

        if len(y_grid) == 0:
            raise ValueError("Empty grid window. Check vintage/frequency and bounds.")

        # Optional: auto-select n_buckets under constraints
        n_buckets = int(args.n_buckets)
        if args.n_buckets_candidates:
            cands = [int(x.strip()) for x in args.n_buckets_candidates.split(",") if x.strip()]
            if not cands:
                raise ValueError("--n-buckets-candidates was provided but empty.")
            n_buckets = choose_n_buckets(
                scores_grid=scores_grid,
                y_grid=y_grid,
                candidates=cands,
                min_bucket_count=args.min_bucket_count,
                min_bucket_bad=args.min_bucket_bad,
            )
            print(f"[N_BUCKETS] auto-selected = {n_buckets} (candidates={cands})")
            args.n_buckets = n_buckets

        edges, stats_grid, mono_grid = create_risk_buckets(
            scores_grid, y_grid, n_buckets=args.n_buckets, fixed_edges=None
        )
        print(f"\n[GRID] window={args.grid_window_years}y | end={end_period} | monotone(grid)={'OK' if mono_grid else 'NO'}")
    else:
        edges = fixed_edges
        va_periods = parse_vintage_to_period(out["meta_va"][time_col], freq=freq)
        end_period = va_periods.max()
        print("\n[GRID] edges provided via --risk-buckets-in (fixed grid)")

    # Full TRAIN/VALIDATION bucket stats
    _, stats_tr_full, mono_tr = create_risk_buckets(out["score_tr"], out["y_tr"], n_buckets=args.n_buckets, fixed_edges=edges)
    _, stats_va_full, mono_va = create_risk_buckets(out["score_va"], out["y_va"], n_buckets=args.n_buckets, fixed_edges=edges)

    print("\n[METRICS]")
    for k, v in out["metrics"].items():
        if isinstance(v, float):
            print(f"  {k:22s} : {v:.6f}")
        else:
            print(f"  {k:22s} : {v}")

    print("\n[MONOTONICITY TRAIN] :", "OK" if mono_tr else "NOT MONOTONE")
    print("[MONOTONICITY VAL]   :", "OK" if mono_va else "NOT MONOTONE")

    # ------------------------------------------------------------------
    # TTC PD window
    # ------------------------------------------------------------------
    if time_col not in out["meta_va"].columns or time_col not in out["meta_tr"].columns:
        raise ValueError(f"Column '{time_col}' is missing; cannot compute TTC window.")

    va_periods = parse_vintage_to_period(out["meta_va"][time_col], freq=freq)
    end_period = va_periods.max()

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

    if args.ttc_mode == "train":
        stats_train_for_json = stats_tr_win
    else:
        stats_train_for_json = stats_longrun_win

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
        "train": stats_train_for_json.to_dict(orient="records"),

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
    print(f"✔ train_scored saved: {train_scored_path}")

    validation_scored = out["meta_va"].copy()
    validation_scored[args.target] = out["y_va"]
    validation_scored["score_ttc"] = out["score_va"]
    validation_scored["pd"] = out["pd_va"]
    validation_scored["grade"] = grade_va

    validation_scored_path = scored_dir / "validation_scored.parquet"
    validation_scored.to_parquet(validation_scored_path, index=False)
    print(f"✔ validation_scored saved: {validation_scored_path}")

    print(f"\nTTC PD mode used for 'train': {args.ttc_mode}")
    print(f"Grid window: {args.grid_window_years}y ending at {end_period} | TTC window: {args.ttc_window_years}y ending at {end_period}")

    if args.timing:
        print("\n[TIMING] Main sections:")
        for k, v in timer.records.items():
            print(f"  {k:40s} {v:.3f}s")


if __name__ == "__main__":
    main()
