#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, time, contextlib, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# -----------------------------
# I/O helpers
# -----------------------------
def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")

# -----------------------------
# Timing utils (overhead négligeable)
# -----------------------------
class Timer:
    def __init__(self, live: bool = False, stream=None):
        self.records: Dict[str, float] = {}
        self._stack: List[str] = []
        self.live = bool(live)
        self.stream = stream or sys.stdout

    @contextlib.contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        self._stack.append(name)
        if self.live:
            print(f"▶ {name} ...", file=self.stream, flush=True)
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.records[name] = self.records.get(name, 0.0) + dt
            self._stack.pop()
            if self.live:
                print(f"  ✓ {name:22s} {dt:8.3f}s", file=self.stream, flush=True)

    def add(self, name: str, dt: float):
        self.records[name] = self.records.get(name, 0.0) + dt

# -----------------------------
# Metrics & utils
# -----------------------------
def ks_best_threshold(y, p):
    y = pd.Series(y).astype(int).to_numpy()
    p = pd.Series(p).astype(float).to_numpy()
    if np.unique(y).size < 2:
        return np.nan, np.nan
    fpr, tpr, thr = roc_curve(y, p)
    ks_arr = tpr - fpr
    i = int(np.nanargmax(ks_arr))
    return float(ks_arr[i]), float(thr[i])

def decile_table(y, p, q=10):
    df = pd.DataFrame({"y": pd.Series(y).astype(int), "p": pd.Series(p).astype(float)})
    try:
        df["decile"] = pd.qcut(df["p"], q=q, labels=False, duplicates="drop")
    except Exception:
        n = len(df)
        ranks = df["p"].rank(method="first") / max(n, 1)
        df["decile"] = pd.cut(ranks, bins=np.linspace(0, 1, q + 1), labels=False, include_lowest=True)
    tab = (
        df.groupby("decile", dropna=True)
        .agg(events=("y", "sum"), count=("y", "size"), avg_p=("p", "mean"))
        .sort_index(ascending=False)
    )
    if tab.empty:
        return tab
    tab["rate"] = tab["events"] / tab["count"].where(tab["count"] > 0, 1)
    tab["cum_events"] = tab["events"].cumsum()
    tab["cum_count"] = tab["count"].cumsum()
    tot_e = float(tab["events"].sum())
    cum_good = tab["count"] - tab["events"]
    denom_good = float(cum_good.sum()) if float(cum_good.sum()) > 0 else 1.0
    tab["TPR"] = tab["cum_events"] / (tot_e if tot_e > 0 else 1.0)
    tab["FPR"] = cum_good.cumsum() / denom_good
    tab["KS"] = tab["TPR"] - tab["FPR"]
    return tab

def psi(a, b, bins=10, eps=1e-9):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    q = np.quantile(a, np.linspace(0, 1, bins + 1))
    q = np.unique(q)
    if q.size < 2:
        return 0.0
    q[0], q[-1] = -np.inf, np.inf
    for i in range(1, len(q)):
        if not (q[i] > q[i - 1]):
            q[i] = np.nextafter(q[i - 1], np.inf)
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    pa, pb = ca / max(ca.sum(), 1), cb / max(cb.sum(), 1)
    return float(np.sum((pa - pb) * np.log((pa + eps) / (pb + eps))))

def psi_by_feature(a, b, bins=10, eps=1e-9):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    q = np.quantile(a, np.linspace(0, 1, bins + 1))
    q = np.unique(q); q[0], q[-1] = -np.inf, np.inf
    for i in range(1, len(q)):
        if not (q[i] > q[i - 1]):
            q[i] = np.nextafter(q[i - 1], np.inf)
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    pa, pb = ca / max(ca.sum(), 1), cb / max(cb.sum(), 1)
    return float(np.sum((pa - pb) * np.log((pa + eps) / (pb + eps))))

def prior_shift_adjust(p, base_train, base_val, eps=1e-9):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    delta = np.log((base_val + eps) / (1 - base_val + eps)) - np.log((base_train + eps) / (1 - base_train + eps))
    z = logit + delta
    return 1 / (1 + np.exp(-z))

# -----------------------------
# BIN helpers (préfixe/suffixe)
# -----------------------------
def find_bin_columns(df: pd.DataFrame, tag: str) -> List[str]:
    cols = []
    for c in df.columns:
        if c.startswith(tag) or c.endswith(tag):
            cols.append(c)
    return sorted(cols)

def raw_name_from_bin(col: str, tag: str) -> str:
    if col.startswith(tag): return col[len(tag):]
    if col.endswith(tag):  return col[:-len(tag)]
    return col

def resolve_bin_col(df: pd.DataFrame, raw: str, tag: str) -> Optional[str]:
    cand_prefix = f"{tag}{raw}"
    cand_suffix = f"{raw}{tag}"
    if cand_prefix in df.columns: return cand_prefix
    if cand_suffix in df.columns: return cand_suffix
    return None

# -----------------------------
# WOE from BIN
# -----------------------------
def build_woe_maps_from_bins(
    df_enrichi: pd.DataFrame,
    target: str,
    raw_to_bin: Dict[str, str],
    smooth: float = 0.5
) -> Dict[str, Dict]:
    maps = {}
    y = df_enrichi[target].astype(int)
    B_all = float(y.sum())
    G_all = float(len(y) - y.sum())
    global_woe = float(np.log((B_all + smooth) / (G_all + smooth)))
    for raw, bcol in raw_to_bin.items():
        tab = df_enrichi.groupby(bcol, dropna=True)[target].agg(["sum", "count"])
        if tab.empty:
            maps[raw] = {"map": {}, "default": global_woe}
            continue
        tab["good"] = tab["count"] - tab["sum"]
        B = float(tab["sum"].sum())
        G = float(tab["good"].sum())
        K = max(int(len(tab)), 1)
        denom_bad  = (B + smooth * K) if (B + smooth * K) > 0 else 1.0
        denom_good = (G + smooth * K) if (G + smooth * K) > 0 else 1.0
        w = np.log(((tab["sum"] + smooth) / denom_bad) / ((tab["good"] + smooth) / denom_good)).replace([np.inf, -np.inf], np.nan)
        maps[raw] = {"map": {int(k) if pd.notna(k) else -9999: float(v) for k, v in w.items()}, "default": global_woe}
    return maps

def apply_woe_with_maps(
    df_any: pd.DataFrame,
    maps: Dict[str, Dict],
    kept_vars_raw: List[str],
    bin_tag: str
) -> pd.DataFrame:
    cols = []
    for raw in kept_vars_raw:
        bcol = resolve_bin_col(df_any, raw, bin_tag)
        if bcol is None or raw not in maps:
            continue
        ser = df_any[bcol].astype("Int64")
        wmap = maps[raw]["map"]
        wdef = float(maps[raw]["default"])
        x = ser.map(wmap).astype(float).fillna(wdef)
        cols.append((f"{raw}_WOE", x))
    if not cols:
        return pd.DataFrame(index=df_any.index)
    return pd.concat([s for _, s in cols], axis=1)

# -----------------------------
# Sélection par anti-colinéarité
# -----------------------------
def select_woe_columns(X_woe: pd.DataFrame, order_hint: List[str], corr_thr: float = 0.85) -> List[str]:
    cols = [c for c in order_hint if c in X_woe.columns] or list(X_woe.columns)
    corr = X_woe[cols].corr().abs().fillna(0.0)
    selected = []
    for c in cols:
        if not selected:
            selected.append(c)
            continue
        mc = corr.loc[c, corr.columns.intersection(selected)]
        max_corr = float(mc.max()) if len(mc) else 0.0
        if not np.isfinite(max_corr) or np.isnan(max_corr):
            max_corr = 0.0
        if max_corr < corr_thr:
            selected.append(c)
    return selected

# -----------------------------
# Entraînement principal + ablation gloutonne
# -----------------------------
def train_from_binned(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    target: str,
    use_existing_woe_prefixes: Tuple[str, ...] = ("woe__",),
    bin_suffix: str = "__BIN",
    corr_threshold: float = 0.85,
    cv_folds: int = 5,
    isotonic: bool = True,
    # options PSI/ablation
    drop_proxy_cutoff: Optional[float] = None,           # PSI(feature) cutoff
    conditional_proxies: Optional[List[str]] = None,     # raw var names
    do_prior_shift_adjust: bool = True,
    ablation_max_steps: int = 10,
    ablation_max_auc_loss: float = 0.02,
    timer: Optional[Timer] = None
) -> Dict[str, Any]:
    nullctx = contextlib.nullcontext()
    tctx = (lambda name: timer.section(name)) if timer else (lambda name: nullctx)

    # 1) Y
    with tctx("prep_y"):
        y_tr = df_tr[target].astype(int).values
        y_va = df_va[target].astype(int).values if target in df_va.columns else None

    # 2) WOE existants ?
    with tctx("detect_woe"):
        woe_cols = [c for c in df_tr.columns if any(c.startswith(p) for p in use_existing_woe_prefixes if p)]
        woe_cols += [c for c in df_tr.columns if c.endswith("_WOE")]
        woe_cols = sorted(set(woe_cols) - {target})

    computed_woe = False
    woe_maps = None

    if not woe_cols:
        # Sinon BIN -> WOE
        with tctx("detect_bin_cols"):
            bin_cols = [c for c in df_tr.columns if c.endswith(bin_suffix) or c.startswith(bin_suffix)]
        if not bin_cols:
            raise SystemExit("Aucune colonne WOE ni BIN détectée. Données attendues (__BIN ou préfixe) ou WOE.")
        with tctx("woe_build"):
            raw_to_bin = {raw_name_from_bin(c, bin_suffix): c for c in bin_cols}
            keep_raw = sorted(raw_to_bin.keys())
            woe_maps = build_woe_maps_from_bins(df_tr, target, raw_to_bin, smooth=0.5)
            Xtr_woe_full = apply_woe_with_maps(df_tr, woe_maps, keep_raw, bin_tag=bin_suffix)
            Xva_woe_full = apply_woe_with_maps(df_va, woe_maps, keep_raw, bin_tag=bin_suffix) if y_va is not None else None
            if Xva_woe_full is not None:
                Xva_woe_full = Xva_woe_full.reindex(columns=Xtr_woe_full.columns, fill_value=0.0)
            computed_woe = True
    else:
        with tctx("use_existing_woe"):
            Xtr_woe_full = df_tr[woe_cols].astype(float).copy()
            Xva_woe_full = df_va.reindex(columns=woe_cols).astype(float).fillna(0.0) if y_va is not None else None

    # 3) Drop proxies instables via PSI(feature)
    if drop_proxy_cutoff is not None and conditional_proxies and Xva_woe_full is not None:
        with tctx("psi_drop"):
            to_drop = []
            for raw in conditional_proxies:
                c = f"{raw}_WOE"
                if c in Xtr_woe_full.columns and c in Xva_woe_full.columns:
                    v = psi_by_feature(Xtr_woe_full[c], Xva_woe_full[c])
                    if v is not None and np.isfinite(v) and v > float(drop_proxy_cutoff):
                        to_drop.append(c)
            if to_drop:
                Xtr_woe_full = Xtr_woe_full.drop(columns=to_drop, errors="ignore")
                Xva_woe_full = Xva_woe_full.drop(columns=to_drop, errors="ignore")

    # 4) Sélection anti-colinéarité (ordre = variance décroissante)
    with tctx("feature_selection"):
        order_hint = list(Xtr_woe_full.var().sort_values(ascending=False).index)
        kept_woe = select_woe_columns(Xtr_woe_full, order_hint, corr_thr=corr_threshold)
        Xtr = Xtr_woe_full[kept_woe].copy()
        Xva = Xva_woe_full[kept_woe].copy() if Xva_woe_full is not None else None

    # 5) GridSearch + calibration
    with tctx("gridsearch"):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid = {"C": [0.1, 0.3, 1.0, 3.0, 10.0], "penalty": ["l2"], "solver": ["lbfgs"], "class_weight": [None], "max_iter": [2000]}
        base_lr = LogisticRegression()
        gs = GridSearchCV(base_lr, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True).fit(Xtr, y_tr)
        best_lr = gs.best_estimator_
    with tctx("calibration_fit"):
        model = CalibratedClassifierCV(best_lr, method=("isotonic" if isotonic else "sigmoid"), cv=cv).fit(Xtr, y_tr)

    with tctx("metrics_train"):
        p_tr = model.predict_proba(Xtr)[:, 1]
        metrics = {
            "train_auc": float(roc_auc_score(y_tr, p_tr)),
            "train_brier": float(brier_score_loss(y_tr, p_tr)),
            "train_logloss": float(log_loss(y_tr, p_tr)),
            "n_kept_features": int(len(kept_woe)),
        }

    dec = pd.DataFrame()
    p_va = None

    if Xva is not None and y_va is not None:
        with tctx("metrics_val"):
            p_va = model.predict_proba(Xva)[:, 1]
            ks, thr = ks_best_threshold(y_va, p_va)
            metrics.update({
                "val_auc": float(roc_auc_score(y_va, p_va)),
                "val_brier": float(brier_score_loss(y_va, p_va)),
                "val_logloss": float(log_loss(y_va, p_va)),
                "val_ks": float(ks),
                "val_ks_threshold": float(thr),
                "psi_train_to_val_proba": float(psi(p_tr, p_va, bins=10))
            })
            dec = decile_table(y_va, p_va, q=10)

    # 6) Ablation gloutonne (PSI(probas) ↓, perte AUC ≤ seuil)
    X_train_curr, X_val_curr = Xtr.copy(), (Xva.copy() if Xva is not None else None)
    model_curr, p_tr_curr, p_va_curr = model, p_tr, (p_va if p_va is not None else None)
    best_auc = float(roc_auc_score(y_va, p_va_curr)) if p_va_curr is not None else np.nan
    best_psi = float(psi(p_tr_curr, p_va_curr)) if p_va_curr is not None else np.nan
    keep_cols = list(X_train_curr.columns)

    if ablation_max_steps > 0 and p_va_curr is not None:
        with tctx("ablation"):
            for _ in range(min(ablation_max_steps, len(keep_cols))):
                psi_feat_now = pd.Series({c: psi_by_feature(X_train_curr[c], X_val_curr[c]) for c in keep_cols}).sort_values(ascending=False)
                cand = psi_feat_now.index[0]
                Xtr_try = X_train_curr.drop(columns=[cand])
                Xva_try = X_val_curr.drop(columns=[cand], errors="ignore")
                gs2 = GridSearchCV(best_lr, {**grid}, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True).fit(Xtr_try, y_tr)
                lr2 = gs2.best_estimator_
                cal2 = CalibratedClassifierCV(lr2, method=("isotonic" if isotonic else "sigmoid"), cv=cv).fit(Xtr_try, y_tr)
                p_tr2 = cal2.predict_proba(Xtr_try)[:, 1]
                p_va2 = cal2.predict_proba(Xva_try)[:, 1]
                auc2 = float(roc_auc_score(y_va, p_va2))
                psi2 = float(psi(p_tr2, p_va2))
                loss_auc = best_auc - auc2
                if (psi2 + 1e-6) < best_psi and (np.isnan(loss_auc) or loss_auc <= ablation_max_auc_loss):
                    X_train_curr, X_val_curr = Xtr_try, Xva_try
                    model_curr, p_tr_curr, p_va_curr = cal2, p_tr2, p_va2
                    best_auc, best_psi = auc2, psi2
                    keep_cols.remove(cand)
                else:
                    break

    # 7) Prior-shift adjust + metrics adj
    p_va_adj = None
    if p_va_curr is not None and do_prior_shift_adjust:
        with tctx("prior_shift_adjust"):
            base_train = float(np.mean(y_tr))
            base_val = float(np.mean(y_va))
            p_va_adj = prior_shift_adjust(p_va_curr, base_train, base_val)
            ks_adj, thr_adj = ks_best_threshold(y_va, p_va_adj)
            metrics.update({
                "val_auc_adj": float(roc_auc_score(y_va, p_va_adj)),
                "val_brier_adj": float(brier_score_loss(y_va, p_va_adj)),
                "val_logloss_adj": float(log_loss(y_va, p_va_adj)),
                "val_ks_adj": float(ks_adj),
                "val_ks_threshold_adj": float(thr_adj),
                "psi_train_to_val_proba_adj": float(psi(p_tr_curr, p_va_adj, bins=10))
            })
            dec = decile_table(y_va, p_va_adj, q=10) if dec.empty else dec

    # 8) Importance standardisée
    imp_df = None
    with tctx("importance"):
        lr_final = getattr(model_curr, "base_estimator", None) or getattr(model_curr, "estimator", None) or best_lr
        if hasattr(lr_final, "coef_"):
            beta = np.asarray(lr_final.coef_).ravel()
            feat_cols = list(X_train_curr.columns)
            if len(beta) != len(feat_cols):
                m = min(len(beta), len(feat_cols))
                beta = beta[:m]
                feat_cols = feat_cols[:m]
            stds = X_train_curr[feat_cols].std(ddof=0).replace(0, np.nan)
            std_coef = pd.Series(beta, index=feat_cols) * stds
            imp_df = (
                pd.DataFrame({"feature": feat_cols, "coef": beta, "std": stds.values, "std_coef": std_coef.values})
                .dropna(subset=["std_coef"])
                .sort_values("std_coef", key=lambda s: s.abs(), ascending=False)
            )

    out: Dict[str, Any] = {
        "model": model_curr,            # calibré après ablation (si faite)
        "best_lr": lr_final,
        "kept_woe": list(X_train_curr.columns),
        "computed_woe": bool(computed_woe),
        "woe_maps": woe_maps,           # None si WOE déjà fournis
        "metrics": metrics,
        "deciles_val": dec,
        "importance": imp_df,
        # sorties supplémentaires pour bucketing
        "p_tr_final": p_tr_curr,
        "p_va_final": p_va_adj if p_va_adj is not None else p_va_curr,
        "y_va": y_va,
    }
    return out

# -----------------------------
# Bucketing helpers (quantiles robustes)
# -----------------------------
def safe_quantile_edges(scores: np.ndarray, n: int = 10) -> np.ndarray:
    """Bornes strictement croissantes dans [0,1] à partir de quantiles."""
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    qs = np.linspace(0.0, 1.0, n + 1)
    edges = np.quantile(s, qs, method="linear")
    edges[0] = 0.0
    edges[-1] = 1.0
    for i in range(1, len(edges)):
        if not (edges[i] > edges[i-1]):
            edges[i] = np.nextafter(edges[i-1], 1.0)
    return edges

def assign_bucket(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Retourne des buckets 1..K-1 en utilisant les bornes `edges`."""
    inner = edges[1:-1]
    return (np.digitize(scores, inner, right=False) + 1).astype(int)

def bucket_stats(scores: np.ndarray, y: Optional[np.ndarray], edges: np.ndarray) -> pd.DataFrame:
    b = assign_bucket(scores, edges)
    dfb = pd.DataFrame({"bucket": b, "p": scores})
    if y is not None and len(y) == len(scores):
        dfb["y"] = y.astype(int)
        tab = (dfb.groupby("bucket", as_index=True)
                  .agg(count=("y", "size"),
                       events=("y", "sum"),
                       proba_mean=("p", "mean"))
                  .sort_index())
        tab["pd"] = tab["events"] / tab["count"].where(tab["count"] > 0, 1)
    else:
        tab = (dfb.groupby("bucket", as_index=True)
                 .agg(count=("p", "size"),
                      proba_mean=("p", "mean"))
                 .sort_index())
        tab["events"] = np.nan
        tab["pd"] = np.nan
    tab.reset_index(inplace=True)
    return tab

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Train depuis données binners (ou WOE) + calibration isotonic + ablation + prior-shift.")
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--artifacts", default="artifacts/model_from_binned")
    p.add_argument("--bin-suffix", default="__BIN")     # tag (préfixe OU suffixe)
    p.add_argument("--woe-prefixes", default="woe__")   # séparés par virgule si plusieurs
    p.add_argument("--corr-threshold", type=float, default=0.85)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--no-isotonic", action="store_true")
    # PSI / ablation / prior-shift
    p.add_argument("--drop-proxy-cutoff", type=float, default=None)
    p.add_argument("--conditional-proxies", default="")
    p.add_argument("--no-prior-shift-adjust", action="store_true")
    p.add_argument("--ablation-max-steps", type=int, default=10)
    p.add_argument("--ablation-max-auc-loss", type=float, default=0.02)
    # timing
    p.add_argument("--timing", action="store_true", help="Sauve timings.json + résumé final en stdout")
    p.add_argument("--timing-live", action="store_true", help="Affiche le timing de chaque section en live")
    # bucketing
    p.add_argument("--n-buckets", type=int, default=10, help="Nombre de classes risque (quantiles)")
    return p.parse_args()

def main():
    args = parse_args()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    timer = Timer(live=args.timing_live) if (args.timing or args.timing_live) else None
    nullctx = contextlib.nullcontext()
    tctx = (lambda name: timer.section(name)) if timer else (lambda name: nullctx)

    with tctx("load_data"):
        df_tr = load_any(args.train)
        df_va = load_any(args.validation)
        if args.target not in df_tr.columns or args.target not in df_va.columns:
            raise SystemExit(f"Target '{args.target}' absente de train/validation.")

    with tctx("parse_args"):
        prefixes = tuple([s.strip() for s in args.woe_prefixes.split(",") if s.strip()])
        cond_proxies = [s.strip() for s in args.conditional_proxies.split(",") if s.strip()] if args.conditional_proxies else None

    with tctx("train_from_binned"):
        out = train_from_binned(
            df_tr=df_tr, df_va=df_va, target=args.target,
            use_existing_woe_prefixes=prefixes, bin_suffix=args.bin_suffix,
            corr_threshold=args.corr_threshold, cv_folds=args.cv_folds, isotonic=(not args.no_isotonic),
            drop_proxy_cutoff=args.drop_proxy_cutoff, conditional_proxies=cond_proxies,
            do_prior_shift_adjust=not args.no_prior_shift_adjust,
            ablation_max_steps=int(args.ablation_max_steps), ablation_max_auc_loss=float(args.ablation_max_auc_loss),
            timer=timer
        )

    # ---------------- Buckets (bornes + stats) ----------------
    with tctx("save_buckets"):
        # base = probas de validation (après prior-shift si calculé)
        p_tr_final = np.asarray(out.get("p_tr_final", []), float)
        p_va_final = out.get("p_va_final", None)
        y_va = out.get("y_va", None)
        if p_va_final is None or (isinstance(p_va_final, np.ndarray) and p_va_final.size == 0):
            # fallback si pas de validation exploitable
            base_scores = p_tr_final
        else:
            base_scores = np.asarray(p_va_final, float)

        edges = safe_quantile_edges(base_scores, n=args.n_buckets)
        # stats sur validation si y dispo, sinon sur base_scores sans PD
        stats_df = bucket_stats(
            scores=(np.asarray(p_va_final, float) if p_va_final is not None else base_scores),
            y=(np.asarray(y_va, int) if y_va is not None else None),
            edges=edges
        )
        save_json({"edges": edges.tolist()}, artifacts / "risk_buckets.json")
        save_json({
            "n_buckets": int(len(edges) - 1),
            "edges": edges.tolist(),
            "by_bucket": stats_df.to_dict(orient="records")
        }, artifacts / "bucket_stats.json")
        print(f"✔ Buckets sauvés → {artifacts/'risk_buckets.json'}")
        print(f"✔ Stats buckets → {artifacts/'bucket_stats.json'}")

    with tctx("save_artifacts"):
        dump({
            "model": out["model"],
            "kept_woe": out["kept_woe"],
            "computed_woe": out["computed_woe"],
            "woe_maps": out["woe_maps"],
            "target": args.target
        }, artifacts / "model_best.joblib")

        pd.DataFrame([out["metrics"]]).to_csv(artifacts / "reports.csv", index=False)

        save_json({
            "train_path": str(Path(args.train).resolve()),
            "validation_path": str(Path(args.validation).resolve()),
            "artifacts_dir": str(artifacts.resolve()),
            "target": args.target,
            "bin_suffix": args.bin_suffix,
            "woe_prefixes": prefixes,
            "cv_folds": int(args.cv_folds),
            "corr_threshold": float(args.corr_threshold),
            "isotonic": bool(not args.no_isotonic),
            "kept_woe": out["kept_woe"],
            "computed_woe": bool(out["computed_woe"]),
            "drop_proxy_cutoff": args.drop_proxy_cutoff,
            "conditional_proxies": cond_proxies,
            "prior_shift_adjust": bool(not args.no_prior_shift_adjust),
            "ablation_max_steps": int(args.ablation_max_steps),
            "ablation_max_auc_loss": float(args.ablation_max_auc_loss),
            "metrics": out["metrics"],
            "n_buckets": int(args.n_buckets)
        }, artifacts / "meta.json")

        if isinstance(out["deciles_val"], pd.DataFrame) and not out["deciles_val"].empty:
            out["deciles_val"].to_csv(artifacts / "deciles_val.csv", index=False)
        if isinstance(out["importance"], pd.DataFrame) and not out["importance"].empty:
            out["importance"].to_csv(artifacts / "importance.csv", index=False)

    # Affichage + export timings
    if timer:
        total = sum(timer.records.values())
        print("\n⏱ Timings (s):")
        for k, v in sorted(timer.records.items(), key=lambda kv: -kv[1]):
            pct = (v / total * 100.0) if total > 0 else 0.0
            print(f"  - {k:22s} {v:8.3f}s  ({pct:5.1f}%)")
        save_json({"timings_seconds": timer.records, "total_seconds": total}, artifacts / "timings.json")

    print("✔ Saved:", artifacts / "model_best.joblib")
    print("✔ Saved:", artifacts / "reports.csv")
    print("✔ Saved:", artifacts / "meta.json")
    if (artifacts / "deciles_val.csv").exists():
        print("✔ Saved:", artifacts / "deciles_val.csv")
    if (artifacts / "importance.csv").exists():
        print("✔ Saved:", artifacts / "importance.csv")
    if (artifacts / "risk_buckets.json").exists():
        print("✔ Saved:", artifacts / "risk_buckets.json")
    if (artifacts / "bucket_stats.json").exists():
        print("✔ Saved:", artifacts / "bucket_stats.json")

if __name__ == "__main__":
    main()
