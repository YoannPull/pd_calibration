#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply a persisted model (joblib) to a new dataset and optionally:
- append risk class (bucket) using precomputed score edges,
- compute and save OOS performance metrics when the target is present.

This version is robust to two training setups:
  A) Model trained on existing WOE columns (e.g., prefix 'woe__' or suffix '_WOE')
  B) Model trained after computing WOE from BIN columns (__BIN). In that case,
     we rely on 'woe_maps' and 'kept_woe' stored in the joblib bundle.

Outputs:
  - Scored file with 'proba' (+ optional bucket col)
  - Optional OOS metrics JSON if target is present
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


# ---------------- I/O helpers ----------------
def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_any(df: pd.DataFrame, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)


def save_json(obj: Dict[str, Any], path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=float))


# ---------------- Bucketing ----------------
def load_buckets(path: str) -> np.ndarray:
    spec = json.loads(Path(path).read_text())
    if "edges" not in spec:
        raise ValueError(f"Bucket file '{path}' must contain an 'edges' array.")
    edges = np.asarray(spec["edges"], dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("Invalid 'edges': need at least 2 values.")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("'edges' must be strictly increasing.")
    return edges


def assign_bucket(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(edges) < 2:
        return np.ones_like(scores, dtype=int)
    inner = edges[1:-1]
    b = np.digitize(scores, inner, right=False) + 1  # 1..K-1
    return b.astype(int)


# ---------------- Metrics ----------------
def compute_oos_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["n"] = int(len(y_true))
    out["base_rate"] = float(np.mean(y_true)) if len(y_true) else None
    out["proba_mean"] = float(np.mean(proba)) if len(proba) else None
    out["proba_std"] = float(np.std(proba)) if len(proba) else None

    proba_clip = np.clip(proba, 1e-15, 1 - 1e-15)
    try:
        out["auc"] = float(roc_auc_score(y_true, proba))
    except Exception:
        out["auc"] = None
    try:
        out["brier"] = float(brier_score_loss(y_true, proba))
    except Exception:
        out["brier"] = None
    try:
        out["logloss"] = float(log_loss(y_true, proba_clip))
    except Exception:
        out["logloss"] = None
    return out


# ---------------- Minimal WOE utils (for scoring when model learned WOE from BIN) ----------------
def resolve_bin_col(df: pd.DataFrame, raw: str, bin_tag: str) -> Optional[str]:
    pref = f"{bin_tag}{raw}"
    suff = f"{raw}{bin_tag}"
    if pref in df.columns:
        return pref
    if suff in df.columns:
        return suff
    return None


def apply_woe_with_maps_for_scoring(
    df_any: pd.DataFrame,
    woe_maps: Dict[str, Dict],
    kept_vars_raw: List[str],
    bin_tag: str
) -> pd.DataFrame:
    """
    Rebuild WOE columns expected by the trained model from BIN columns using saved maps.
    Creates columns <raw>_WOE for each raw in kept_vars_raw found in df_any.
    Missing BIN values map to the global default WOE stored in woe_maps[raw]["default"].
    """
    cols: List[pd.Series] = []
    names: List[str] = []
    for raw in kept_vars_raw:
        if raw not in woe_maps:
            continue
        bcol = resolve_bin_col(df_any, raw, bin_tag)
        if bcol is None or bcol not in df_any.columns:
            continue
        ser = df_any[bcol].astype("Int64")  # bin index
        wmap = woe_maps[raw]["map"]           # dict {bin_idx -> woe}
        wdef = float(woe_maps[raw]["default"])
        w = ser.map(wmap).astype(float).fillna(wdef)
        cols.append(w)
        names.append(f"{raw}_WOE")
    if not cols:
        return pd.DataFrame(index=df_any.index)
    out = pd.concat(cols, axis=1)
    out.columns = names
    return out


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Apply a saved model to a new dataset (with optional bucketing + OOS metrics)")
    p.add_argument("--data", required=True, help="CSV/Parquet with same features used in training or the BIN columns used to derive WOE")
    p.add_argument("--model", required=True, help="Path to artifacts/model_best.joblib")
    p.add_argument("--out", required=True, help="Output path (CSV/Parquet)")
    # segmentation
    p.add_argument("--buckets", default=None, help="Path to risk_buckets.json (with 'edges')")
    p.add_argument("--bucket-col", default="risk_bucket", help="Name of the output bucket column (if --buckets is given)")
    # metrics
    p.add_argument("--metrics-out", default=None, help="Path to save OOS metrics JSON (if target is available)")
    # override for BIN tag if needed (must match training)
    p.add_argument("--bin-suffix", default="__BIN", help="BIN tag used at training to reconstruct WOE if needed")
    return p.parse_args()


# ---------------- Main ----------------
def main():
    args = parse_args()

    # Load model bundle
    bundle = load(args.model)
    model = bundle["model"]

    # Training metadata saved by train_model.py
    target_col: Optional[str] = bundle.get("target", None)
    kept_woe: Optional[List[str]] = bundle.get("kept_woe", None)  # list of feature names actually used by the model
    computed_woe: bool = bool(bundle.get("computed_woe", False))  # True if WOE were computed from BIN at training
    woe_maps: Optional[Dict[str, Dict]] = bundle.get("woe_maps", None)

    # Backward-compat: allow an explicit "features" field if present
    features: Optional[List[str]] = bundle.get("features", None)
    if features is None:
        features = kept_woe

    if not features:
        raise SystemExit("Model bundle does not contain 'features' or 'kept_woe' — cannot determine input columns.")

    # Load data
    df_raw = load_any(args.data)

    # Prepare feature matrix:
    #  - if model expects WOE and we TRAINED with computed_woe=True, rebuild WOE from BIN using woe_maps
    #  - else, assume features are already present in df_raw
    if computed_woe:
        if not isinstance(woe_maps, dict):
            raise SystemExit("Model bundle indicates 'computed_woe=True' but no 'woe_maps' were found.")
        # Derive the underlying raw variable names from kept_woe = ["<raw>_WOE", ...]
        kept_raw = [c[:-4] if c.endswith("_WOE") else c for c in features]
        X_woe = apply_woe_with_maps_for_scoring(df_raw, woe_maps, kept_raw, bin_tag=args.bin_suffix)
        missing = [c for c in features if c not in X_woe.columns]
        if missing:
            raise SystemExit(f"Cannot rebuild required WOE features from BIN: missing {missing}. "
                             f"Check BIN columns and --bin-suffix (got '{args.bin_suffix}').")
        X = X_woe[features].copy()
    else:
        # Expect features already in df (e.g., existing WOE columns were used at training)
        missing = [f for f in features if f not in df_raw.columns]
        if missing:
            raise SystemExit(f"Missing feature(s) in data: {missing}")
        X = df_raw[features].copy()

    # Predict probabilities
    proba = model.predict_proba(X)[:, 1]
    out = df_raw.copy()
    out["proba"] = proba

    # Optional bucketing
    if args.buckets:
        edges = load_buckets(args.buckets)
        out[args.bucket_col] = assign_bucket(out["proba"].values, edges)

    # Optional OOS metrics if target present
    metrics_path = args.metrics_out
    if target_col and (target_col in df_raw.columns):
        y = df_raw[target_col].astype(int).values
        metrics = compute_oos_metrics(y, proba)
        if metrics_path is None:
            out_path = Path(args.out)
            metrics_path = str(out_path.with_suffix("")) + "_metrics.json"
        save_json(metrics, metrics_path)
        print(f"✔ OOS metrics saved to: {metrics_path}")
    elif args.metrics_out:
        print(f"[WARN] --metrics-out provided but target column '{target_col}' not found in data — no metrics saved.")

    # Save
    save_any(out, args.out)
    print(f"✔ Predictions saved to: {args.out}")

    # Small tail: bucket distribution
    if args.buckets:
        try:
            counts = out[args.bucket_col].value_counts().sort_index()
            print("Risk bucket distribution:")
            for k, v in counts.items():
                print(f"  bucket {k}: {v}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
