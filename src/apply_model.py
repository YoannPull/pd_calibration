#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply a persisted model (joblib) to a new dataset and optionally:
- append risk class (bucket) using precomputed score edges,
- compute and save OOS performance metrics when the target is present.

Examples
--------
# Proba only
python src/apply_model.py \
  --data data/processed/binned/test.parquet \
  --model artifacts/model/model_best.joblib \
  --out data/processed/scored/test_scored.parquet

# Proba + risk class + OOS metrics
python src/apply_model.py \
  --data data/processed/binned/oos.parquet \
  --model artifacts/model/model_best.joblib \
  --out data/processed/scored/oos_scored.parquet \
  --buckets artifacts/model/risk_buckets.json \
  --bucket-col risk_bucket \
  --metrics-out reports/model/oos_metrics.json
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Any, List

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
    """
    Expect a JSON containing at least: {"edges": [e0, e1, ..., ek]} with e0≈0, ek≈1, increasing.
    Returns numpy array of edges.
    """
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
    """
    Map scores in [0,1] to bucket indices using provided edges.
    - If edges has length K, there are K-1 buckets.
    - Uses np.digitize on inner edges (exclude the first and last bound).
    Returns integer buckets in [1..K-1] (1-based, convenient for reporting).
    """
    if len(edges) < 2:
        return np.ones_like(scores, dtype=int)

    inner = edges[1:-1]  # split points
    # bucket 0..K-2 then +1 to get 1..K-1
    b = np.digitize(scores, inner, right=False) + 1
    return b.astype(int)


# ---------------- Metrics ----------------
def compute_oos_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """
    Robust metrics computation. Returns None for metrics that cannot be computed.
    """
    out: Dict[str, float] = {}
    # base stats
    out["n"] = int(len(y_true))
    out["base_rate"] = float(np.mean(y_true)) if len(y_true) else None
    out["proba_mean"] = float(np.mean(proba)) if len(proba) else None
    out["proba_std"] = float(np.std(proba)) if len(proba) else None

    # clip probabilities for logloss stability
    proba_clip = np.clip(proba, 1e-15, 1 - 1e-15)

    # AUC
    try:
        out["auc"] = float(roc_auc_score(y_true, proba))
    except Exception:
        out["auc"] = None

    # Brier
    try:
        out["brier"] = float(brier_score_loss(y_true, proba))
    except Exception:
        out["brier"] = None

    # LogLoss
    try:
        out["logloss"] = float(log_loss(y_true, proba_clip))
    except Exception:
        out["logloss"] = None

    return out


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Apply a saved model to a new dataset (with optional bucketing + OOS metrics)")
    p.add_argument("--data", required=True, help="CSV/Parquet with same features used in training")
    p.add_argument("--model", required=True, help="Path to artifacts/model_best.joblib")
    p.add_argument("--out", required=True, help="Output path (CSV/Parquet)")
    # segmentation
    p.add_argument("--buckets", default=None, help="Path to risk_buckets.json (with 'edges')")
    p.add_argument("--bucket-col", default="risk_bucket", help="Name of the output bucket column (if --buckets is given)")
    # metrics
    p.add_argument("--metrics-out", default=None, help="Path to save OOS metrics JSON (if target is available)")
    return p.parse_args()


# ---------------- Main ----------------
def main():
    args = parse_args()

    # Load model bundle
    bundle = load(args.model)  # dict: {"model": estimator, "features": [...], "target": "..."}
    model = bundle["model"]
    features: List[str] = bundle["features"]
    target_col = bundle.get("target", None)

    # Load data and check features
    df = load_any(args.data)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature(s) in data: {missing}")

    # Predict probabilities
    proba = model.predict_proba(df[features])[:, 1]
    out = df.copy()
    out["proba"] = proba

    # Optional: risk bucketing
    if args.buckets:
        edges = load_buckets(args.buckets)
        out[args.bucket_col] = assign_bucket(out["proba"].values, edges)

    # Optional: OOS metrics if target is present
    metrics_path = args.metrics_out
    if target_col and (target_col in df.columns):
        y = df[target_col].astype(int).values
        metrics = compute_oos_metrics(y, proba)

        # auto path if not provided
        if metrics_path is None:
            out_path = Path(args.out)
            metrics_path = str(out_path.with_suffix("")) + "_metrics.json"

        save_json(metrics, metrics_path)
        print(f"✔ OOS metrics saved to: {metrics_path}")
    else:
        if args.metrics_out:
            print(f"[WARN] --metrics-out provided but target column '{target_col}' not found in data. No metrics saved.", file=sys.stderr)

    # Save scored dataset
    save_any(out, args.out)
    print(f"✔ Predictions saved to: {args.out}")

    # Small tail: if segmentation used, print a quick distribution
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
